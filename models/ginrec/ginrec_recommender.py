import inspect
from collections import defaultdict

import dgl
import numpy as np
import torch
from dgl.dataloading import DataLoader, as_edge_prediction_sampler
from loguru import logger
from torch import optim
from tqdm import tqdm

from models.dgl_recommender_base import RecommenderBase
from models.ginrec.ginrec import GInRec
from shared.efficient_validator import Validator
from shared.enums import Sentiment, RecommenderEnum
from shared.graph_utility import UniformRecommendableItemSampler
from shared.meta import Meta
from shared.utility import is_debug_mode


class GInRecRecommender(RecommenderBase):
    def __init__(self, meta: Meta, attention=None, relations=True, activation=True, vae=False,
                 bipartite=False, n_layer_samples=None, use_uva=False, **kwargs):
        features = kwargs.pop('features', True)  # Ignore
        super().__init__(RecommenderEnum.FULL_COLD_START, meta, **kwargs, features=True)
        self.use_uva = use_uva
        self.attention = attention
        self.relations = relations
        self.n_entities = len(meta.entities)
        self.user_fn = lambda u: u + self.n_entities
        self.n_users = len(meta.users)
        self.activation = activation
        self.bipartite = bipartite
        self.lr = 0.001
        self.weight_decay = 1e-5
        self.n_layers = 3
        self.autoencoder_layer_dims = [128, 64]
        self.layer_dims = [64, 32, 16, 8, 8, 8]
        self.use_out_layer = False
        self.autoencoder_weight = 0.001
        self.dropouts = [0.1, 0.1, 0.1]
        self.gate_type = 'concat'
        self.normalization = None
        self.aggregator = 'gcn'
        self.l2_loss = 0
        self.n_layer_samples = [10, 10, 10] if n_layer_samples is None else n_layer_samples
        self.feature_dim = None
        self.attention_dim = None if self.attention is None else 32
        self.neg_sampling_method = None
        self.shuffle = True
        self.grad_clip_norm = 5.
        self.optimizer_name = 'adam'

        self.optimizer = None
        self.entity_embeddings = None
        self.user_embeddings = None
        self.multi_features = False

        self.end_types = set()

    def _get_features(self):
        train_e, test_e = self.entity_embeddings

        e = train_e if self._model.training else test_e
        u = self.user_embeddings

        f = torch.cat([e.cpu(), u.cpu()])

        return f

    def _create_model(self, trial):
        self.set_seed()
        n_users = len(self.meta.users)
        (train_g, test_g), = self.graphs
        n_relations = len(self.infos[0][0])

        tanh_range = True
        
        # Load features and add users.
        self.feature_dim = self._features[0].shape[-1]
        self.entity_embeddings = [torch.FloatTensor(np.array(f)) for f in self._features]
        self.multi_features = False

        # if features in range -1 to 1
        min_ = min([e.min() for e in self.entity_embeddings])
        max_ = max([e.max() for e in self.entity_embeddings])
        tanh_range = tanh_range and min_ >= -1 and max_

        self.user_embeddings = torch.zeros(n_users, self.feature_dim, requires_grad=False)

        self._model = GInRec(n_relations, self.feature_dim, self.user_fn, n_edges=train_g.number_of_edges(),
                             device=self.device,
                             autoencoder_layer_dims=self.autoencoder_layer_dims, dropouts=self.dropouts,
                             gate_type=self.gate_type, aggregator=self.aggregator, attention=self.attention,
                             attention_dim=self.attention_dim, relations=self.relations,
                             dimensions=self.layer_dims[:self.n_layers], activation=self.activation,
                             tanh_range=tanh_range, use_out_layer=self.use_out_layer)

        optimizer_kwargs = {'lr': self.lr, 'weight_decay': self.weight_decay}
        opt = getattr(optim, self.optimizer_name)
        args = inspect.signature(opt).parameters
        self._optimizer = opt(self._model.parameters(), **{k: v for k, v in optimizer_kwargs.items() if k in args})

        if self.use_cuda:
            self._model = self._model.cuda()

        self._sherpa_load(trial)

    def _get_graph(self, graph_type='train'):
        pos_relation_id = self.infos[0][0][Sentiment.POSITIVE.name]
        (train_g, test_g), = self.graphs

        if graph_type == 'train':
            g = train_g
        elif graph_type == 'test':
            g = test_g
        else:
            raise NotImplementedError(f'Graph type {graph_type} not implemented')

        # Assign node type, user=0, item=1, entity=2
        g.ndata['type'] = torch.zeros(g.number_of_nodes(), dtype=torch.long)
        g.ndata['type'][torch.LongTensor(self.meta.entities)] = 2
        g.ndata['type'][torch.LongTensor(self.meta.items)] = 1

        if self.bipartite:
            pos_rev = pos_relation_id + len(self.infos[0]) // 2
            mask = torch.logical_or(g.edata['type'] == pos_relation_id, g.edata['type'] == pos_rev)
            eids = g.edges('eid')[mask]
            g = g.edge_subgraph(eids, relabel_nodes=False)

        return g

    def fit(self, validator: Validator):
        super(GInRecRecommender, self).fit(validator)

        with torch.no_grad():
            self._model.eval()
            g = self._get_graph('test')
            features = self._get_features()
            if self.multi_features:
                features = {k.name: v.to(self.device) for k, v in features.items()}
            else:
                features = features.to(self.device)

            self._inference(g=g, embeddings=features)
            logger.info(validator.validate(self))

    def _fit(self, validator: Validator, first_epoch, final_epoch=1000, trial=None):
        # # Get relevant information
        pos_relation_id = self.infos[0][0][Sentiment.POSITIVE.name]
        g = self._get_graph()
        test_g = self._get_graph('test')
        features = self._get_features()
        test_features = None

        features = features.to(self.device)

        # Get positive relations.
        mask = g.edata['type'] == pos_relation_id
        eids = g.edges('eid')[mask]

        # In graph creation reverse relation are added in same order after all original relations. Reverse eids can
        # therefore be defined as in DGL documentation.
        n_edges = g.number_of_edges()
        reverse_eids = torch.cat([torch.arange(n_edges // 2, n_edges), torch.arange(0, n_edges // 2)])

        prob = 'edge_probability'

        # Create cf dataloader
        bs = self.batch_size

        n_layer_samples = self.n_layer_samples
        cf_sampler = dgl.dataloading.MultiLayerNeighborSampler(n_layer_samples, prefetch_edge_feats=['type'],
                                                               prob=prob)

        neg_sampler = UniformRecommendableItemSampler(1, edges=g.find_edges(eids),
                                                      methodology=self.neg_sampling_method,
                                                      share_negative=False)
        cf_sampler = as_edge_prediction_sampler(
            cf_sampler, exclude='reverse_id', reverse_eids=reverse_eids,
            negative_sampler=neg_sampler)

        dataloader = DataLoader(
            g, eids, cf_sampler, device=self.device, batch_size=bs,
            shuffle=self.shuffle, drop_last=True, use_uva=self.use_uva, num_workers=self.workers)

        n_iter = len(dataloader)
        for e in range(first_epoch, final_epoch):

            self._model.train()
            if self._no_improvements < self._early_stopping:
                tot_losses = defaultdict(float)
                with tqdm(dataloader, disable=not is_debug_mode()) as progress:
                    for i, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(progress, 1):
                        cur_losses = defaultdict(float)
                        input_features = features[input_nodes]

                        input_features, ae_loss = self._model.loss_ae(input_features, blocks[0].srcdata['type'])

                        cf_loss, l2_loss, correct = \
                            self._model.loss(positive_graph, negative_graph, blocks, input_features)

                        ae_loss = self.autoencoder_weight * ae_loss
                        l2_loss = self.l2_loss * l2_loss
                        if self.autoencoder_weight != 0:
                            cur_losses['loss/ael'] += ae_loss.detach().item()
                        if self.l2_loss != 0:
                            cur_losses['loss/l2l'] += l2_loss.detach().item()

                        loss = cf_loss + ae_loss + l2_loss

                        loss.backward()
                        if self.grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.grad_clip_norm)

                        self._optimizer.step()
                        self._optimizer.zero_grad()

                        cf_loss = cf_loss.detach()
                        acc = (torch.sum(correct) / correct.shape.numel()).detach()
                        cur_losses['loss/cfl'] += cf_loss.item()
                        cur_losses['loss/acc'] += acc.item()

                        step = e * n_iter + i
                        for k, v in cur_losses.items():
                            tot_losses[k] += v
                            self.summary_writer.add_scalar(k, v, step)

                        string = ', '.join([f'{k.split("/", 1)[-1]}: {v / i:.5f}' for k, v in tot_losses.items()])
                        progress.set_description(f'Epoch {e}, ' + string)
            elif trial is None:  # Skip last iterations as irrelevant
                break

            if test_features is None:
                self._model.eval()  # Set model to eval mode thus get features returns test features.
                test_features = self._get_features().to(self.device)

            self._to_report(first_epoch, final_epoch, e, validator, trial, g=test_g, embeddings=test_features)

    def _inference(self, **kwargs):
        self._model.eval()
        with torch.no_grad():
            self._model.inference(**kwargs, batch_size=max(1024, self.batch_size // 8))

    def predict_all(self, users) -> np.array:
        with torch.no_grad():
            user_t = torch.LongTensor(users) + len(self.meta.entities)
            items = torch.LongTensor(self.meta.items)
            preds = self._model(user_t, items, rank_all=True, apply_user_fn=False)

        return preds.cpu().numpy()

    def set_parameters(self, parameters):
        self.n_layers = parameters.get('layers', 3)
        self.lr = parameters['learning_rates']
        self.dropouts = [parameters['dropouts']] * self.n_layers
        self.gate_type = parameters['gate_types']
        self.aggregator = parameters['aggregators']
        self.autoencoder_weight = parameters['autoencoder_weights']
        self.weight_decay = parameters.get('weight_decays', 0.)
        if self.n_layers > len(self.n_layer_samples):
            self.n_layer_samples += [self.n_layer_samples[-1]] * (self.n_layers - len(self.n_layer_samples))
        elif self.n_layers < len(self.n_layer_samples):
            self.n_layer_samples = self.n_layer_samples[:self.n_layers]
        self.kl_weight = parameters.get('kl_weights', 0.1)
        self.l2_loss = parameters.get('l2_loss', 0)
        self.neg_sampling_method = parameters.get('neg_sampling_methods', 'uniform')
        self.normalization = parameters.get('normalizations', 'none')
        self.batch_size = parameters.get('batch_sizes', self.batch_size)
        self.grad_clip_norm = parameters.get('grad_clip_norms', 5.)
        self.optimizer_name = parameters.get('optimizers', 'Adam')
        self.shuffle = parameters.get('shuffle', True)
        self.use_out_layer = parameters.get('use_out_layer', False)

    def invalid_configuration(self, parameters):
        return False
