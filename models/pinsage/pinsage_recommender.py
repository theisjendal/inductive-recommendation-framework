import cProfile

import dgl
import numpy as np
import torch
from dgl.dataloading import DataLoader
from loguru import logger
from tqdm import tqdm

from models.dgl_recommender_base import RecommenderBase
from models.pinsage.pinsage import PinSAGE
from models.shared.dgl_dataloader import PinSAGESampler, HhopPredictionSampler, PinSAGESampler2
from models.utility import construct_user_item_heterogeneous_graph
from shared.enums import RecommenderEnum
from shared.graph_utility import UniformRecommendableItemSampler
from shared.efficient_validator import Validator
from shared.utility import is_debug_mode


class PinSAGERecommender(RecommenderBase):
    def __init__(self, bpr=False, **kwargs):
        super().__init__(RecommenderEnum.FULL_COLD_START, **kwargs)
        self._model = None
        self._model_dict = None
        self.lr = 1.e-4
        self.layer_dims = [64, 64, 64]
        self.dropout_prop = 0.2
        self.weight_decay = 1e-5
        self.delta = 1.
        self.num_traversals = 3
        self.termination_prob = 0.5,
        self.num_random_walks = 200
        self.num_neighbors = 10
        self.bpr = bpr

        self.sampler_args = None

        self.feature_dim = 0
        self.entity_features = None
        self.users_features = None

    def _create_model(self, trial):
        self._model = PinSAGE(self.feature_dim, self.layer_dims, self.delta, self.dropout_prop)
        if self.use_cuda:
            self._model = self._model.cuda()

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self._sherpa_load(trial)

    def _inference(self, **kwargs):
        _, (g, h) = self.graphs
        self._model.eval()
        with torch.no_grad():
            if self.bpr:
                self._model.inference(g, g.nodes().to(self.device), self.sampler_args, self.bpr,
                                      self.use_cuda, self.batch_size)
            else:
                self._model.inference(h, h.nodes('item'), self.sampler_args, self.bpr,
                                      self.use_cuda, self.batch_size, ntype='item')

    def fit(self, validator: Validator):
        # Load features and add users.
        self.feature_dim = self._features[0].shape[-1]
        self.entity_features = [torch.FloatTensor(np.array(f)) for f in self._features]

        self.users_features = torch.zeros(len(self.meta.users), self.feature_dim, requires_grad=False)

        out_gs = []
        for i, g in enumerate(self.graphs[0]):
            g_h = construct_user_item_heterogeneous_graph(self.meta, g, norm=None)
            e_feat = self.entity_features[i]

            g.ndata['feats'] = torch.cat([e_feat, self.users_features])
            g_h.nodes['item'].data['feats'] = e_feat[self.meta.items]

            out_gs.append((g, g_h))

        self.graphs = out_gs

        super(PinSAGERecommender, self).fit(validator)
        self._model.eval()
        self._inference()
        with torch.no_grad():
            logger.info(validator.validate(self, self.batch_size*2))

    def _fit(self, validator: Validator, first_epoch, final_epoch=1000, trial=None):
        (g, h), _ = self.graphs

        n_entities = len(self.meta.entities)

        neg_sampler = UniformRecommendableItemSampler(1, edges=g.edges())
        if not self.bpr:
            sampler = PinSAGESampler2(h, len(self.layer_dims), *self.sampler_args, prefetch_node_feats=['feats'])
            sampler = HhopPredictionSampler(sampler, neg_sampler)
            nodes, _ = h.out_edges(h.nodes(ntype='item'), etype='iu')
            nodes = nodes.unique()

            nodes = nodes.repeat(g.num_edges() // len(nodes))  # Scale to get similar number of edges
            dataloader = DataLoader(
                h, nodes, sampler, device=self.device, batch_size=self.batch_size,
                shuffle=True, drop_last=False, use_uva=self.use_cuda
            )
        else:
            nodes = g.nodes()
            users = nodes[nodes >= n_entities]
            eids = g.out_edges(users, form='eid')
            g = dgl.AddReverse(g)
            sampler = PinSAGESampler(g, len(self.layer_dims), *self.sampler_args, prefetch_node_feats=['feats'])
            sampler = dgl.dataloading.as_edge_prediction_sampler(
                sampler, negative_sampler=neg_sampler
            )
            dataloader = DataLoader(
                g, eids.to(self.device), sampler, device=self.device, batch_size=self.batch_size,
                shuffle=True, drop_last=False, use_uva=self.use_cuda
            )

        for e in range(first_epoch, final_epoch):
            tot_losses = 0
            tot_correct = 0

            # Only report progress if validation was performed for gridsearch
            if self._no_improvements < self._early_stopping:
                with tqdm(dataloader, total=len(dataloader), desc=f'Epoch {e}', disable=not is_debug_mode()) as progress:
                    for i, (_, positive_graph, negative_graph, blocks) in enumerate(progress, 1):
                        loss, correct = self._model.loss(positive_graph, negative_graph, blocks, self.bpr)

                        self._optimizer.zero_grad()
                        loss.backward()
                        self._optimizer.step()

                        tot_losses += loss.detach()
                        tot_correct += (torch.sum(correct) / correct.size().numel()).detach()
                        progress.set_description(f'Epoch {e:2d}, Loss: {tot_losses / i:.5f}, '
                                                 f'Correct: {tot_correct / i:.5f}')

            elif trial is None:  # Skip last iterations as irrelevant
                break

            self._model.eval()
            with torch.no_grad():
                self._to_report(first_epoch, final_epoch, e, validator, trial,)

    def predict_all(self, users) -> np.array:
        with torch.no_grad():
            user_t = torch.LongTensor(users).to(self.device)
            items = torch.LongTensor(self.meta.items).to(self.device)
            preds = self._model(user_t, items, self.bpr, self.graphs[-1][-1])

        return preds.cpu().numpy()

    def set_parameters(self, parameters):
        self.lr = parameters['learning_rates']
        self.dropout_prop = parameters['dropouts']
        self.weight_decay = parameters['weight_decays']
        self.delta = parameters.get('deltas', 1.)

        self.sampler_args = (self.num_traversals, self.termination_prob, self.num_random_walks,
                             self.num_neighbors)
