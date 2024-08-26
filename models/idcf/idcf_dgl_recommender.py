import os
from copy import deepcopy

import dgl
import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from models.dgl_recommender_base import RecommenderBase
from models.idcf.idcf import IDCF
from models.igmc.subgraph_extraction import SubgraphExtractor
from models.shared.dgl_dataloader import BPRSampler, UserBlockSampler
from models.utility import construct_user_item_heterogeneous_graph
from shared.efficient_validator import Validator
from shared.enums import RecommenderEnum
from shared.graph_utility import UniformRecommendableItemSampler
from shared.utility import is_debug_mode


class IDCFDGLRecommender(RecommenderBase):
    def __init__(self, use_bpr=False, **kwargs):
        # Note only GC implementation and for extrapolation. We assume we do not know new users on training time.
        super(IDCFDGLRecommender, self).__init__(RecommenderEnum.USER_COLD_START, **kwargs)
        self.f_dim = 32
        self.layers_dim = [128, 32, 32, 1]
        self.attention_heads = 4
        self.attention_sample = 200
        self.weight_decay = 0.001
        self.decay_factor = 1.  # 0.95
        self.adapt_weight_decay = 10
        self.pretrain_lr = 0.01
        self.learning_rate = 0.002
        self.n_pretrain_epochs = 500
        self.use_bpr = use_bpr
        (train_g, test_g), = self.graphs
        train_g.edata['etypes'] = torch.ones(train_g.num_edges(), dtype=torch.long)  # Only have positive ratings
        user_nodes = train_g.nodes()[train_g.nodes() >= len(self.meta.entities)]
        edges = torch.vstack(train_g.out_edges(user_nodes, 'uv')).T
        path = os.path.join(self.parameter_path, 'subgraphs', self.fold.name)
        os.makedirs(path, exist_ok=True)
        self.extractor = SubgraphExtractor(train_g, edges, path)
        self.matrix_factorization = False

    def _create_model(self, trial):
        (train_g, test_g), = self.graphs
        self._model = IDCF(train_g, self.f_dim, self.layers_dim, self.attention_heads)

        if self.use_cuda:
            self._model = self._model.cuda()

        exclude = ['features', 'fcs', 'layer']
        params = [p for n, p in self._model.named_parameters() if not any([n.startswith(ex) for ex in exclude])]
        self._optimizer = optim.Adam(params, lr=self.learning_rate, weight_decay=5e-2)
        self._scheduler = torch.optim.lr_scheduler.ExponentialLR(self._optimizer, gamma=self.decay_factor)

    def fit(self, validator: Validator):
        (train_g, test_g), = self.graphs
        train_g = construct_user_item_heterogeneous_graph(self.meta, train_g, self_loop=False)
        test_g = construct_user_item_heterogeneous_graph(self.meta, test_g, self_loop=False)

        self.graphs = [(train_g, test_g)]

        for g in [train_g, test_g]:
            for etype in g.etypes:
                g.edges[etype].data['e'] = torch.ones(g.num_edges(etype=etype))

        super(IDCFDGLRecommender, self).fit(validator)

        key_users = train_g.in_edges(train_g.nodes('user'), etype='iu')[1]
        self._inference(g=train_g, mf=True)
        self._inference(g=test_g, key_users=key_users, num_samples=self.attention_sample,
                        num_heads=self.attention_heads, batch_size=self.batch_size // 8)

    def _fit(self, validator: Validator, first_epoch, final_epoch=1000, trial=None):
        (g, test_g), = self.graphs

        # Get user nodes
        eids = g.edges(etype='ui', form='eid')

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        neg_sampler = UniformRecommendableItemSampler(1)
        sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, exclude='reverse_types',
                                                             reverse_etypes={'ui': 'iu'}, negative_sampler=neg_sampler)
        mf_dataloader = dgl.dataloading.DataLoader(
            g, {'ui': eids.to(self.device)}, sampler, batch_size=self.batch_size * 4, device=self.device,
            use_uva=self.use_cuda, shuffle=True, drop_last=False
        )

        sampler = UserBlockSampler(g.nodes('user').to(self.device), self.attention_sample,
                                   self.attention_heads)
        sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, exclude='reverse_types',
                                                             reverse_etypes={'ui': 'iu'}, negative_sampler=neg_sampler)
        idcf_dataloader = dgl.dataloading.DataLoader(
            g, {'ui': eids.to(self.device)}, sampler, batch_size=self.batch_size, device=self.device,
            use_uva=self.use_cuda, shuffle=True, drop_last=False
        )

        # Pretrain model using matrix factorization if starting from epoch 0.
        validator.batch_size = 128
        if first_epoch == 0:
            self.matrix_factorization = True
            self._model.use_user_bias = True
            best_state = deepcopy(self._model.state_dict())
            best_score = 0
            score = 0

            # Noting that the first two parameters are user and item embeddings.
            optimizer = optim.Adam(self._model.parameters(), self.pretrain_lr)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.decay_factor)
            with tqdm(range(self.n_pretrain_epochs), total=self.n_pretrain_epochs, disable=not is_debug_mode()) \
                    as progress:
                for e in progress:
                    tot_loss = 0.
                    tot_reg = 0.
                    tot_correct = 0
                    self._model.train()
                    # for i, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(mf_dataloader, 1):
                    for i, (input_nodes, pos, neg, blocks) in enumerate(mf_dataloader, 1):
                        # trained towards ranking for better performance
                        loss, reg_loss, correct = self._model.pretrain_loss(input_nodes, pos, neg, blocks[0],
                                                                            use_bpr=self.use_bpr)
                        reg_loss = self.weight_decay * reg_loss
                        tot_loss += loss.item()
                        tot_reg += reg_loss.item()
                        tot_correct += (correct.sum() / correct.size().numel()).item()
                        loss = loss + reg_loss
                        loss.backward()

                        optimizer.step()
                        optimizer.zero_grad()
                        progress.set_description(f'Pretraining {e}, BS: {best_score:.5f}, '
                                                 f'LS: {score:.5f}, '
                                                 f'L: {tot_loss / i:.3f}, '
                                                 f'l2: {tot_reg / i:.3f}, '
                                                 f'C: {tot_correct / i:.3f}')

                    scheduler.step()
                    if (e + 1) % self._eval_intermission == 0:
                        self._model.eval()
                        with torch.no_grad():
                            self._model.mf_inference(g)
                            score = validator.validate(self, 0, verbose=False)

                        if score > best_score:
                            best_state = deepcopy(self._model.state_dict())
                            best_score = score
                        elif self._no_improvements * self._eval_intermission + 1 < self._early_stopping:
                            self._no_improvements += 1
                        else:
                            break

            # Reset mf flag and load best state
            self.matrix_factorization = False
            self._no_improvements = 0
            self._model.use_user_bias = False
            self._model.load_state_dict(best_state)

        self._inference(g=g, mf=True)

        key_users = g.in_edges(g.nodes('user'), etype='iu')[1]
        for e in range(first_epoch, final_epoch):
            self._model.train()
            if self._no_improvements < self._early_stopping:
                tot_loss = 0
                tot_c = 0.
                tot_correct = 0
                progress = tqdm(idcf_dataloader, disable=not is_debug_mode())
                for i, (input_nodes, pos, neg, blocks) in enumerate(progress, 1):
                    loss, contrastive_loss, correct = self._model.adaptation_loss(input_nodes, pos, neg, blocks,
                                                                                  use_bpr=self.use_bpr)
                    contrastive_loss *= self.adapt_weight_decay
                    loss = loss + contrastive_loss
                    tot_correct += (correct.sum() / correct.size().numel()).item()
                    loss.backward()
                    self._optimizer.step()
                    self._optimizer.zero_grad()

                    tot_loss += loss.detach()
                    tot_c += contrastive_loss.item()
                    progress.set_description(f'Epoch {e}, Loss: {tot_loss / i:.5f}, Closs: {tot_c / i:.5f}'
                                             f'C: {tot_correct / i:.3f}')

                self._scheduler.step()
            elif trial is None:  # Skip last iterations as irrelevant
                break

            self._to_report(first_epoch, final_epoch, e, validator, trial, g=g,
                            key_users=key_users, num_samples=self.attention_sample,
                            num_heads=self.attention_heads, batch_size=self.batch_size)

    def _inference(self, **kwargs):
        self._model.eval()
        with torch.no_grad():
            if kwargs.pop('mf', False):
                self._model.mf_inference(**kwargs)
            else:
                self._model.adaptive_inference(**kwargs)

    def predict_all(self, users, items_indices=None) -> np.array:
        with torch.no_grad():
            items = torch.LongTensor(self.meta.items)
            user_t = torch.LongTensor(users)
            if self.use_cuda:
                user_t = user_t.cuda()
                items = items.cuda()

            scores = []
            n_batches = items.size(0) // self.batch_size if items.size(0) % self.batch_size == 0 \
                else (items.size(0) // self.batch_size) + 1

            for i in range(n_batches):
                scores.append(self._model.predict(user_t, items[i * self.batch_size:(i + 1) * self.batch_size],
                                                  query=not self.matrix_factorization))

            scores = torch.cat(scores, dim=-1)

        return scores.cpu().numpy()

    def set_parameters(self, parameters):
        self.learning_rate = parameters['learning_rates']
        self.pretrain_lr = parameters.get('pretrain_learning_rates', 0.001)
        self.attention_sample = parameters['attention_samples']
        self.weight_decay = parameters['weight_decays']
        self.f_dim, self.layers_dim = eval(parameters['dims'])
