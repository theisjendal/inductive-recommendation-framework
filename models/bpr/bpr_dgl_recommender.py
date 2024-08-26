import cProfile
import pstats

import dgl.dataloading
import numpy as np
import torch
from dgl.dataloading import DataLoader, as_edge_prediction_sampler
from torch import optim
from tqdm import tqdm

from models.bpr.bpr import BPR
from models.dgl_recommender_base import RecommenderBase
from models.shared.dgl_dataloader import BPRSampler
from shared.efficient_validator import Validator
from shared.enums import RecommenderEnum
from shared.graph_utility import UniformRecommendableItemSampler
from shared.utility import is_debug_mode


class BPRDGLRecommender(RecommenderBase):
    def __init__(self, user_bias=True, item_bias=True, **kwargs):
        super().__init__(RecommenderEnum.FULL_COLD_START, **kwargs)
        self.user_bias = user_bias
        self.item_bias = item_bias

        # self._max_epochs = 100  # Only method to increase number of epochs.
        # self._min_epochs = 0

        self.lr = 0.1
        self.dim = 32
        self.num_neg = 1
        self.weight_decay = 1e-5
        self.n_anchors = 32

        has_f_conf = self._feature_configuration is not None

        if has_f_conf:
            self._eval_intermission = 1
        # elif self._gridsearch:
        #     self._early_stopping = 5
        #     self._eval_intermission = 10

        self._user_map = None
        self._item_map = None

        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')

        if has_f_conf:
            self._max_epochs = 1

    def _create_model(self, trial):
        self.set_seed()

        self._model = BPR(len(self.meta.users), len(self.meta.entities), self.dim,
                          self.user_bias, self.item_bias)
        self._optimizer = optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self._feature_configuration is not None:
            f = torch.FloatTensor(self.get_features({'n_anchors': self.n_anchors}, {}))
            self._model.item_emb.weight = torch.nn.Parameter(f[:len(self.meta.entities)], requires_grad=False)
            self._model.user_emb.weight = torch.nn.Parameter(f[len(self.meta.entities):], requires_grad=False)

        if self.use_cuda:
            self._model = self._model.cuda()

        self._sherpa_load(trial)

    def fit(self, validator: Validator):
        super(BPRDGLRecommender, self).fit(validator)

        self._model.eval()

    def _fit(self, validator: Validator, first_epoch, final_epoch=1000, trial=None):
        # Get relevant information - Trains on validation users as we assume BPR to be fast enough.
        (_, g), = self.graphs

        # g.ndata['feats'] = torch.zeros((g.num_nodes(), self.dim))

        # Get positive relations.
        eids = g.edges('eid')

        # Create cf dataloader
        cf_sampler = BPRSampler()
        cf_sampler = as_edge_prediction_sampler(cf_sampler,
            negative_sampler=UniformRecommendableItemSampler(self.num_neg))
        cf_dataloader = DataLoader(
            g, eids, cf_sampler, device=self.device, batch_size=self.batch_size,
            shuffle=True, drop_last=False)

        for e in range(first_epoch, final_epoch):
            self._model.train()

            tot_loss = 0
            length = len(self.meta.entities)
            if self._no_improvements < self._early_stopping:
                with tqdm(cf_dataloader, disable=not is_debug_mode()) as progress:
                    for i, (_, pos_graph, neg_graph, _) in enumerate(progress):
                        if self._feature_configuration is not None:
                            break
                        nid = pos_graph.ndata[dgl.NID]
                        user, item_i = [nid[index] for index in pos_graph.edges('uv')]
                        _, item_j = [nid[index] for index in neg_graph.edges('uv')]

                        loss = self._model.loss(user - length, item_i, item_j)

                        loss.backward()

                        self._optimizer.step()
                        self._optimizer.zero_grad()

                        tot_loss += loss.detach()
                        progress.set_description(f'Epoch {e}, CFLoss: {tot_loss / i:.5f}')
            elif trial is None:  # Skip last iterations as irrelevant
                break

            self._model.eval()
            with torch.no_grad():
                self._to_report(first_epoch, final_epoch, e, validator, trial)

    def _inference(self, **kwargs):
        pass

    def predict_all(self, users) -> np.array:
        with torch.no_grad():
            user_t = torch.LongTensor(users).to(self.device)
            items = torch.LongTensor(self.meta.items).to(self.device)
            preds = self._model(user_t, items, rank_all=True)

        return preds.cpu().numpy()

    def set_parameters(self, parameters):
        self.lr = parameters.get('learning_rates', 1)
        self.dim = parameters.get('latent_factors', 32)
        self.num_neg = parameters.get('num_negatives', 1)
        self.weight_decay = parameters.get('weight_decays', 1e-5)
        self.n_anchors = parameters.get('anchors', 32)
