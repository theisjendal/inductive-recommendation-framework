import cProfile
import pstats

import dgl
import numpy as np
import torch.random
from dgl.dataloading import DataLoader, as_edge_prediction_sampler
from torch import optim
from tqdm import tqdm

from models.dgl_recommender_base import RecommenderBase
from models.graphsage.graphsage import GraphSAGEDGLPredictor
from models.shared.dgl_dataloader import BPRSampler
from shared.efficient_validator import Validator
from shared.enums import RecommenderEnum
from shared.graph_utility import UniformRecommendableItemSampler
from shared.utility import is_debug_mode


class GraphSAGEDGLRecommender(RecommenderBase):
    def __init__(self, **kwargs):
        super().__init__(RecommenderEnum.FULL_COLD_START, **kwargs, features=True)
        self.lr = 0.01
        self.dropout = 0.
        self.aggregator = 'mean'
        self.n_samples = 10
        self.n_layers = 3
        self.dim = 256
        self.num_negative = 10

        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')

    def _create_model(self, trial):
        self.set_seed()

        self._model = GraphSAGEDGLPredictor(self.graphs, len(self.meta.entities), num_layers=self.n_layers,
                                      aggregator=self.aggregator, dim=self.dim, num_samples=self.n_samples,
                                      dropout=self.dropout, features=self._features, use_cuda=self.use_cuda)

        self._optimizer = optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=1e-5)

        if self.use_cuda:
            self._model = self._model.cuda()

        self._sherpa_load(trial)

    def fit(self, validator: Validator):
        self._features = [torch.FloatTensor(np.array(f)) for f in self._features]

        if self.use_cuda:
            self._features = [f.cuda() for f in self._features]

        super(GraphSAGEDGLRecommender, self).fit(validator)

        self._model.eval()

    def _fit(self, validator: Validator, first_epoch, final_epoch=1000, trial=None):
        # Get relevant information
        _, (g, _) = self.graphs

        # Get positive relations.
        eids = g.edges('eid')

        # Create cf dataloader
        cf_sampler = BPRSampler()
        cf_sampler = as_edge_prediction_sampler(cf_sampler,
            negative_sampler=UniformRecommendableItemSampler(1, g.edges()))
        cf_dataloader = DataLoader(
            g, eids.to(self.device), cf_sampler, device=self.device, batch_size=self.batch_size,
            shuffle=True, drop_last=False, use_uva=True)

        for e in range(first_epoch, final_epoch):
            self._model.train()

            tot_loss = 0
            if self._no_improvements < self._early_stopping:
                with tqdm(cf_dataloader, disable=not is_debug_mode()) as progress:
                    for i, (_, pos_graph, neg_graph, _) in enumerate(progress, 1):
                        nid = pos_graph.ndata[dgl.NID]
                        user, item_i = [nid[index] for index in pos_graph.edges('uv')]
                        _, item_j = [nid[index] for index in neg_graph.edges('uv')]

                        loss = self._model.loss(user, item_i, item_j)

                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 5.)

                        self._optimizer.step()
                        self._optimizer.zero_grad()

                        tot_loss = loss.detach()
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
            user_t = torch.LongTensor(users) + len(self.meta.entities)
            items = torch.LongTensor(self.meta.items)
            preds = self._model(user_t, items, rank_all=True)

        return preds.cpu().numpy()

    def set_parameters(self, parameters):
        self.lr = parameters['learning_rates']
        self.dropout = parameters['dropouts']
        self.aggregator = parameters['aggregators']
