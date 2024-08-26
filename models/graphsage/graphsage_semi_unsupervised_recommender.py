import cProfile
import pstats

import dgl
import numpy as np
import torch.random
from dgl.dataloading import EdgeDataLoader
from torch import optim, nn
from tqdm import tqdm

from models.dgl_recommender_base import RecommenderBase
from models.graphsage.graphsage import GraphSAGEDGLPredictor
from models.graphsage.graphsage_semi_supervised import GraphSAGE
from models.shared.dgl_dataloader import BPRSampler
from shared.efficient_validator import Validator
from shared.graph_utility import UniformRecommendableItemSampler
from shared.utility import is_debug_mode


class GraphSAGESemiUnsupervisedRecommender(RecommenderBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, features=True)
        self.batch_size = 1024
        self.lr = 0.01
        self.dropout = 0.
        self.aggregator = 'mean'
        self.n_samples = 10
        self.n_layers = 3
        self.dim = 256
        self.out_dim = 64
        self.num_negative = 10

        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')

    def _create_model(self, trial):
        self.set_seed()

        self._model = GraphSAGE(self._features.size(-1), self.dim, self.out_dim, self.n_layers, nn.ReLU(), self.dropout,
                                self.aggregator, self._features)

        self._optimizer = optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=1e-5)

        if self.use_cuda:
            self._model = self._model.cuda()

        self._sherpa_load(trial)

    def fit(self, validator: Validator):
        self._features = torch.FloatTensor(np.array(self._features))

        if self.use_cuda:
            self._features = self._features.cuda()

        super(GraphSAGESemiUnsupervisedRecommender, self).fit(validator)

        self._model.eval()

    def _fit(self, validator: Validator, first_epoch, final_epoch=1000, trial=None):
        # Get relevant information
        _, g = self.graphs

        # Get positive relations.
        eids = g.edges('eid')

        # Create cf dataloader
        cf_sampler = BPRSampler()

        cf_dataloader = EdgeDataLoader(
            g, eids.to(self.device), cf_sampler, device=self.device,
            negative_sampler=UniformRecommendableItemSampler(1), batch_size=self.batch_size,
            shuffle=True, drop_last=False, use_uva=True)

        for e in range(first_epoch, final_epoch):
            self._model.train()

            tot_loss = 0
            score = self._best_score
            if self._no_improvements < self._early_stopping:
                progress = tqdm(cf_dataloader, disable=not is_debug_mode())
                for i, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(progress):
                    loss = self._model.loss(input_nodes, pos_graph, neg_graph, blocks)

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), 5.)

                    self._optimizer.step()
                    self._optimizer.zero_grad()

                    tot_loss = loss.detach()
                    progress.set_description(f'Epoch {e}, CFLoss: {tot_loss / i:.5f}')

                self._model.eval()
                with torch.no_grad():
                    score = validator.validate(self, 4)
            elif trial is None:  # Skip last iterations as irrelevant
                break

            self._on_epoch_end(trial, score, e, tot_loss)

        if trial is not None:
            self._study.finalize(trial=trial)

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
