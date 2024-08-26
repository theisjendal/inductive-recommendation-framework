import os

import dgl
import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from models.dgl_recommender_base import RecommenderBase
from models.igmc.igmc import IGMC
from models.igmc.subgraph_extraction import SubgraphExtractor
from shared.efficient_validator import Validator
from shared.enums import RecommenderEnum
from shared.graph_utility import UniformRecommendableItemSampler
from shared.utility import is_debug_mode


class GraphRankSampler(dgl.dataloading.EdgePredictionSampler):
    def __init__(self, sampler, subgraph_extractor: SubgraphExtractor, negative_sampler, exclude=None,
                 reverse_eids=None, reverse_etypes=None, prefetch_labels=None):
        super(GraphRankSampler, self).__init__(sampler, exclude, reverse_eids, reverse_etypes, negative_sampler,
                                                     prefetch_labels)
        self.subgraph_extractor = subgraph_extractor

    def sample(self, g, seed_edges):
        _, pos_graph, neg_graph, _ = super(GraphRankSampler, self).sample(g, seed_edges)
        nid = pos_graph.ndata[dgl.NID]
        users, items_i = [nid[index] for index in pos_graph.edges('uv')]
        _, items_j = [nid[index] for index in neg_graph.edges('uv')]

        device = users.device

        pos_graphs = []
        neg_graphs = []
        for user, item_i, item_j in zip(users.cpu(), items_i.cpu(), items_j.cpu()):
            pos_graphs.append(self.subgraph_extractor.create_graph(user, item_i))
            neg_graphs.append(self.subgraph_extractor.create_graph(user, item_j))

        return dgl.batch(pos_graphs).to(device), dgl.batch(neg_graphs).to(device)


class GraphPredictSampler(dgl.dataloading.Sampler):
    def __init__(self, subgraph_extractor: SubgraphExtractor, device):
        self.subgraph_extractor = subgraph_extractor
        self.device = device

    def sample(self, g, seed_edges):
        users, items = g.find_edges(seed_edges)

        graphs = []

        for user, item in zip(users.cpu(), items.cpu()):
            graphs.append(self.subgraph_extractor.create_graph(user, item))

        return dgl.batch(graphs).to(self.device)


class IGMCDGLRecommender(RecommenderBase):
    def __init__(self, **kwargs):
        super(IGMCDGLRecommender, self).__init__(RecommenderEnum.USER_COLD_START, **kwargs)
        self.batch_size = 50  # as defined in their paper
        self.hop = 1
        self.learning_rate = 0.001
        self.learning_rate_factor = 0.1
        self.learning_rate_step = 50
        g, = self.graphs
        g.edata['etypes'] = torch.ones(g.num_edges(), dtype=torch.long)  # Only have positive ratings
        self._min_epochs = 10  # first epochs are unstable.
        user_nodes = g.nodes()[g.nodes() >= len(self.meta.entities)]
        edges = torch.vstack(g.out_edges(user_nodes, 'uv')).T
        path = os.path.join(self.parameter_path, 'subgraphs', self.fold.name)
        os.makedirs(path, exist_ok=True)
        self.extractor = SubgraphExtractor(g, edges, path)

    def _create_model(self, trial):
        in_feats = (self.hop + 1) * 2
        self._model = IGMC(in_feats, regression=True, num_bases=4)

        if self.use_cuda:
            self._model = self._model.cuda()

        self._optimizer = optim.Adam(self._model.parameters(), lr=self.learning_rate, weight_decay=0)
        self._scheduler = optim.lr_scheduler.StepLR(self._optimizer, step_size=self.learning_rate_step,
                                                    gamma=self.learning_rate_factor)

    def fit(self, validator: Validator):
        super(IGMCDGLRecommender, self).fit(validator)
        self._model.eval()

    def _fit(self, validator: Validator, first_epoch, final_epoch=1000, trial=None):
        g, = self.graphs

        # Get user nodes
        user_nodes = g.nodes()[g.nodes() >= len(self.meta.entities)]
        eids = g.out_edges(user_nodes, 'eid')
        reverse_eids = g.in_edges(user_nodes, 'eid')
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        neg_sampler = UniformRecommendableItemSampler(1)

        sampler = GraphRankSampler(
            sampler, self.extractor, negative_sampler=neg_sampler, reverse_eids=reverse_eids
        )

        dataloader = dgl.dataloading.DataLoader(g, eids, sampler, num_workers=8,
                                                    batch_size=self.batch_size, device=self.device)

        score = self._best_score
        for e in range(first_epoch, final_epoch):
            if self._no_improvements < self._early_stopping:
                tot_loss = 0
                progress = tqdm(dataloader, disable=not is_debug_mode())
                for i, (pos, neg) in enumerate(progress):
                    loss, correct = self._model.loss(pos, neg)
                    loss.backward()
                    self._optimizer.step()
                    self._optimizer.zero_grad()

                    tot_loss += loss.detach()
                    progress.set_description(f'Epoch {e}, Loss: {tot_loss / i:.5f}, '
                                             f'{torch.sum(correct)}/'
                                             f'{torch.prod(torch.tensor(correct.shape))}')
                self._model.eval()
                with torch.no_grad():
                    # Due to ranking it is too expensive to rank all users and we choose a very small subset.
                    # Generating all preemptively takes too much space, i.e., >25GB for the smallest dataset.
                    score = validator.validate(self, 1, max_validation=5, max_users=200)

                self._scheduler.step()
            elif trial is None:  # Skip last iterations as irrelevant
                break

            self._on_epoch_end(trial, score, e)

        if trial is not None:
            self._study.finalize(trial=trial)

    def predict_all(self, users, items_indices=None) -> np.array:
        with torch.no_grad():
            n_entities = len(self.meta.entities)
            user_t = torch.LongTensor(users) + n_entities
            items = torch.LongTensor(self.meta.items)
            if items_indices is None:
                length = len(items)
                edges = torch.cartesian_prod(user_t, items).T
            else:
                items_tmp = torch.LongTensor(items_indices)
                length = items_tmp.shape[-1]
                user_tmp = torch.repeat_interleave(user_t, length)
                edges = (user_tmp, items[torch.flatten(items_tmp)])

            g = dgl.graph((edges[0], edges[1]))
            sampler = GraphPredictSampler(self.extractor, self.device)
            d = dgl.dataloading.EdgeDataLoader(
                g, g.edges('eid'), sampler, shuffle=False, batch_size=self.batch_size, drop_last=False,
                device=self.device, num_workers=self.workers*2)

            scores = torch.zeros((len(user_t), length), device=self.device)
            iteration = 0
            for batch in d:
                ss = self._model(batch)
                for s in ss:
                    u = iteration // length
                    i = iteration - u * length
                    scores[u, i] = s
                    iteration += 1

        return scores.cpu().numpy()

    def set_parameters(self, parameters):
        self.hop = parameters['hops']
        self.learning_rate = parameters['learning_rates']
        self.learning_rate_factor = parameters['learning_rate_factors']
        self.learning_rate_step = parameters['learning_rate_steps']
