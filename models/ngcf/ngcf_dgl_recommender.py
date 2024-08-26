import cProfile

import dgl
import numpy as np
import torch
from dgl.dataloading import DataLoader, as_edge_prediction_sampler
from tqdm import tqdm

from models.dgl_recommender_base import RecommenderBase
from models.ngcf.ngcf import NGCF
from models.utility import construct_user_item_heterogeneous_graph
from shared.enums import RecommenderEnum
from shared.graph_utility import UniformRecommendableItemSampler
from shared.efficient_validator import Validator
from shared.utility import is_debug_mode


class NGCFDGLRecommender(RecommenderBase):
    def __init__(self, lightgcn=True, **kwargs):
        super().__init__(RecommenderEnum.WARM_START, **kwargs)
        self._model = None
        self._model_dict = None
        self.batch_size = 1024
        self.dim = 64
        self.layer_dims = [64, 64, 64]
        self.dropouts = [0.1, 0.1, 0.1]
        self.reg_weight = 1e-5
        self.lr = 0.05
        self.lightgcn = lightgcn
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        self.seed = self.seed_generator.get_seed()

    def _create_model(self, trial):
        self._model = NGCF(self.graphs[0][0], self.dim, self.layer_dims, self.dropouts, lightgcn=self.lightgcn,
                     use_cuda=self.use_cuda)

        if self.use_cuda:
            self._model = self._model.cuda()

        self._optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self._model.parameters()), lr=self.lr,
                                           weight_decay=0 if self.lightgcn else self.reg_weight)

        self._sherpa_load(trial)

    def fit(self, validator: Validator):
        (train_g, test_g), = self.graphs
        # Do not use self loop when using lightgcn
        self.graphs = [
            (construct_user_item_heterogeneous_graph(self.meta, train_g, not self.lightgcn),
            construct_user_item_heterogeneous_graph(self.meta, test_g, not self.lightgcn))
        ]

        super(NGCFDGLRecommender, self).fit(validator)

        self._model.eval()
        with torch.no_grad():
            self._model.inference(test_g, self.batch_size)

    def _fit(self, validator: Validator, first_epoch, final_epoch=1000, trial=None):
        (train_g, test_g), = self.graphs
        subgraph = train_g.edge_type_subgraph(['ui'])

        eids = subgraph.edges(etype='ui', form='eid')

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3, prefetch_edge_feats={etype: ['norm']
                                                                                        for etype in train_g.etypes})
        neg_sampler = UniformRecommendableItemSampler(1)#, g.edges(etype='ui'))
        sampler = as_edge_prediction_sampler(sampler, negative_sampler=neg_sampler, exclude='reverse_types',
            reverse_etypes={'ui': 'iu'},)
        dataloader = DataLoader(
            train_g, {'ui': eids}, sampler, device=self.device, batch_size=self.batch_size,
            shuffle=True, drop_last=False,  num_workers=self.workers,)

        for e in range(first_epoch, final_epoch):
            tot_losses = 0

            # Only report progress if validation was performed for gridsearch
            to_report = (self._gridsearch and ((e+1) % self._eval_intermission) == 0) or not self._gridsearch

            score = self._best_score
            if self._no_improvements < self._early_stopping:
                progress = tqdm(dataloader, total=len(dataloader), desc=f'Epoch {e}', disable=not is_debug_mode())
                for i, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(progress):
                    loss, correct = self._model.loss(input_nodes, positive_graph, negative_graph, blocks)

                    if self.lightgcn:
                        loss, reg_loss = loss
                        reg_loss = reg_loss * self.reg_weight
                        loss += reg_loss

                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()

                    tot_losses += loss.detach()
                    progress.set_description(f'Epoch {e:2d}, CFLoss: {tot_losses / (i+1):.5f}, '
                                             f'{f"RegLoss: {reg_loss:.5f}, " if self.lightgcn else ""}'
                                             f'{torch.sum(correct)}/'
                                             f'{torch.prod(torch.tensor(correct.shape))}')

                if to_report:
                    self._model.eval()
                    with torch.no_grad():
                        self._model.inference(test_g, self.batch_size)
                        score = validator.validate(self, self.batch_size*2)
            elif trial is None:  # Skip last iterations as irrelevant
                break

            if to_report:
                self._on_epoch_end(trial, score, e, tot_losses)

        if trial is not None:
            self._study.finalize(trial=trial)

    def predict_all(self, users) -> np.array:
        with torch.no_grad():
            user_t = torch.LongTensor(users)
            items = torch.LongTensor(self.meta.items)
            preds = self._model(user_t, items, rank_all=True)

        return preds.cpu().numpy()

    def set_parameters(self, parameters):
        self.lr = parameters['learning_rates']
        self.reg_weight = parameters['weight_decays']
        self.dropouts = [parameters['dropouts']] * len(self.layer_dims)
        self.batch_size = parameters.get('batch_sizes', 1024)
