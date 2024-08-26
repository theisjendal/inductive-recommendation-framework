import cProfile

import dgl
import numpy as np
import torch
from dgl.dataloading import DataLoader, as_edge_prediction_sampler
from tqdm import tqdm

from models.dgl_recommender_base import RecommenderBase
from models.inmo.inmo import INMO
from models.shared.dgl_dataloader import BPRSampler
from models.utility import construct_user_item_heterogeneous_graph
from shared.enums import RecommenderEnum
from shared.graph_utility import UniformRecommendableItemSampler
from shared.efficient_validator import Validator
from shared.utility import is_debug_mode


class AverageMeter:
    def __init__(self):
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class INMORecommender(RecommenderBase):
    def __init__(self, **kwargs):
        super().__init__(RecommenderEnum.FULL_COLD_START, **kwargs)
        self._model = None
        self._model_dict = None
        self.dim = 64
        self.layer_dims = [64, 64, 64]
        self.dropouts = [0.1, 0.1, 0.1]
        self.reg_weight = 1e-5
        self.aux_weight = 1e-5
        self.n_layers = 3
        self.lr = 0.05
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        self.seed = self.seed_generator.get_seed()

    def _create_model(self, trial):
        # Graphs for inductive part for train and test.
        (_, i_g), (_, i_test_g) = [[g.to(self.device) for g in gs] for gs in self.graphs]
        n_nodes = {'user': len(self.meta.users)+1, 'item': len(self.meta.entities)+1}
        self._model = INMO(i_g, i_test_g, n_nodes, self.dim, self.layer_dims, self.dropouts,
                           use_cuda=self.use_cuda)

        if self.use_cuda:
            self._model = self._model.cuda()

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        self._sherpa_load(trial)

    def fit(self, validator: Validator):
        (g, test_g), = self.graphs

        # Get propagation graphs for train and test.
        g = construct_user_item_heterogeneous_graph(self.meta, g, self_loop=False, global_nodes=False)
        test_g = construct_user_item_heterogeneous_graph(self.meta, test_g, self_loop=False, global_nodes=False)

        # Find warm_start users and items (edges are symmetric)
        src, dst = [torch.unique(e) for e in g.edges(etype='ui')]

        # Find inductive edges used for inference.
        inf_edges = {}
        inf_edges['ui'] = test_g.out_edges(src, form='eid', etype='ui')
        inf_edges['iu'] = test_g.out_edges(dst, form='eid', etype='iu')

        # Create inductive graphs for train and test. For warm all edges are used.
        inductive_g = g.edge_type_subgraph(['ui', 'iu'])
        inductive_test_g = test_g.edge_subgraph(inf_edges, store_ids=True, relabel_nodes=False).\
            edge_type_subgraph(['ui', 'iu'])

        self.graphs = [(g, inductive_g), (test_g, inductive_test_g)]

        super(INMORecommender, self).fit(validator)

        self._model.eval()
        with torch.no_grad():
            self._model.update_alpha(self._best_epoch + 1)
            self._model.set_norms(self._model.train_g)
            self._model.set_norms(self._model.inference_g)
            self._model.inference(test_g)

    def _fit(self, validator: Validator, first_epoch, final_epoch=1000, trial=None):
        (g, _), (test_g, _) = self.graphs
        subgraph = g.edge_type_subgraph(['ui'])

        eids = subgraph.edges(etype='ui', form='eid')

        # CF
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3, prefetch_edge_feats={etype: ['norm'] for etype in g.etypes})
        neg_sampler = UniformRecommendableItemSampler(1, g.edges(etype='ui'), methodology='non-rated')
        sampler = as_edge_prediction_sampler(sampler, negative_sampler=neg_sampler)
        dataloader = DataLoader(
            g, {'ui': eids}, sampler, device=self.device, batch_size=self.batch_size,
            shuffle=True, drop_last=True, num_workers=self.workers)

        # AUX
        sampler = BPRSampler()
        neg_sampler = UniformRecommendableItemSampler(1, g.edges(etype='ui'), methodology='non-rated')
        sampler = as_edge_prediction_sampler(sampler, negative_sampler=neg_sampler)
        aux_dataloader = DataLoader(
            g, {'ui': eids}, sampler, device=self.device, batch_size=self.batch_size,
            shuffle=True, drop_last=True, num_workers=self.workers)

        n_iter = len(dataloader)
        for e in range(first_epoch, final_epoch):
            tot_losses = 0
            test = AverageMeter()
            self._model.update_alpha(e)
            self._model.set_norms(self._model.train_g)
            self._model.set_norms(self._model.inference_g)
            self._model.train()

            # Only report progress if validation was performed for gridsearch
            to_report = (self._gridsearch and ((e+1) % self._eval_intermission) == 0) or not self._gridsearch
            if self._no_improvements < self._early_stopping:
                progress = tqdm(zip(dataloader, aux_dataloader), total=len(dataloader), desc=f'Epoch {e}',
                                disable=not is_debug_mode())
                for i, ((input_nodes, positive_graph, negative_graph, blocks), (_, pos_graph, neg_graph, _)) \
                        in enumerate(progress):
                    loss, correct = self._model.loss(input_nodes, positive_graph, negative_graph, blocks)

                    aux_loss = self._model.aux_loss(pos_graph, neg_graph)

                    loss_cf, reg_loss = loss
                    reg_loss = reg_loss * self.reg_weight
                    aux_loss = aux_loss * self.aux_weight
                    loss = loss_cf + reg_loss + aux_loss

                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()

                    tot_losses += loss.detach()
                    acc = torch.sum(correct) / correct.shape.numel()
                    test.update(loss.detach())
                    progress.set_description(f'Epoch {e:2d}, CFL: {test.avg:.5f}, '
                                             f'{f"RL: {reg_loss:.5f}, "}'
                                             f'{f"AL: {aux_loss:.5f}, "}'
                                             f'{torch.sum(correct)}/'
                                             f'{torch.prod(torch.tensor(correct.shape))}')

                    step = e * n_iter + i
                    self.summary_writer.add_scalar('loss/cfl', loss_cf, step)
                    self.summary_writer.add_scalar('loss/reg', reg_loss, step)
                    self.summary_writer.add_scalar('loss/aux', aux_loss, step)
                    self.summary_writer.add_scalar('loss/acc', acc, step)

                self._model.update_alpha(e + 1)
                self._model.set_norms(self._model.train_g)
                self._model.set_norms(self._model.inference_g)

                progress.close()
            elif trial is None:  # Skip last iterations as irrelevant
                break

            self._model.eval()
            with torch.no_grad():
                self._to_report(first_epoch, final_epoch, e, validator, trial, g=test_g, batch_size=None)

    def _inference(self, **kwargs):
        self._model.inference(**kwargs)

    def predict_all(self, users) -> np.array:
        with torch.no_grad():
            user_t = torch.LongTensor(users)
            items = torch.LongTensor(self.meta.items)
            preds = self._model(user_t, items, rank_all=True)

        return preds.cpu().numpy()

    def set_parameters(self, parameters):
        self.n_layers = parameters.get('n_layers', 3)
        self.lr = parameters['learning_rates']
        self.reg_weight = parameters['weight_decays']
        self.dropouts = [parameters['dropouts']] * self.n_layers
        self.dim = parameters['dim']
        self.layer_dims = [parameters['dim']] * self.n_layers
        self.batch_size = parameters.get('batch_sizes', self.batch_size)
        self.aux_weight = parameters.get('auxiliary_weight', 1e-5)
