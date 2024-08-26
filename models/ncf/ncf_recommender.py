import dgl.dataloading
import numpy as np
import torch
from dgl.dataloading import DataLoader, as_edge_prediction_sampler
from loguru import logger
from torch import optim
from tqdm import tqdm

from models.dgl_recommender_base import RecommenderBase
from models.ncf.ncf import NCF
from models.shared.dgl_dataloader import BPRSampler
from shared.efficient_validator import Validator
from shared.enums import RecommenderEnum
from shared.graph_utility import UniformRecommendableItemSampler
from shared.utility import is_debug_mode


class NCFRecommender(RecommenderBase):
    def __init__(self, learned_embeddings=False, aggregator=False, **kwargs):
        super().__init__(RecommenderEnum.WARM_START, **kwargs)
        # self._max_epochs = 100  # Only method to increase number of epochs.
        self._min_epochs = 0
        self.lr = 0.1
        self.dims = [64, 32, 16, 8, 8, 8]
        self.num_layers = 3
        self.batch_size = 1024
        self.num_neg = 1
        self.weight_decay = 1e-5
        self.feature_dim = 0
        self.features = None
        self.n_anchors = 100
        self.path_length = 2
        self.learned_embeddings = learned_embeddings
        self.aggregator = aggregator
        self.attention_heads = 4
        self.rel_emb = 16

        if self._gridsearch:
            self._early_stopping = 5
            self._eval_intermission = 10

    def _create_model(self, trial):
        self.set_seed()

        (train_g, test_g), = self.graphs

        if self.learned_embeddings:
            logger.warning(f'Using learned embeddings.')
            f = torch.zeros((train_g.num_nodes(), self.dims[0]))
            torch.nn.init.xavier_uniform_(f)
            self.features = torch.nn.Parameter(f, requires_grad=True)
        elif self.aggregator:
            self.features = torch.IntTensor(np.array(self.get_features({'n_anchors': self.n_anchors}, {})))
        else:
            self.features = torch.FloatTensor(np.array(self.get_features({'n_anchors': self.n_anchors,
                                                                          'p_length': self.path_length}, {})))

        self.feature_dim = self.features.size(-1)

        p_length = 0
        n_relations = 0
        if not self.learned_embeddings:
            p_length = self.features.size(-1) // self.n_anchors
            n_relations = torch.max(self.features).cpu().item() + 1

        self._model = NCF(self.feature_dim, self.dims[:self.num_layers], self.aggregator, self.rel_emb, self.dims[0],
                          self.attention_heads, p_length, self.n_anchors, n_relations)
        self._optimizer = optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.learned_embeddings:
            self._model.register_parameter('feature_embedding', self.features)

        if self.use_cuda:
            self._model = self._model.cuda()
            self.features = self.features.cuda()

        self._sherpa_load(trial)

    def fit(self, validator: Validator):
        super(NCFRecommender, self).fit(validator)

        self._model.eval()

    def _fit(self, validator: Validator, first_epoch, final_epoch=1000, trial=None):
        # Get relevant information
        (train_g, test_g), = self.graphs

        if self.use_cuda:
            # Get the memory available
            t = torch.cuda.get_device_properties(self.device).total_memory

            # We want the mem usage (m) to be less than t/4 (approx of model usage). We have the formula t/4 < m.
            # m = input feature dim * 2 * #items * bytes used * batch_size
            # It can be rewritten to calculate the max batch size we can use as:
            emb_dim = max(self.feature_dim, self.dims[0])

            validator.batch_size = min(int(((emb_dim * 2 * len(self.meta.items) *
                                             self.features.element_size())/(t/4))**-1), 1024)

        # Get positive relations.
        eids = train_g.edges('eid')

        # Create cf dataloader
        cf_sampler = BPRSampler()
        cf_sampler = as_edge_prediction_sampler(cf_sampler, negative_sampler=UniformRecommendableItemSampler(self.num_neg))
        cf_dataloader = DataLoader(
            train_g, eids.to(self.device), cf_sampler, device=self.device,
            batch_size=self.batch_size,
            shuffle=True, drop_last=False, use_uva=self.use_cuda)

        for e in range(first_epoch, final_epoch):
            self._model.train()

            # Only report progress if validation was performed for gridsearch
            to_report = (self._gridsearch and ((e+1) % self._eval_intermission) == 0) or not self._gridsearch

            tot_loss = 0
            acc = 0
            score = self._best_score
            if self._no_improvements < self._early_stopping:
                progress = tqdm(cf_dataloader, disable=not is_debug_mode())
                for i, (input_nodes, pos_graph, neg_graph, _) in enumerate(progress, 1):
                    features = self.features[input_nodes]
                    loss, correct = self._model.loss(pos_graph, neg_graph, features)

                    loss.backward()

                    self._optimizer.step()
                    self._optimizer.zero_grad()

                    tot_loss += loss.detach()
                    acc += (correct.sum() / correct.size().numel()).detach()
                    progress.set_description(f'Epoch {e}, CFLoss: {tot_loss / i:.5f}, '
                                             f'Accuracy: {acc / i:.5f}')

                if to_report:
                    self._model.eval()
                    with torch.no_grad():
                        score = validator.validate(self, 0)
            elif trial is None:  # Skip last iterations as irrelevant
                break

            if to_report:
                self._on_epoch_end(trial, score, e, tot_loss)
        if trial is not None:
            self._study.finalize(trial=trial)

    def predict_all(self, users) -> np.array:
        with torch.no_grad():
            user_t = (torch.LongTensor(users) + len(self.meta.entities)).to(self.device)
            items = torch.LongTensor(self.meta.items).to(self.device)

            preds = self._model(user_t, items, self.features)

        return preds.cpu().numpy()

    def set_parameters(self, parameters):
        self.lr = parameters['learning_rates']
        self.num_neg = parameters.get('num_negatives', 1)
        self.weight_decay = parameters.get('weight_decays', 1e-5)
        self.num_layers = parameters['layers']
        self.n_anchors = parameters.get('anchors', 100)
