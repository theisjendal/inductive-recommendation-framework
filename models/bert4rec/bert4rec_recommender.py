import numpy as np
import torch
from dgl import DGLHeteroGraph
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.bert4rec.bert4rec import BERT4Rec
from models.bert4rec.dataloader import BertTrainDataset
from models.dgl_recommender_base import RecommenderBase
from shared.efficient_validator import Validator
from shared.enums import RecommenderEnum
from shared.user import User
from shared.utility import is_debug_mode


class PolynomialLR(_LRScheduler):
    """Decays the learning rate of each parameter group using a polynomial function
    in the given total_iters. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_iters (int): The number of steps that the scheduler decays the learning rate. Default: 5.
        power (int): The power of the polynomial. Default: 1.0.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # xdoctest: +SKIP("undefined vars")
        >>> # Assuming optimizer uses lr = 0.001 for all groups
        >>> # lr = 0.001     if epoch == 0
        >>> # lr = 0.00075   if epoch == 1
        >>> # lr = 0.00050   if epoch == 2
        >>> # lr = 0.00025   if epoch == 3
        >>> # lr = 0.0       if epoch >= 4
        >>> scheduler = PolynomialLR(self.opt, total_iters=4, power=1.0)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """
    def __init__(self, optimizer, total_iters=5, power=1.0, last_epoch=-1, verbose=False):
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        decay_factor = ((1.0 - self.last_epoch / self.total_iters) / (1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
        return [group["lr"] * decay_factor for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [
            (
                base_lr * (1.0 - min(self.total_iters, self.last_epoch) / self.total_iters) ** self.power
            )
            for base_lr in self.base_lrs
        ]

class BERT4RecRecommender(RecommenderBase):
    def __init__(self, position_aware=True, **kwargs):
        super().__init__(RecommenderEnum.USER_COLD_START, **kwargs)

        # Data splitting parameters
        self._max_sequence_size = 200
        self._sliding_window_prob = 0.5
        self._samples_per_user = 5
        self._max_mask = 25
        self._mask_prop = 0.1
        self._sliding_step = (int)(self._sliding_window_prob * self._max_sequence_size) \
            if self._sliding_window_prob != -1.0 else self._max_sequence_size

        # Model parameters
        self.batch_size = 512
        self.learning_rate = 0.01
        self.position_aware = position_aware

        # Misc
        self.uis = None

    def _get_rating_time(self, g: DGLHeteroGraph, mask=None):
        times = g.edata.get('rating_time', torch.zeros(g.num_edges()))
        if mask is not None:
            times = times[mask]

        return times.tolist()

    def _create_model(self, trial):
        self.set_seed()
        n_entities = len(self.meta.entities)
        self._model = BERT4Rec(n_entities, self._max_sequence_size, self.dim, self.num_hidden_layers,
                               self.num_attention_heads,
                               use_positional=self.position_aware, dropout=self.dropout)
        self._optimizer = optim.AdamW(self._model.parameters(), lr=self.learning_rate, weight_decay=0)
        self._scheduler = PolynomialLR(self._optimizer, total_iters=400000, power=1.0)
        if self.use_cuda:
            self._model = self._model.cuda()

        self._sherpa_load(trial)

    def fit(self, validator: Validator):
        super(BERT4RecRecommender, self).fit(validator)
        self._model.eval()
        with torch.no_grad():
            self._inference()

    def _fit(self, validator: Validator, first_epoch, final_epoch=1000, trial=None):
        # Get data
        n_entities = len(self.meta.entities)
        n_items = len(self.meta.items)
        train_g: DGLHeteroGraph
        (train_g, val_g), = self.graphs
        ce = torch.nn.CrossEntropyLoss()
        ui_map = {u: [i.item() for i in train_g.out_edges(u+n_entities)[1]] for u in self.meta.users}
        s, d, t = train_g.edges('all')
        mask = s >= n_entities
        s = s[mask] - n_entities
        time_map = {(s, d): t for s, d, t in zip(s.tolist(), d[mask].tolist(), self._get_rating_time(train_g, mask))}

        # create windows
        windows = []
        for u in self.meta.users:
            u: User
            rated = ui_map[u]
            rated = list(sorted(rated, key=lambda x: time_map[(u, x)]))
            ui_map[u] = rated  # update with sorted
            beg_idx = list(range(len(rated)-self._max_sequence_size, 0, -self._sliding_step))

            if not len(beg_idx):
                beg_idx = []

            beg_idx.append(0)
            seqs = [rated[i:i + self._max_sequence_size] for i in beg_idx[::-1]]
            seqs = [s for s in seqs if len(s) > 0]
            if len(seqs):
                windows.extend(seqs)

        # self.uis = ui_map

        # Initialize dataset and dataloader
        dataset = BertTrainDataset(windows, n_items, n_items, self._random, self._max_sequence_size)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers, drop_last=True)
        n_ratings = train_g.num_edges() // 2 // self.batch_size + 1
        for e in range(first_epoch, final_epoch):
            self._model.train()
            dataset.reset_rng()
            tot_losses = 0
            tot_acc = 0
            n_unique = 0
            if self._no_improvements < self._early_stopping:
                with tqdm(dataloader, total=min(n_ratings, len(dataloader)), disable=not is_debug_mode()) as progress:
                    for i, batch in enumerate(progress, 1):
                        cur_batch, logits, = batch
                        cur_batch, logits = cur_batch.to(self.device), logits.to(self.device)
                        masks = (logits >= 0)
                        pred = self._model.embedder(cur_batch, masks)
                        preds = pred.reshape(-1, pred.size(-1))
                        logits = logits[masks]
                        acc = torch.sum(torch.argmax(preds, dim=1) == logits) / logits.numel()

                        # TODO: Should have a mask ensuring only train items are used
                        loss = ce(preds, logits.flatten())
                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)

                        self._optimizer.step()
                        self._optimizer.zero_grad()
                        self._scheduler.step()

                        tot_losses += loss.detach()
                        tot_acc += acc.detach()
                        cur_n_unique = len(torch.argmax(preds, dim=1).unique())
                        n_unique += cur_n_unique

                        progress.set_description(f'Epoch {e}, CFL: {tot_losses / i:.5f}, '
                                                 f'acc: {tot_acc / i:.5f}, '
                                                 f'u: {n_unique / i:.5f}, '
                                                 f'cu: {cur_n_unique}')
            elif trial is None:  # Skip last iterations as irrelevant
                break

            self._model.eval()
            with torch.no_grad():
                self._to_report(first_epoch, final_epoch, e, validator, trial, tot_losses)

    def _inference(self, **kwargs):
         if self.uis is None:
            (_, val_g), = self.graphs
            n_entities = len(self.meta.entities)
            s, d, t = val_g.edges('all')
            mask = s >= n_entities
            s = s[mask] - n_entities
            time_map = {(s, d): t for s, d, t in zip(s.tolist(), d[mask].tolist(), self._get_rating_time(val_g, mask))}
            ui_map = {}
            for u in self.meta.users:
                rated = val_g.out_edges(u+n_entities)[1].tolist()
                rated = list(sorted(rated, key=lambda x: time_map[(u, x)]))
                ui_map[u] = rated  # update with sorted

            self.uis = ui_map

    def predict_all(self, users) -> np.array:
        tokens = np.zeros((len(users), self._max_sequence_size), dtype=np.int64) - 1
        last_indices = np.zeros(len(users), dtype=np.int64)
        for i, u in enumerate(users):
            items = self.uis[u]
            items = items if len(items) < self._max_sequence_size else items[-self._max_sequence_size+1:]
            tokens[i, :len(items)] = items
            tokens[i, len(items)] = len(self.meta.items)
            last_indices[i] = len(items)

        tokens = torch.LongTensor(tokens).to(self.device) + 1
        last_indices = torch.LongTensor(last_indices).to(self.device)
        items = torch.LongTensor(self.meta.items).to(self.device)
        with torch.no_grad():
            mask = torch.zeros_like(tokens)
            mask[torch.arange(mask.size(0)), last_indices] = 1
            preds = self._model.embedder(tokens, mask.to(torch.bool))
            preds = preds[:, items]

        return preds.cpu().numpy()

    def set_parameters(self, parameters):
        self.learning_rate = parameters['learning_rates']
        self.att_dropout = parameters['att_dropouts']
        self.activation = parameters['activations']
        self.dropout = parameters['dropouts']
        self.dim = parameters['dims']
        self.normalization_range = parameters['normalization_ranges']
        self.layer_dim = parameters['layer_dims']
        self.max_position_embeddings = parameters['max_position_embeddings']
        self.num_attention_heads = parameters['num_attention_heads']
        self.num_hidden_layers = parameters['num_hidden_layers']
