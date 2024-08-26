from copy import deepcopy
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from shared.enums import Sentiment
from shared.meta import Meta
from shared.user import User
from shared.utility import is_debug_mode


class BayesianDataset(Dataset):
    def __init__(self, train: List[User], meta: Meta, seed, num_neg_samples: int = 1):
        pos_rating = meta.sentiment_utility[Sentiment.POSITIVE]
        self._train = torch.LongTensor([[user.index, item] for user in train for item, rating in user.ratings
                                        if rating == pos_rating])
        self._len = len(self._train)
        self._items = torch.LongTensor(meta.items)
        self._seed = seed
        self._num_neg_samples = num_neg_samples

    def __len__(self):
        return self._len * self._num_neg_samples

    def __getitem__(self, index):
        index = index % self._len
        sample = self._train[index]
        neg_sample = self._items[torch.randint(len(self._items), (1, ))]
        item = torch.cat((sample, neg_sample))
        return item

    def fit(self, model, optimizer, batch_size, validator, validation,
                 shuffle=True, num_workers=0, epochs=1000, early_stopping=50, clipping=None):
        loader = DataLoader(self, batch_size, shuffle=shuffle, num_workers=num_workers)
        desc = 'Epoch {0:4d}, Loss {1:6.5f}'
        desc_val = desc + ', Val {2:6.5f}'
        no_improvement = 0
        state = None
        best = 0
        for epoch in range(epochs):
            losses = []
            with tqdm(enumerate(loader), total=len(loader), disable=not is_debug_mode()) as pbar:
                model.train()
                for i, data in pbar:
                    optimizer.zero_grad()
                    users, items_i, items_j = torch.transpose(data, 0, 1)

                    loss = model.loss(users, items_i, items_j)

                    loss.backward()
                    optimizer.step()

                    if len(losses) >= 20:
                        losses.pop(0)

                    losses.append(loss.item())
                    loss_avg = sum(losses) / len(losses)

                    pbar.set_description(desc.format(epoch, loss_avg))

                    if len(loader) <= (i + 1):
                        with torch.no_grad():
                            model.eval()
                            score = validator(validation, batch_size)

                        pbar.set_description(desc_val.format(epoch, loss_avg, score))

                if score > best:
                    no_improvement = 0
                    state = deepcopy(model.state_dict())
                    best = score
                elif no_improvement >= early_stopping:
                    break
                else:
                    no_improvement += 1

        return best, state
