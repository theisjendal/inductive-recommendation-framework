from collections import defaultdict
import random

import dgl
import numpy as np
import torch
from loguru import logger

from models.dgl_recommender_base import RecommenderBase
from shared.efficient_validator import Validator
from shared.enums import RecommenderEnum


class DictWrapper(defaultdict):
    def state_dict(self):
        return self

    def load_state_dict(self, state_dict):
        self.update(state_dict)


class TopPopDGLRecommender(RecommenderBase):
    def __init__(self, **kwargs):
        super().__init__(RecommenderEnum.FULL_COLD_START, **kwargs)

        self.cut_off = 10  # is in months before last interaction
        self.factor = (365 * 24 * 60 * 60) / 12  # seconds to months

    def _create_model(self, trial):
        (_, g), = self.graphs
        u, v, eids = g.in_edges(self.meta.items, form='all')
        timestamps = g.edata['rating_time'][eids] / self.factor
        last_timestamp = timestamps.max()

        # Use all if cut_off is -1, else use cut_off
        if self.cut_off == -1:
            rating_filter = torch.ones_like(timestamps, dtype=torch.bool)
        else:
            rating_filter = timestamps > (last_timestamp - self.cut_off)

        u, v = u[rating_filter], v[rating_filter]
        unique, counts = torch.unique(v, return_counts=True)
        popularity = DictWrapper(int)
        for item, count in zip(unique.tolist(), counts.tolist()):
            popularity[item] = count

        self._model = popularity

    def fit(self, validator: Validator):
        # use test graph to get the popularity as no training is needed.
        self.set_seed()
        super(TopPopDGLRecommender, self).fit(validator)
        logger.debug(validator.validate(self))

    def _fit(self, validator: Validator, first_epoch, final_epoch=1000, trial=None):
        self._to_report(first_epoch, final_epoch, 0, validator, trial)

    def _inference(self, **kwargs):
        pass

    def predict_all(self, users) -> np.array:
        predictions = np.zeros((len(users), len(self.meta.items)))
        preds = [self._model.get(item, 0) for item in self.meta.items]

        predictions[:] = preds

        return predictions

    def set_seed(self):
        random.seed(self._seed)
        np.random.seed(self._seed)
        dgl.seed(self._seed)
        dgl.random.seed(self._seed)

    def set_parameters(self, parameters):
        self.cut_off = parameters.get('cut_off', 10)