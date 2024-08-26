import random

import dgl
import numpy as np
from loguru import logger

from models.dgl_recommender_base import RecommenderBase
from shared.efficient_validator import Validator
from shared.enums import RecommenderEnum


class RandomDGLRecommender(RecommenderBase):
    def __init__(self, **kwargs):
        super().__init__(RecommenderEnum.FULL_COLD_START, **kwargs)

    def _create_model(self, trial):
        pass

    def fit(self, validator: Validator):
        self.set_seed()

        self._model = np.random.RandomState(self._seed)
        score = validator.validate(self)
        logger.debug(f'Val: {score}')

    def _fit(self, validator: Validator, first_epoch, final_epoch=1000, trial=None):
        pass

    def _inference(self, **kwargs):
        pass

    def predict_all(self, users) -> np.array:
        predictions = self._model.rand(len(users), len(self.meta.items))

        return predictions

    def set_seed(self):
        random.seed(self._seed)
        np.random.seed(self._seed)
        dgl.seed(self._seed)
        dgl.random.seed(self._seed)

    def set_parameters(self, parameters):
        pass