import dgl
import numpy as np
import torch

from models.dgl_recommender_base import RecommenderBase
from models.ppr.ppr import PersonalizedPagerank
from shared.efficient_validator import Validator
from shared.enums import RecommenderEnum


class PPRDGLRecommender(RecommenderBase):
    def __init__(self, **kwargs):
        super().__init__(RecommenderEnum.FULL_COLD_START, **kwargs)
        self.alpha = 0.85
        
    def _create_model(self, trial):
        self.set_seed()

        # Only use test graph as we do not need to train anything.
        (_, g), = self.graphs
        g = dgl.to_networkx(g)
        
        self._model = PersonalizedPagerank(g, self.alpha)
        self._model.forward.cache_clear()
        
        self._sherpa_load(trial)

    def fit(self, validator: Validator):
        super(PPRDGLRecommender, self).fit(validator)

        self._model.forward.cache_clear()

    def _fit(self, validator: Validator, first_epoch, final_epoch=1000, trial=None):
        self._to_report(first_epoch, final_epoch, 0, validator, trial)

    def _inference(self, **kwargs):
        pass

    def _predict(self, source_nodes, items):
        scores = self._model(tuple(source_nodes))

        return {item: scores.get(item, 0) for item in items}

    def predict_all(self, users) -> np.array:
        array = np.zeros((len(users), len(self.meta.items)))
        (_, g), = self.graphs
        u, v = g.out_edges(users+len(self.meta.entities), form='uv')
        unique, counts = torch.unique(u, return_counts=True)
        for i, count in enumerate(counts):
            rated = v[i:i+count]
            scores = self._predict(rated.cpu().tolist(), self.meta.items)

            for j, item in enumerate(self.meta.items):
                array[i][j] = scores[item]

        return array

    def set_parameters(self, parameters):
        self.alpha = parameters['alphas']
