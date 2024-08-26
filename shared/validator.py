from statistics import mean
from typing import List, Tuple

from shared.meta import Meta
from shared.metrics import coverage, hr_at_k, ndcg_at_k, tau_at_k
from shared.enums import Metric
from shared.ranking import Ranking


class Validator:
    def __init__(self, metric: Metric, meta: Meta, cutoff: int = 10):
        self.metric = metric
        self.meta = meta
        self.cutoff = cutoff

    def score(self, predictions: List[Tuple[Ranking, List[int]]], reverse=True,
              metric: Metric = None, cutoff: int = None) -> float:
        covered = set()
        scores = list()

        # By default, use the metric and cutoff defined in the validator
        metric = metric if metric else self.metric
        cutoff = cutoff if cutoff else self.cutoff

        for ranking, ranked_list in predictions:
            prediction = ranking.get_relevance(ranked_list)

            if metric == Metric.COV:
                covered.update(set(prediction[:cutoff]))
            elif metric == Metric.HR:
                scores.append(hr_at_k(ranking.get_relevance(prediction), cutoff))
            elif metric == Metric.NDCG:
                scores.append(ndcg_at_k(ranking.get_utility(ranked_list, self.meta.sentiment_utility), cutoff))
            elif metric == Metric.TAU:
                scores.append(tau_at_k(ranking.get_utility(ranked_list, self.meta.sentiment_utility), cutoff))
            else:
                raise RuntimeError('Unsupported metric for validation.')

        if metric == Metric.COV:
            return coverage(covered, self.meta.items)
        else:
            return mean(scores) if scores else 0
