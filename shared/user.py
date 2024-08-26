from typing import List, Tuple, Dict

from shared.enums import Sentiment
from shared.meta import Meta
from shared.ranking import Ranking


class UserBase:
    def __init__(self, index, ratings, rating_time=None):
        """
        The base class for all users. They must have ratings and an index.
        :param index: unique index of the user.
        :param ratings: list of item-rating tuples.
        :param rating_time: list of item-rating-time tuples. Assumes item can only be rated once.
        """
        self.index = index
        self.ratings = ratings if ratings else []
        self.rating_time = rating_time if rating_time else []


class User(UserBase):
    def __init__(self, original_id: str, index: int, ratings: List[Tuple[int, int]] = None,
                 rating_time: List[Tuple[int, int]] = None):
        """
        Creates a User object.
        :param original_id: original id in the source data.
        :param index: unique index of the user.
        """
        super().__init__(index, ratings, rating_time)
        self.original_id = original_id


class ColdStartUser(UserBase):
    def __init__(self, index, ratings,  evaluation_ratings):
        """
        Creates a cold start users, which contains a set of ratings that can be trained on, and one for evaluation.
        :param evaluation_ratings: ratings used for evaluation.
        """
        super().__init__(index, ratings)
        self.index = index
        self.evaluation_ratings = evaluation_ratings


class LeaveOneOutUser(UserBase):
    def __init__(self, index, ratings, items: List[Tuple[int, int]], rating_time: List[Tuple[int, int]] = None):
        """
        A user containing test information.
        :param items: items to evaluate for leave one out.
        """
        super().__init__(index, ratings, rating_time)
        self.loo = items
        self._ranking = None  # type: Ranking

    def _get_samples(self, meta: Meta) -> Dict[Sentiment, list]:
        # Initialize sentiment
        sentiments = {s: [] for s in meta.sentiment_utility.keys()}
        items = set(meta.items)

        # Get rated and unseen items
        rated = set([item for item, _ in self.ratings]).union([item for item, _ in self.loo])
        unseen = items.difference(rated)
        sentiments[Sentiment.UNSEEN].extend(unseen)

        # Add all matching sentiment value, if sentiment_utility is bijective, no overlap will occur.
        for sentiment, value in meta.sentiment_utility.items():
            sentiments[sentiment].extend([i for i, r in self.loo if r == value])

        return sentiments

    def get_ranking(self, seed: int, meta: Meta) -> Ranking:
        if self._ranking is None:
            self._ranking = Ranking(seed)
            self._ranking.sentiment_samples = self._get_samples(meta)

        return self._ranking


