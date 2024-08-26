import random
from typing import List, Dict
import itertools

import pandas as pd

from shared.enums import Sentiment


class Ranking:
    def __init__(self, seed: int):
        self._rand = random.Random(seed)
        self.sentiment_samples = dict()
        self._list = None

    def get_seen_samples(self):
        """
        Get all seen samples
        @return: Seen sample
        """
        return set(itertools.chain.from_iterable(
            [value for key, value in self.sentiment_samples.items() if key != Sentiment.UNSEEN]))

    def to_list(self, neg_samples=100) -> List[int]:
        """
        Get sentiment values and shuffles them, to insure no bias is created
        @return: a list of sentiment values
        """
        if self._list is None:
            items = self.sentiment_samples[Sentiment.UNSEEN]

            if neg_samples is not None:
                self._rand.shuffle(items)

            self.sentiment_samples[Sentiment.UNSEEN] = items[:neg_samples]
            as_list = list(itertools.chain.from_iterable(self.sentiment_samples.values()))

            # Shuffle to avoid any bias arising from positive sample being first or last
            self._rand.shuffle(as_list)
            self._list = as_list

        return self._list

    def _get_utility(self, entity_idx, sentiment_utility: Dict[Sentiment, float]) -> float:
        for sentiment, utility in sentiment_utility.items():
            if entity_idx in self.sentiment_samples[sentiment]:
                return utility

        return 0

    def get_relevance(self, entity_indices: List[int]) -> List[bool]:
        return [entity_idx in self.sentiment_samples[Sentiment.POSITIVE] for entity_idx in entity_indices]

    def get_utility(self, entity_indices: List[int], sentiment_utility: Dict[Sentiment, float]) -> List[float]:
        utilities = []
        for sentiment, utility in sentiment_utility.items():
            df = pd.DataFrame(0., self.sentiment_samples[sentiment], columns=['sent'])
            df['sent'] = utility

            utilities.append(df)

        utilities = pd.concat(utilities)
        return utilities.loc[entity_indices]['sent'].tolist()
