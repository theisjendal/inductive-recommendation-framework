from typing import List, Dict

from shared.entity import Entity
from shared.enums import Sentiment
from shared.relation import Relation


class Meta:
    def __init__(self, entities: List[Entity], users: list, relations: List[Relation],
                 sentiment_utility: Dict[Sentiment, float], rated=None):
        self.entities = [entity.index for entity in entities]
        rated = {item for user in users for item, _ in user.ratings} if rated is None else rated
        self.items = [entity.index for entity in entities if entity.recommendable and entity.index in rated]
        self.users = [user.index for user in users]
        self.relations = relations
        self.sentiment_utility = sentiment_utility
