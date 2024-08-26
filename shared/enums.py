from enum import Enum, IntEnum


class Metric(Enum):
    HR = 1
    NDCG = 2
    SER = 3
    COV = 4
    TAU = 5


class Sentiment(IntEnum):
    NEGATIVE = 1
    UNKNOWN = 2
    POSITIVE = 3
    UNSEEN = 4
    ANY = 5


class Evaluation(IntEnum):
    LEAVE_ONE_OUT = 1


class ExperimentEnum(IntEnum):
    TEMPORAL = 1


class GraphTypesEnum(IntEnum):
    KNOWLEDGE_GRAPH = 1
    COLLABORATIVE_GRAPH = 2
    COLLABORATIVE_KNOWLEDGE_GRAPH = 3


class FeatureEnum(IntEnum):
    ITEMS = 1
    DESC_ENTITIES = 2
    ENTITIES = 3
    USERS = 4
    ALL = 5


class RecommenderEnum(IntEnum):
    WARM_START = 1
    USER_COLD_START = 2
    ITEM_COLD_START = 3
    FULL_COLD_START = 4