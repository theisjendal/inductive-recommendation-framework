from typing import Dict, Callable, List, Union

from shared.enums import Sentiment, Evaluation, ExperimentEnum, FeatureEnum

_DEFAULT_SENTIMENT = {
    Sentiment.NEGATIVE: -1,
    Sentiment.POSITIVE: 1,
    Sentiment.UNSEEN: 0,
}


class CountFilter:
    def __init__(self, filter_func: Callable[[int], bool], sentiment: Sentiment):
        self.sentiment = sentiment
        self.filter_func = filter_func


class ConfigurationBase:
    def __init__(self, name: str, seed: int):
        """
        Base class for configurations. Implements a method for getting instance, e.g., getting a dataset instance given a dataset configuration.
        :param name: Should be unique for all configurations
        """
        self.name = name
        self.seed = seed


class DatasetConfiguration(ConfigurationBase):
    def __init__(self, name, ratings_mapping, sentiment_utility: Dict[Sentiment, float] = None,
                 filters: List[CountFilter] = None, max_users=None, time_based_sampling=False,
                 time_based_pruning=False, max_ratings=None, seed=42, k_core=10, prune_duplicates=True,
                 is_relation_type=None, remove_unseen=False):
        """
        Dataset configurations
        :param name: the dataset name, also used for storing data.
        :param ratings_mapping: Maps ratings to other rating scale.
        :param sentiment_utility: Map sentiment to same rating scale as above, default is seen in enums under shared.
        Preferably ensure reverse mapping is possible, i.e., it is bijective.
        :param max_total_ratings: Max number of positive ratings. Higher leads to slow training for multiple models.
        """
        super().__init__(name, seed)
        self.ratings_mapping = ratings_mapping
        self.sentiment_utility = _DEFAULT_SENTIMENT if sentiment_utility is None else sentiment_utility
        self.filters = [] if filters is None else filters
        self.max_users = max_users
        self.time_based_sampling = time_based_sampling
        self.time_based_pruning = time_based_pruning
        self.max_ratings = max_ratings
        self.k_core = k_core
        self.prune_duplicates = prune_duplicates  # Ignores time
        self.is_relation_type = [] if is_relation_type is None else is_relation_type
        self.remove_unseen = remove_unseen


class ExperimentConfiguration(ConfigurationBase):
    def __init__(self, name: str, dataset: DatasetConfiguration, experiment: ExperimentEnum, folds: int = 5,
                 test_size: int = 20, validation_size: int = 50, evaluation: Evaluation = Evaluation.LEAVE_ONE_OUT,
                 seed: int = 42, max_inf_ratings=-1):
        """
        Experiment configuration
        :param name: the name of the experiment such as warm-start.
        :param dataset: dataset to work on.
        :param folds: number of folds to make.
        :param test_size: size of the test set - only used if folds == 1. and experiment is temporal.
        :param validation_size: size of the validation set (taken from the test set).
        :param evaluation: type of evaluation to make, i.e leave one out or other.
        :param seed: seed to use.
        """
        super().__init__(name, seed)
        self.dataset = dataset
        self.experiment = experiment
        self.folds = 1 if self.experiment == ExperimentEnum.TEMPORAL else folds
        self.test_size = test_size
        self.validation_size = validation_size
        self.evaluation = evaluation
        self.max_inf_ratings = max_inf_ratings


class FeatureConfiguration(ConfigurationBase):
    def __init__(self, name: str, extractor: Union[str, None], graph_name: str, require_ratings: bool,
                 feature_span: FeatureEnum, scale: bool = False, seed: int = 42, **kwargs):
        """
        Configuration of the feature extractor. Takes a name, extractor and if it uses ratings.
        If using ratings, then the extracted features are split based.
        :param name: the name of the configuration.
        :param extractor: the name of the feature extractor to use.
        :param graph_name: name of the graph to use (see experiment_to_dgl under datasets/converts for all possible graphs).
        :param require_ratings: whether the current feature require training ratings.
        :param feature_span: defines for which nodes features should be extracted to.
        :param scale: whether to scale the feature or not.
        :param **kwargs: parameters to pass to feature extractor.
        """
        super().__init__(name, seed)
        self.extractor = extractor
        self.graph_name = graph_name
        self.require_ratings = require_ratings
        self.feature_span = feature_span  # Maybe not needed
        self.scale = scale
        self.kwargs = kwargs
