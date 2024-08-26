from shared.configuration_classes import DatasetConfiguration, CountFilter
from shared.enums import Sentiment



# All users have at least 5 positive ratings, for 5 folds.
# lambda x: {x > 3: 1}.get(True, 0) if above 3 creates a dict with True:1, which we try to return, otherwise 0.

mindreader = DatasetConfiguration('mindreader', lambda x: {x > 0: 1}.get(True, 0),
                                  filters=[CountFilter(lambda c: c >= 5, sentiment=Sentiment.POSITIVE)])

movielens = DatasetConfiguration('movielens', lambda x: {x < 3: -1, x > 3: 1}.get(True, 0),
                                 filters=[CountFilter(lambda c: c >= 5, sentiment=Sentiment.POSITIVE)])
ml_mr = DatasetConfiguration('ml-mr', lambda x: {x > 3: 1}.get(True, 0),
                             filters=[CountFilter(lambda c: c >= 5, sentiment=Sentiment.POSITIVE)],
                             time_based_sampling=True, k_core=5)
ml_mr_1m = DatasetConfiguration('ml-mr-1m', lambda x: {x > 3: 1}.get(True, 0),
                                filters=[CountFilter(lambda c: c >= 5, sentiment=Sentiment.POSITIVE)],
                                max_users=None, max_ratings=3000000, time_based_sampling=True, time_based_pruning=True,
                                k_core=5)
amazon_book = DatasetConfiguration('amazon-book', lambda x: x, k_core=1)
yelp = DatasetConfiguration('yelpkg', lambda x: {x > 3: 1}.get(True, 0),
                            filters=[CountFilter(lambda c: c >= 5, sentiment=Sentiment.POSITIVE)],
                            k_core=5, time_based_sampling=True)

datasets = [mindreader, ml_mr, ml_mr_1m, amazon_book, yelp]
dataset_names = [d.name for d in datasets]