from collections import Counter
from datetime import date

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from typing import List
import time

from tqdm import tqdm

from datasets.partitioners.utilities import set_seeds, get_ratings_matrix
from shared.configuration_classes import ExperimentConfiguration
from shared.entity import Entity
from shared.enums import Sentiment
from shared.relation import Relation
from shared.seed_generator import SeedGenerator
from shared.user import User, LeaveOneOutUser


def create_loo_users(experiment: ExperimentConfiguration, rating_matrix: np.ndarray, seen_ratings: np.ndarray, unique_tuple):
    rating_matrix = rating_matrix[np.lexsort((rating_matrix[:, -1], rating_matrix[:, 0]))]
    unique, indices, counts = np.unique(rating_matrix[:, 0], return_counts=True, return_index=True)
    rating_matrix = rating_matrix[:, 1:]
    warm_user_map = {uid: (index, count) for uid, index, count in zip(*unique_tuple)}
    cold_user_map = {uid: (index, count) for uid, index, count in zip(unique, indices, counts)}

    # Create users
    users = {}
    i = 0
    for uid in unique:
        if uid not in warm_user_map:
            index, count = cold_user_map[uid]
            if count <= 3:
                continue

            # Split ratings into seen and unseen using a minimum of 3 ratings.
            num_ratings = int((experiment.test_size) * count / 100)
            num_ratings = num_ratings if experiment.max_inf_ratings == -1 \
                else min(num_ratings, experiment.max_inf_ratings)
            num_ratings = max(3, num_ratings)

            # Inference ratings and times
            sr = rating_matrix[index:index + num_ratings, [0, 1]]  # Seen ratings
            st = rating_matrix[index:index + num_ratings, [0, -1]]  # Seen ratings time

            # Test ratings
            ur = rating_matrix[index + num_ratings:index + count, [0, 1]]  # Unseen ratings (for evaluation)
        else:
            seen_index, seen_count = warm_user_map[uid]
            index, count = cold_user_map[uid]
            sr = seen_ratings[seen_index:seen_index + seen_count, [1, 2]]
            st = seen_ratings[seen_index:seen_index + seen_count, [1, -1]]
            ur = rating_matrix[index:index + count, [0, 1]]

        # Ignore negative ratings.
        mask = ur[:, 1] == experiment.dataset.sentiment_utility[Sentiment.POSITIVE]
        ur = ur[mask]

        users[i] = LeaveOneOutUser(
            uid,
            list(map(tuple, sr.tolist())),
            list(map(tuple, ur.tolist())),
            list(map(tuple, st.tolist()))
        )
        i += 1

    return users


def _plotting(users, train, validation, test):
    factor = ((60*60*24*365)/12)
    rating_times = [int(t / factor) for u in users for _, t in u.rating_time]
    rating_spans = []
    for user in users:
        rt = [t for _, t in user.rating_time]
        span = max(rt) - min(rt)
        rating_spans.append(span / (60 * 60 * 24))
    rating_spans = sorted(rating_spans)
    plt.plot(rating_spans)
    plt.yscale('log')
    plt.show()

    rtc = Counter(rating_times)
    plt.axvline(min(validation['time'] / factor))
    plt.axvline(min(test['time'] / factor))
    x, y = zip(*sorted(rtc.items()))
    plt.plot(x, y)
    plt.show()

    wip = Counter(train['item'].tolist())
    vip = Counter(validation['item'].tolist())
    tip = Counter(test['item'].tolist())

    pop_over_time = {}
    for f in range(min(rtc), max(rtc)+1):
        rated_items = [i for u in users for i, t in u.rating_time if int(t / factor) == f]
        pop_over_time[f] = Counter(rated_items)

    most_popular = set.union(*[set(i for i, _ in c.most_common(2)) for c in pop_over_time.values()])
    times = sorted(pop_over_time.keys())
    items_over_time = [(i, [pop_over_time[t].get(i, 0) for t in times]) for i in most_popular]
    for i, p in items_over_time:
        plt.plot(times, p, label=i)
    plt.legend()
    plt.show()

    for name, c in [('train', wip), ('validation', vip), ('test', tip)]:
        x, y = zip(*sorted(c.items(), key=lambda x: (x[1], x[0]), reverse=True))
        plt.plot(range(len(x)), y)
        plt.title(name)
        plt.yscale('log')
        plt.show()


def fold_data_iterator(path, experiment: ExperimentConfiguration, kfold: KFold, entities: List[Entity],
                           users: List[User], relations: List[Relation], sg: SeedGenerator):
    # This only creates one fold due to stratification over time.
    set_seeds(sg)
    ratings = []
    for i, user in enumerate(tqdm(users, desc='Creating ratings matrix')):
        for item, rating in user.ratings:
            rating_time = {item: t for item, t in user.rating_time}
            ratings.append((user.index, item, rating, rating_time[item]))

    a = np.array(ratings, dtype=[('user', int), ('item', int), ('rating', int), ('time', int)])
    a = np.sort(a, order=['time', 'user', 'item', 'rating'])

    # Assert no duplicates in
    if len(np.unique(np.stack([a['user'], a['item']]).T, axis=0)) != len(a):
        logger.warning('Duplicates in dataset, may be an error in conversion.')

    # Split dataset by time
    train, test = np.split(a, [int((100 - experiment.test_size) * len(a) / 100)])
    validation, test = np.split(test, [int((experiment.validation_size) * len(test) / 100)])
    for a, b in [(train, validation), (validation, test)]:
        assert max(a['time']) <= min(b['time'])

    # plot if needed
    # _plotting(users, train, validation, test)

    train_users = set(train['user'])
    train_items = set(train['item'])

    logger.info(f'Train, total number of users: {len(train_users)}, items: {len(train_items)}')
    logger.info(f'Val, total number of users: {len(set(validation["user"]))}, items: {len(set(validation["item"]))}')
    logger.info(f'Test, total number of users: {len(set(test["user"]))}, items: {len(set(test["item"]))}')

    # Test amount of new users in validation and test sets
    val_n = len(set(validation['user']).difference(set(train['user'])))
    test_n = len(set(test['user']).difference(set(train['user'])))

    logger.info(f'New users found in: validation: {val_n}, test: {test_n}')

    # Test amount of new items in validation an test sets
    val_n = len(set(validation['item']).difference(set(train['item'])))
    test_n = len(set(test['item']).difference(set(train['item'])))

    logger.info(f'New items found in: validation: {val_n},  test: {test_n}')

    # Transform structured arrays to normal numpy arrays
    train, validation, test = [x.view((np.int64, len(x.dtype.names))) for x in [train, validation, test]]

    train = train[np.lexsort((train[:, -1], train[:, 0]))]
    unique, indices, counts = np.unique(train[:, 0], return_counts=True, return_index=True)
    unique_tuple = unique, indices, counts

    logger.info('Creating train users')
    users = {user.index: user for user in users}
    # Set ratings for test and validation sets
    for i, uid in enumerate(unique):
        user = users[uid]
        index, count = indices[i], counts[i]
        user.ratings = [tuple(t) for t in train[index:index + count, [1, 2]].tolist()]
        if hasattr(user, 'rating_time'):
            user.rating_time = [tuple(t) for t in train[index:index + count, [1, -1]].tolist()]

    # Remove users not in train set
    for user in set(users.keys()).difference(set(unique)):
        users[user].ratings = []
        if hasattr(users[user], 'rating_time'):
            users[user].rating_time = []

    w = [u for i, u in sorted(users.items()) if len(u.ratings) > 0]
    yield w

    logger.info('Creating validation users')
    v = [u for i, u in sorted(create_loo_users(experiment, validation, train, unique_tuple).items())]
    yield v

    # Assume later timestep, therefore add validation ratings to 'train' set.
    train = np.concatenate((train, validation))
    train = train[np.lexsort((train[:, -1], train[:, 0]))]
    unique, indices, counts = np.unique(train[:, 0], return_counts=True, return_index=True)
    unique_tuple = unique, indices, counts

    logger.info('Creating test users')
    t = [u for i, u in sorted(create_loo_users(experiment, test, train, unique_tuple).items())]
    yield t
