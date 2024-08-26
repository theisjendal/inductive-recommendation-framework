import argparse
import os
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from typing import List

import numpy as np
from tqdm import tqdm

from evaluate.dgl_evaluator import get_experiment
from evaluate.metric_calculator import _pickle_load_users
from shared.enums import Sentiment
from shared.experiments import Fold
from shared.user import User
from shared.utility import get_experiment_configuration

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='../datasets')
parser.add_argument('--result_path', type=str, default='../results')
parser.add_argument('--experiment', type=str, default='ml_1m_temporal')
parser.add_argument('--models', type=str, nargs='+', default=['simplerec'])
parser.add_argument('--ext', type=str, default=None)
parser.add_argument('--settings', type=str, nargs='+', default=['standard', 'cold_user', 'warm_user',
                                                                'cold_item', 'cold_item_limit', 'cold_user_item',
                                                                'cold_no_item_ratings'])


def old_split(user, warm_users, warm_items, rated_items, settings):
    uid = user['user']
    predicted = np.array(user['predicted'])
    relevance = np.array(user['relevance'])
    utility = user['utility']
    n_liked = user['n_liked']
    warm = None
    cold_user = None
    cold_item = None
    cold_item_limit = None
    cold_user_item = None
    cold_no_r = None

    if 'items' not in user:
        utility = np.array(utility)
        c = sum(utility)
        is_old = True
    else:
        c = len(utility[1])  # Liked ratings

    assert c == n_liked, f'Count {c} != n_liked {n_liked}'

    warm_items_filter = np.isin(predicted, warm_items)
    rated_items_filter = np.isin(predicted, rated_items)

    is_warm = user['user'] in warm_users
    if is_warm and 'warm_user' in settings:
        warm = pickle.dumps(user)
    elif not is_warm and 'cold_user' in settings:
        cold_user = pickle.dumps(user)

    utility[warm_items_filter] = False
    relevance[warm_items_filter] = 0
    count = sum(utility)
    if count > 0:
        if 'cold_item' in settings:
            cold_item = pickle.dumps({'user': uid,
                   'predicted': predicted.tolist(),
                   'relevance': relevance.tolist(), 'utility': utility.tolist(),
                   'n_liked': count})
        if 'cold_item_limit' in settings:
            cold_item_limit = pickle.dumps({'user': uid,
                   'predicted': predicted[~warm_items_filter].tolist(),
                   'relevance': relevance[~warm_items_filter].tolist(),
                   'utility': utility[~warm_items_filter].tolist(),
                   'n_liked': count})
        if not is_warm and 'cold_user_item' in settings:
            cold_user_item = pickle.dumps({'user': uid,
               'predicted': predicted[~warm_items_filter].tolist(),
               'relevance': relevance[~warm_items_filter].tolist(),
               'utility': utility[~warm_items_filter].tolist(),
               'n_liked': count})
    utility[rated_items_filter] = False
    relevance[rated_items_filter] = 0
    count = sum(utility)
    if count > 0 and 'cold_no_item_ratings' in settings:
        l = len(utility[~warm_items_filter].tolist())
        if l <= 50:
            print(l)
        cold_no_r = pickle.dumps({'user': uid,
               'predicted': predicted[~warm_items_filter].tolist(),
               'relevance': relevance[~warm_items_filter].tolist(),
               'utility': utility[~warm_items_filter].tolist(),
               'n_liked': count})

    return warm, cold_user, cold_item, cold_item_limit, cold_user_item, cold_no_r


def new_split(user, warm_users, warm_items, rated_items, settings):
    uid = user['user']
    predicted = np.array(user['predicted'])
    relevance = np.array(user['relevance'])
    utility = user['utility']
    items = user['items']
    cold_preds = user['cold_pred']
    warm, cold_user, cold_item, cold_item_limit, cold_user_item, cold_no_r = None, None, None, None, None, None

    if uid in warm_users and 'warm_user' in settings:
        warm = pickle.dumps(user)
    elif 'cold_user' in settings:
        cold_user = pickle.dumps(user)

    # Filter out warm items
    cold_info = {k: (np.array(v)[mask], np.array(utility[k])[mask]) for k, v in items.items() if (mask := ~np.isin(v, warm_items)).any()}

    # compute relevance and count
    inner_u = {k: v.tolist() for k, (_, v) in cold_info.items()}
    inner_r = list(sorted([r for _, (_, rels) in cold_info.items() for r in rels]))

    count = len(inner_u.get(1, []))

    if count <= 0:
        return warm, cold_user, cold_item, cold_item_limit, cold_user_item, cold_no_r

    # Subsequent partitions are only relevant if cold item ratings are present

    if 'cold_item' in settings:
        cold_item = pickle.dumps({'user': uid,
                                  'predicted': predicted.tolist(),
                                  'relevance': inner_r,
                                  'utility': inner_u,
                                  'n_liked': count})

    # Recompute positions based on cold predictions. I.e., we limit negative set to be only cold items, we need to shift
    # ranking positions accordingly
    # pred, id
    # rel, index
    # utility, dict index
    # items, dict id
    # cold_pred, id
    u = {k: np.argwhere(np.isin(cold_preds, v)).flatten().tolist() for k, v in items.items()}
    rel = np.sort(np.concatenate(list(u.values()))).tolist()
    assert len(inner_u[1]) == count

    if uid not in warm_users and 'cold_user_item' in settings:
        cold_user_item = pickle.dumps({'user': uid,
                                      'predicted': cold_preds,
                                      'relevance': rel,
                                      'utility': u,
                                      'n_liked': count})

    if 'cold_item_limit' in settings:
        cold_item_limit = pickle.dumps({'user': uid,
                                      'predicted': cold_preds,
                                      'relevance': rel,
                                      'utility': u,
                                      'n_liked': count})

    # Find if any rated item has no ratings
    inner_i = {k: np.array(v)[~np.isin(v, rated_items)].tolist() for k, v in items.items()}

    # Use this subset to get predictions
    u = {k: np.argwhere(np.isin(cold_preds, v)).flatten().tolist() for k, v in inner_i.items()}
    rel = np.sort(np.concatenate(list(u.values()))).tolist()
    count = len(u[1])
    if count > 0 and 'cold_no_item_ratings' in settings:
        cold_no_r = pickle.dumps({'user': uid,
                                  'predicted': cold_preds,
                                  'relevance': rel,
                                  'utility': u,
                                  'n_liked': count})

    return warm, cold_user, cold_item, cold_item_limit, cold_user_item, cold_no_r


def process_fold(train, test, validation, result_path, file_name, start, stop, settings):
    warm_users = [u.index for u in train + validation]
    warm_items = list(set([i for u in train + validation for i, _ in u.ratings]))
    rated_items = list(set([i for u in train + validation + test for i, _ in u.ratings]))
    warm_set = []
    cold_item_set = []
    cold_user_set = []
    cold_item_limit_set = []
    cold_user_item_set = []
    cold_no_r_set = []
    names = ['warm_user', 'cold_user', 'cold_item', 'cold_item_limit', 'cold_user_item', 'cold_no_item_ratings']
    with open(os.path.join(result_path, file_name), 'rb') as f:
        i = 0
        # Skip to start
        while i < start:
            pickle.load(f)
            i += 1

        while True:
            if i >= stop:
                break
            i += 1

            try:
                user = pickle.load(f)
                if 'items' not in user:
                    res = old_split(user, warm_users, warm_items, rated_items, settings)
                else:
                    assert len(user['cold_pred']) >= 50
                    res = new_split(user, warm_users, warm_items, rated_items, settings)

                for r, name, set_ in zip(res, names, [warm_set, cold_user_set, cold_item_set, cold_item_limit_set,
                                                      cold_user_item_set, cold_no_r_set]):
                    if name in settings and r is not None:
                        set_.append(r)
            except EOFError:
                break

        return_ = [warm_set, cold_user_set, cold_item_set, cold_item_limit_set, cold_user_item_set, cold_no_r_set]
        return_ = [(set_, name) for set_, name in zip(return_, names) if name in settings]

        return return_


def parallel_data_iterator(train, test, validation, result_path, file_name, settings):
    num_users = len(_pickle_load_users(os.path.join(result_path, file_name)))
    futures = []
    batch_size = 512

    with ProcessPoolExecutor() as executor:
        for i in tqdm(range(0, num_users, batch_size), desc='Submitting processes'):
            # r = process_fold(train, test, validation, result_path, file_name, i, i+batch_size, settings)
            # for res in r:
            #     yield res
            futures.append(executor.submit(process_fold, train, test, validation, result_path, file_name, i, i+batch_size, settings))

        # Maintain order of results
        for future in tqdm(futures, total=len(futures), desc='Processing results'):
            for res, name in future.result():
                yield res, name

    # for future in tqdm(futures, desc='Yielding results'):
    #     for res, name in zip(future.result(), ['warm_users', 'cold_users', 'cold_items']):
    #         yield res, name


def run(data_path, result_path, experiment, models, ext, settings):
    # get experiment
    experiment_conf = get_experiment_configuration(experiment)
    experiment = get_experiment(experiment, data_path)
    for fold in experiment.folds():
        train: List[User]
        train = fold.data_loader.training()
        validation = fold.data_loader.validation()
        test = fold.data_loader.testing()
        for model in models:
            count = defaultdict(int)
            print(f'Processing {model} for {fold.experiment.name}/{fold.name}')
            exp_path = os.path.join(result_path, experiment.name, model)
            model_name = model if ext is None else f'{model}_{ext}'
            file_name = f'{fold.name}_{model_name}_predictions.pickle'
            file_cold_user_name = os.path.join(exp_path, f'{fold.name}_{model_name}_predictions_cold_user.pickle')
            file_warm_user_name = os.path.join(exp_path, f'{fold.name}_{model_name}_predictions_warm_user.pickle')
            file_cold_item_name = os.path.join(exp_path, f'{fold.name}_{model_name}_predictions_cold_item.pickle')
            file_cold_item_limit_name = os.path.join(exp_path, f'{fold.name}_{model_name}_predictions_cold_item_limit.pickle')
            file_cold_user_item_name = os.path.join(exp_path, f'{fold.name}_{model_name}_predictions_cold_user_item.pickle')
            file_cold_no_r_name = os.path.join(exp_path, f'{fold.name}_{model_name}_predictions_cold_no_item_ratings.pickle')
            open_fn = lambda n, s: open(n, 'wb') if s in settings else nullcontext()  # Only open if using setting.
            with open_fn(file_warm_user_name, 'warm_user') as f_wu, \
                    open_fn(file_cold_user_name, 'cold_user') as f_cu, \
                    open_fn(file_cold_item_name, 'cold_item') as f_ci, \
                    open_fn(file_cold_item_limit_name, 'cold_item_limit') as f_cil, \
                    open_fn(file_cold_user_item_name, 'cold_user_item') as f_cui, \
                    open_fn(file_cold_no_r_name, 'cold_no_item_ratings') as f_cnr:
                for res, name in parallel_data_iterator(train, test, validation, exp_path, file_name, settings):
                    if name == 'warm_user' and 'warm_user' in settings:
                        count[name] += len(res)
                        f_wu.writelines(res)
                    elif name == 'cold_user' and 'cold_user' in settings:
                        count[name] += len(res)
                        f_cu.writelines(res)
                    elif name == 'cold_item' and 'cold_item' in settings:
                        count[name] += len(res)
                        f_ci.writelines(res)
                    elif name == 'cold_item_limit' and 'cold_item_limit' in settings:
                        count[name] += len(res)
                        f_cil.writelines(res)
                    elif name == 'cold_user_item' and 'cold_user_item' in settings:
                        count[name] += len(res)
                        f_cui.writelines(res)
                    elif name == 'cold_no_item_ratings' and 'cold_no_item_ratings' in settings:
                        count[name] += len(res)
                        f_cnr.writelines(res)
            print(f'For model {model} found {count}')


if __name__ == '__main__':
    args = parser.parse_args()
    run(**vars(args))