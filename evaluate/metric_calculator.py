import argparse
import concurrent.futures
import multiprocessing
import pickle
import sys

import os
from collections import Counter, defaultdict
from typing import List

import numpy as np
from loguru import logger
from sklearn.metrics import auc, precision_recall_curve
from tqdm import tqdm
import configuration

from configuration.experiments import experiment_names
from configuration.features import feature_conf_names
from configuration.models import dgl_models
from evaluate.dgl_evaluator import get_experiment
from shared.configuration_classes import FeatureConfiguration
from shared.experiments import Dataset, Fold
from shared.metrics import hr_at_k, ndcg_at_k, recall_at_k, precision_at_k, coverage
from shared.utility import valid_dir, join_paths, get_feature_configuration
from train.dgl_trainer import get_model_name

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=valid_dir, help='path to datasets')
parser.add_argument('--results_path', type=valid_dir, help='path to store results')
parser.add_argument('--experiments', nargs='+', type=str, choices=experiment_names, help='name of experiment')
parser.add_argument('--include', nargs='+', choices=dgl_models.keys(), help='models to include')
parser.add_argument('--seed', default=42, type=int, help='seed for reproducibility')
parser.add_argument('--debug', action='store_true', help='enable debug mode')
parser.add_argument('--folds', nargs='*', default=None, help='folds to run')
parser.add_argument('--study', choices=['standard', 'user_sparsity', 'user_popularity', 'user_test_popularity',
                                        'fixed_neg_sampling', 'idcf_neg_sampling', 'user_group'],
                    default='standard', help='evaluation study to perform')
parser.add_argument('--state', choices=experiment_names, default=None,
                    help='use state from another experiment - useful for coldstart methods')
parser.add_argument('--parameter', choices=experiment_names, default=None,
                    help='use parameters from another experiment - useful for training with pre-determined parameters')
parser.add_argument('--other_model', choices=dgl_models.keys(), default=None,
                    help='use parameters from other method i.e. change one parameter')
parser.add_argument('--feature_configuration', nargs='+', choices=feature_conf_names, default=None,
                    help='feature_configuration to use.')
parser.add_argument('--experiment_setting', choices=['standard', 'cold_user', 'cold_item', 'warm_user',
                                                     'cold_item_limit', 'cold_user_item', 'cold_no_item_ratings'],
                    default='standard')


def _fixed_length_negative_sampler(predicted_ranking, relevance, utility, num_samples):
    neg_samples = {item for item, relevance in zip(predicted_ranking, relevance) if not relevance}
    neg_samples = np.random.choice(list(neg_samples), size=num_samples, replace=False)

    # Get indexes of items that are either in the negative subsamples or a positive item
    item_indices = {idx for idx, item in enumerate(predicted_ranking) if item in neg_samples or
                    relevance[idx]}

    # Update to only contain positive or sampled items.
    predicted_ranking = [item for idx, item in enumerate(predicted_ranking) if idx in item_indices]
    relevance = [rel for idx, rel in enumerate(relevance) if idx in item_indices]
    utility = [util for idx, util in enumerate(utility) if idx in item_indices]

    return predicted_ranking, relevance, utility


def _scaled_neg_sampler(predicted_ranking, relevance, utility, samples_per_rating):
    pos = [item for item, r in zip(predicted_ranking, relevance) if r]
    n_pos = len(pos)
    neg_iid = np.random.randint(0, len(predicted_ranking), (samples_per_rating * n_pos))
    neg_samples = [predicted_ranking[index] for index in neg_iid]
    item_rank = {item: rank for rank, item in enumerate(predicted_ranking)}
    predicted_ranking = sorted(pos + neg_samples, key=item_rank.get)
    relevance = [relevance[item_rank.get(item)] for item in predicted_ranking]
    utility = [utility[item_rank.get(item)] for item in predicted_ranking]

    return predicted_ranking, relevance, utility


def _group_users(testing):
    n_bins = 4

    testing = sorted(testing, key=lambda x: (len(x.ratings), x.index))
    bin_size = len(testing) // n_bins
    bins = []
    diff = len(testing) - bin_size*n_bins
    for i in range(n_bins):
        start = i*bin_size + (diff if i != 0 else 0)
        bins.append(testing[start:(i+1)*bin_size+diff])

    return bins


def _group_users_popularity(testing, popular=2, use_testing=False):
    counts = Counter([i for u in testing for i, _ in (u.loo if use_testing else u.ratings)])

    n_items = len(counts)
    k = int(n_items * (popular / 100.))

    top_k = sorted(counts, key=counts.get, reverse=True)[:k]

    n_bins = 4
    percentage_step = 100 / n_bins
    bins = defaultdict(list)

    # Get bin number based on percentage, i.e., bin_fn(70) => 2
    for user in testing:
        n_r = len(user.ratings)  # number of ratings
        tk = len([i for i, _ in user.ratings if i in top_k])  # number of popularity biased ratings
        p = (tk / n_r) * 100
        for i in range(1, n_bins+1):
            bin_step = percentage_step*i
            if p <= bin_step:
                bins[i].append(user)
                break

    return [bins[i] for i in range(1, n_bins+1)]


def _group_users_group(testing):
    # Sort according to number of ratings
    testing = sorted(testing, key=lambda u: len(u.ratings))

    tot = 0
    accumulated = {u: (tot := tot + len(u.ratings)) for u in testing}

    n_bins = 4
    bin_size = tot // n_bins
    bins = []
    cur = 0
    bin_no = -1
    for user, acc in accumulated.items():
        if acc <= cur or bin_no == n_bins-1:
            bins[bin_no].append(user)
        else:
            bins.append([user])
            cur += bin_size
            bin_no += 1

    return bins, {u.index: len(u.ratings) for u in testing}


def _converter(predicted_ranking, relevance, utility):
    max_i = max(max(relevance), len(predicted_ranking))
    rel = np.zeros(max_i+1)
    u = np.zeros(max_i+1, dtype=bool)
    rel[utility[1]] = 1
    u[utility[1]] = True
    return predicted_ranking, rel, u


def calculate_metrics(predicted_ranking, relevance, utility, n_liked, upper_cutoff=50, neg_sampling=None):
    # Metrics
    hits = defaultdict(list)
    ndcgs = defaultdict(list)
    covs = defaultdict(set)
    recall = defaultdict(list)
    max_recall = defaultdict(list)
    precision = defaultdict(list)
    upper_cutoff += 1
    k_range = upper_cutoff

    if isinstance(utility, dict):
        predicted_ranking, relevance, utility = _converter(predicted_ranking, relevance, utility)

    if neg_sampling is not None:
        # Get negative samples and subsample
        study, kwargs = neg_sampling
        if study == 'idcf_neg_sampling':
            predicted_ranking, relevance, utility = _scaled_neg_sampler(predicted_ranking, relevance,
                                                                        utility, **kwargs)
            upper_cutoff = len(utility)
            k_range = upper_cutoff + 1
            n_liked = sum(relevance)  # Negative samples might be a positive by random chance
        else:
            predicted_ranking, relevance, utility = _fixed_length_negative_sampler(predicted_ranking, relevance,
                                                                                   utility, **kwargs)

    inner_ndcg = ndcg_at_k(utility, upper_cutoff, n_liked)

    precisions, recalls, _ = precision_recall_curve(relevance, np.arange(len(relevance))[::-1])
    aucs = auc(recalls, precisions)
    for k in range(1, k_range):

        ranked_cutoff = predicted_ranking[:k]

        hits[k].append(hr_at_k(relevance, k))
        ndcgs[k].append(inner_ndcg[k-1])
        covs[k] = covs[k].union(set(ranked_cutoff))
        recall[k].append(recall_at_k(relevance, k, n_liked=n_liked))
        max_recall[k].append(recall_at_k(relevance, k, max_recall=True, n_liked=n_liked))
        precision[k].append(precision_at_k(relevance, k))

    return dict(hits), dict(ndcgs), dict(covs), dict(recall), dict(max_recall), dict(precision), aucs, n_liked


def _info_fetcher(fp, start_position=0, end_position=-1):
    with open(fp, 'rb') as f:
        cur_pos = 0
        while cur_pos != end_position:
            try:
                s = pickle.load(f)
                if cur_pos >= start_position:
                    s = {k: v for k, v in s.items() if k in ['user', 'predicted', 'relevance', 'utility',
                                                             'n_liked']}
                    yield s.values()
                cur_pos += 1
            except EOFError:
                break


def _info_fetcher_list(fp, users):
    min_max = users.copy()
    end_pos = max(min_max)
    min_pos = min(users)
    with open(fp, 'rb') as f:
        cur_pos = 0
        while cur_pos != end_pos:
            try:
                s = pickle.load(f)
                if cur_pos == min_pos:
                    min_max.remove(min_pos)
                    min_pos = min(min_max)
                    yield s.values()
                cur_pos += 1
            except EOFError:
                break


def iter_users(fp, group, neg_sampling=None):
    np.random.seed(42)
    metrics = {'user': [], 'hr': [], 'ndcg': [], 'cov': [], 'recall': [], 'max_recall': [], 'precision': [], 'auc': [],
               'n_pos': []}

    if isinstance(group, tuple):
        fetcher = _info_fetcher(fp, *group)
    else:
        fetcher = _info_fetcher_list(fp, group)

    for user, predicted_ranking, relevance, utility, _ in fetcher:
        hits, ndcgs, covs, recall, max_recall, precision, aucs, n_liked = \
            calculate_metrics(predicted_ranking, relevance, utility, _, neg_sampling=neg_sampling)

        metrics['user'].append(user)
        metrics['hr'].append(hits)
        metrics['ndcg'].append(ndcgs)
        metrics['cov'].append(covs)
        metrics['recall'].append(recall)
        metrics['max_recall'].append(max_recall)
        metrics['precision'].append(precision)
        metrics['auc'].append(aucs)
        metrics['n_pos'].append(n_liked)

    return list(metrics.values())


def parallel_iter_users(fp, n_users, neg_sampling=None, group=None):
    n_cpus = multiprocessing.cpu_count()-2
    bs = max(min(1024, n_users // n_cpus), 64)  # Have a maximum batch size of 1024 and minimum of 64.
    n_batches = n_users // bs if n_users % bs == 0 else (n_users // bs) + 1
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus) as e:
        for i in tqdm(range(n_batches), desc='Submitting process futures to queue'):
            futures.append(e.submit(iter_users, fp, (i*bs, (i+1)*bs) if group is None else group, neg_sampling))
            # yield iter_users(fp, (i*bs, (i+1)*bs) if group is None else group, neg_sampling)

        # logger.info('Starting processes. Expect first iteration to be slow.')
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass #yield f.result()  # Yield results as futures are complete

    for f in tqdm(futures, desc='Process data'):
        yield f.result()


def _pickle_load_users(file_path):
    users = []
    with open(file_path, 'rb') as f:
        with tqdm(None, desc='Loading users') as t:
            while True:
                try:
                    info = pickle.load(f)
                    users.append(info['user'])
                    t.update()
                except EOFError:
                    break

    return users


def iter_models(results_path: str, model_names, fold: Fold, seed: int, study, other_fold: Fold = None,
                other_parameter: Fold = None, other_model: str = None, f_conf: List[FeatureConfiguration] = None,
                setting='standard'):
    testing = fold.data_loader.testing()
    meta = fold.data_loader.meta()

    neg_sampling = None
    n_ratings = None
    items = meta.items
    if setting in ['cold_user_item', 'cold_item_limit', 'cold_no_item_ratings']:
        training = fold.data_loader.training()
        validation = fold.data_loader.validation()
        items = {i for u in training + validation for i, _ in u.ratings}
        items = set(meta.items) - items

    # Group users based on study
    if any([study.startswith(possibility) for possibility in ['standard', 'fixed_neg_sampling', 'idcf_neg_sampling']]):
        testing = [testing]
        if study == 'fixed_neg_sampling':
            neg_sampling = (study, {'num_samples': 100})
        elif study.startswith('idcf_neg_sampling'):
            try:
                s, n_samples = study.rsplit('_', 1)
                n_samples = int(n_samples)
            except ValueError:
                s = study
                n_samples = 5
            neg_sampling = (s, {'samples_per_rating': n_samples})
    elif study == 'user_sparsity':
        # Group by users if flag is true otherwise wrap testing set
        testing = _group_users(testing)
    elif study == 'user_popularity':
        testing = _group_users_popularity(testing)
    elif study == 'user_test_popularity':
        testing = _group_users_popularity(testing, use_testing=True)
    elif study == 'user_group':  # Same as KGAT
        testing, n_ratings = _group_users_group(testing)
    else:
        raise ValueError('Invalid study')

    testing = [[u.index for u in b] for b in testing]  # Reduce memory

    ext = '' if setting == 'standard' else f'_{setting}'
    for name in model_names:
        results_dir = join_paths(results_path, fold.experiment.name, name)

        if f_conf is not None and len(f_conf) > 0:
            req = True
        else:
            req = False
            f_conf = []

        model_name = get_model_name(name, f_conf, req, other_state=other_fold, other_parameter=other_parameter,
                                    other_model=other_model)
        fp = os.path.join(results_dir, f'{fold.name}_{model_name}_predictions{ext}.pickle')
        users = _pickle_load_users(fp)

        logger.debug(f'Calculating metrics for {name}')
        metrics = {'user': [], 'hr': [], 'ndcg': [], 'cov': [], 'recall': [], 'max_recall': [], 'precision': [],
                   'auc': [], 'group': [], 'n_pos': []}

        user_bin = {u: i for i, b in enumerate(testing) for u in b}

        for user, hits, ndcgs, covs, recall, max_recall, precision, aucs, n_pos in \
                parallel_iter_users(fp, len(users), neg_sampling):
            metrics['user'].extend(user)
            metrics['hr'].extend(hits)
            metrics['ndcg'].extend(ndcgs)
            metrics['cov'].extend(covs)
            metrics['recall'].extend(recall)
            metrics['max_recall'].extend(max_recall)
            metrics['precision'].extend(precision)
            metrics['auc'].extend(aucs)
            metrics['group'].extend([user_bin[u] for u in user])
            metrics['n_pos'].extend(n_pos)

            if n_ratings is not None:
                if 'n_ratings' not in metrics:
                    metrics['n_ratings'] = []
                metrics['n_ratings'].extend([n_ratings[u] for u in user])

        #  If only one group, remove
        if len(testing) <= 1:
            metrics.pop('group')
            # Calculate coverage for each k. Note, only reduced metric.
            if not study.startswith('idcf_neg_sampling'):
                metrics['cov'] = {k: coverage(set.union(*[c[k] for c in metrics['cov']]), items)
                                  for k in metrics['cov'][0].keys()}
            else:
                metrics.pop('cov')
        else:
            if not study.startswith('idcf_neg_sampling'):
                cov = [[] for _ in range(len(testing))]
                for c, g in zip(metrics['cov'], metrics['group']):
                    cov[g].append(c)

                metrics['cov'] = [{k: coverage(set.union(*[c[k] for c in group_coverage]), items)
                                for k in metrics['cov'][0].keys()} for group_coverage in cov]
            else:
                metrics.pop('cov')

        extension = '' if study == 'standard' else f'_{study}'
        with open(os.path.join(results_dir, f'{fold.name}_{model_name}{extension}{ext}_metrics.pickle'), 'wb') as f:
            pickle.dump(metrics, f)


def run():
    args = parser.parse_args()
    args = vars(args)

    if not args.pop('debug'):
        logger.remove()
        logger.add(sys.stderr, level='INFO')

    data_path, results_path, experiments, models, seed, folds, study, other_state, other_parameters, other_model, \
    feature_names, setting = args.values()

    experiment_configs = [e for e in configuration.experiments.experiments if e.name in experiments]
    datasets = set([e.dataset.name for e in experiment_configs])
    feature_configurations = None
    if feature_names:
        feature_configurations = [get_feature_configuration(feature_name) for feature_name in feature_names]

    other_state = get_experiment(other_state, data_path)
    other_parameters = get_experiment(other_parameters, data_path)

    for dataset in datasets:
        dataset = Dataset(os.path.join(data_path, dataset), experiments)
        for experiment in dataset.experiments():
            logger.info(f'Running experiment: {experiment}')
            iterator = zip(*[e.folds() for e in [experiment, other_state, other_parameters] if e is not None])
            for all_fold in iterator:
                all_fold = list(all_fold)
                fold = all_fold.pop(0)

                other_fold = all_fold.pop(0) if other_state is not None else None
                other_params = all_fold.pop(0) if other_parameters is not None else None

                if folds is None or fold.name in folds:
                    logger.info(f'Running fold: {fold}')
                    iter_models(results_path, models, fold, seed, study, other_fold, other_params,
                                other_model, feature_configurations, setting)
                else:
                    logger.warning(f'Skipping fold: {fold.name}')


if __name__ == '__main__':
    run()
