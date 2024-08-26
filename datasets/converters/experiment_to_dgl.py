import argparse
import concurrent
import os.path
import pickle
from concurrent.futures import ProcessPoolExecutor
from os.path import join

import dgl.data.utils
import random

import numpy as np
from sklearn.metrics import dcg_score
from sklearn.metrics._ranking import _dcg_sample_scores
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from configuration.experiments import experiment_names, experiments
from shared.enums import ExperimentEnum
from shared.graph_utility import *
from shared.user import LeaveOneOutUser
from shared.utility import valid_dir, save_pickle

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=valid_dir, help='Location of datasets', default='..')
parser.add_argument('--experiment', choices=experiment_names, help='Datasets to process')
parser.add_argument('--min_samples', type=int, default=800, help='Minimum number of unseen items per user')
parser.add_argument('--graphs_only', action='store_true')
parser.add_argument('--folds', default=None)


def _join_dataset_parts(dataset_a, dataset_b):
    """
    Merge two datasets into one. Assumes that both datasets have the same users. With users in b taking precedence.
    :param dataset_a: Warm-start dataset
    :param dataset_b: Cold-start dataset
    :return: merged dataset
    """
    u_map_a = {u.index: i for i, u in enumerate(dataset_a)}
    u_map_b = {u.index: i for i, u in enumerate(dataset_b)}
    all_users = set(u_map_a).union(u_map_b)

    out = []
    for u in sorted(all_users):
        if u in u_map_b:
            out.append(dataset_b[u_map_b[u]])
        elif u in u_map_a:
            out.append(dataset_a[u_map_a[u]])

    return out

def iter_graphs(train, validation, test, meta) -> Tuple[str, dgl.DGLHeteroGraph]:
    n_entities = len(meta.entities)
    user_fn = lambda u: u + n_entities
    r_label_fn = lambda rs: {r.name: i for i, r in enumerate(rs)}
    num_nodes = len(meta.users) + len(meta.entities)
    relations = meta.relations

    # Should contain all ratings from previous graphs i.e. test superset of validation superset of train
    validation = _join_dataset_parts(train, validation)
    test = _join_dataset_parts(validation, test)

    # Creating homogeneous graphs with relation types
    for name, data in [('train', train), ('validation', validation), ('test', test)]:
        if hasattr(data[0], 'rating_time') and data[0].rating_time is not None:
            rating_times = {(user_fn(u.index), i): t for u in data for i, t in u.rating_time}
        else:
            rating_times = None

        recommendable_items = [i for user in data for i, _ in (user.loo + user.ratings
                               if isinstance(user, LeaveOneOutUser) else user.ratings)]

        positive_relations = create_rating_relations(meta, data, user_fn, [Sentiment.POSITIVE])
        pos_rev_rel = positive_relations + create_reverse_relations(positive_relations)
        all_relations = create_rating_relations(meta, data, user_fn, [Sentiment.POSITIVE, Sentiment.NEGATIVE])

        yield f'{name}_cg_pos.dgl', dgl_homogeneous(meta, recommendable_items, positive_relations, store_edge_types=True,
                                                    num_nodes=num_nodes, rating_times=rating_times), r_label_fn(positive_relations)
        yield f'{name}_cg_pos_rev.dgl', dgl_homogeneous(meta, recommendable_items, pos_rev_rel, store_edge_types=True, num_nodes=num_nodes,
                                            rating_times=rating_times), r_label_fn(pos_rev_rel)
        yield f'{name}_kg.dgl', dgl_homogeneous(meta, recommendable_items, relations, store_edge_types=True, num_nodes=num_nodes,
                                            rating_times=rating_times), r_label_fn(relations)

        # Combine relations and rating relations to create collaborative knowledge graphs.
        positive_relations = relations + positive_relations
        all_relations = relations + all_relations
        yield f'{name}_ckg.dgl', dgl_homogeneous(meta, recommendable_items, all_relations, store_edge_types=True, num_nodes=num_nodes,
                                            rating_times=rating_times), r_label_fn(all_relations)
        yield f'{name}_ckg_pos.dgl', dgl_homogeneous(meta, recommendable_items, positive_relations, store_edge_types=True, num_nodes=num_nodes,
                                            rating_times=rating_times), r_label_fn(all_relations)

        all_relations += create_reverse_relations(all_relations)
        positive_relations += create_reverse_relations(positive_relations)
        yield f'{name}_ckg_rev.dgl', dgl_homogeneous(meta, recommendable_items, all_relations, store_edge_types=True, num_nodes=num_nodes,
                                            rating_times=rating_times), r_label_fn(all_relations)
        yield f'{name}_ckg_pos_rev.dgl', dgl_homogeneous(meta, recommendable_items, positive_relations, store_edge_types=True, num_nodes=num_nodes,
                                            rating_times=rating_times), \
              r_label_fn(positive_relations)


def find_seen_length(validation, skip_value):
    max_seen_length = -1
    for user in validation:
        seen_length = len(user.loo) + len(user.ratings)

        if seen_length > skip_value:
            continue
        elif seen_length > max_seen_length:
            max_seen_length = seen_length

    return max_seen_length


def sample_neg(validation, item_idx, length, meta, skip_value, k):
    item_matrix = []
    label_matrix = []
    users = []
    for user in validation:
        ratings = len(user.ratings)
        loo = len(user.loo)

        if ratings + loo > skip_value:
            continue

        ranking = user.get_ranking(42, meta)
        items = ranking.to_list(neg_samples=length - loo)
        item_matrix.append(items)
        labels = ranking.get_utility(items, meta.sentiment_utility)
        label_matrix.append(labels)
        users.append(user.index)

    item_matrix = np.array(item_matrix)
    label_matrix = np.array(label_matrix)

    # Map item identifier to index matching order of meta.items.
    item_matrix = np.vectorize(item_idx.get)(item_matrix)

    # Partition labels along last axis such that highest k labels are in end
    indices = np.argpartition(label_matrix, kth=-k, axis=-1)[:, -k:]

    # Calculate idcg based on this partitioning.
    true_relevance = np.take_along_axis(label_matrix, indices, axis=-1)
    idcg = _dcg_sample_scores(true_relevance, true_relevance, ignore_ties=True)

    return item_matrix, label_matrix, users, idcg


def sample_iter(validation, item_idx, sample_length, meta, skip_value, k):
    n_cpu = max(1, os.cpu_count() - 1)
    futures = []
    batch_size = 256
    batches = [validation[i*batch_size:(i+1)*batch_size] for i in range(len(validation)//batch_size + 1)]
    # batches = batches[32:35]
    with ProcessPoolExecutor(max_workers=n_cpu) as executor:
        for batch in batches:
            # Uncomment for debug
            # sample_neg(batch, item_idx, sample_length, meta, skip_value, k)

            if batch:  # may be empty
                futures.append(executor.submit(sample_neg, batch, item_idx, sample_length, meta, skip_value, k))

        for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass  # yield f.result()

    for f in tqdm(futures, desc='Processing'):
        yield f.result()


def optimized_metric_calculation(validation, sample_length, meta, skip_value, item_idx, k):
    idcgs = []
    full_im = []
    full_lm = []
    full_ul = []
    for item_matrix, label_matrix, users, idcg in sample_iter(validation, item_idx, sample_length, meta, skip_value, k):
        full_im.append(item_matrix)
        full_lm.append(label_matrix)
        full_ul.append(users)
        idcgs.append(idcg)

    idcgs = np.concatenate(idcgs)
    full_ul = np.concatenate(full_ul)
    full_im = np.concatenate(full_im)
    full_lm = np.concatenate(full_lm)
    sorting = np.argsort(full_ul)

    idcgs = idcgs[sorting]
    full_ul = full_ul[sorting]
    full_im = full_im[sorting]
    full_lm = full_lm[sorting]

    return idcgs, full_im, full_lm, full_ul


def run(path, experiment, min_samples, graphs_only, folds):
    experiment = next(filter(lambda e: e.name == experiment, experiments))
    k = 20
    dataset_path = join(path, experiment.dataset.name)
    experiment_path = join(dataset_path, experiment.name)

    meta = pickle.load(open(join(experiment_path, 'meta.pickle'), 'rb'))

    # Ensure same seed
    random.seed(experiment.seed)
    np.random.seed(experiment.seed)
    dgl.seed(experiment.seed)

    # Create train graphs
    folds = [f'fold_{i}' for i in range(experiment.folds)]
    for fold in tqdm(folds, desc='Creating graphs for fold'):
        fold_path = join(experiment_path, fold)
        train = pickle.load(open(join(fold_path, 'train.pickle'), 'rb'))
        validation = pickle.load(open(join(fold_path, 'validation.pickle'), 'rb'))
        test = pickle.load(open(join(fold_path, 'test.pickle'), 'rb'))
        for name, graph, rmap in iter_graphs(train, validation, test, meta):
            dgl.save_graphs(join(fold_path, name), [graph])
            dgl.data.utils.save_info(join(fold_path, name + '.info'), rmap)

    # Skip users with too many ratings such that validation results would be skewed.
    skip_value = len(meta.items) - min_samples

    if graphs_only:
        return

    # Create validation partition
    for fold in folds:
        print(f'Precalculating idcg and finding labels for {fold}')
        fold_path = join(experiment_path, fold)
        validation = pickle.load(open(join(fold_path, 'validation.pickle'), 'rb'))
        item_idx = {item: idx for idx, item in enumerate(meta.items)}

        s_length = find_seen_length(validation, skip_value)
        sample_length = len(meta.items) - s_length

        idcg, item_matrix, label_matrix, users = optimized_metric_calculation(validation, sample_length, meta, skip_value, item_idx, k)

        np.save(join(fold_path, 'labels.npy'), label_matrix)
        np.save(join(fold_path, 'items.npy'), item_matrix)

        # Save k, idcg and validation users.
        info = {'k': k, 'idcg': idcg, 'users': users}

        pickle.dump(info, open(join(fold_path, 'info.pickle'), 'wb'))


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.path, args.experiment, args.min_samples, args.graphs_only, args.folds)
