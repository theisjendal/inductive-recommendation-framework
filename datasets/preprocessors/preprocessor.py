import argparse
import random
from collections import defaultdict
from os.path import join
from typing import List, Dict, Tuple, Set

import networkx as nx
from loguru import logger
from tqdm import tqdm

from configuration.datasets import dataset_names, datasets
from shared.configuration_classes import DatasetConfiguration
from shared.entity import Entity
from shared.enums import Sentiment
from shared.relation import Relation
from shared.user import User
from shared.utility import valid_dir, save_entities, load_entities, save_relations, load_relations, save_users, \
    load_users

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=valid_dir, help='Location of datasets', default='..')
parser.add_argument('--dataset', choices=dataset_names, help='Datasets to process')


def prune_users(users: List[User], dataset: DatasetConfiguration, k=10) -> Tuple[List[User], bool]:
    pre_c = len(users)
    rating_times = {u.index: [t for _, t in u.rating_time] for u in users}
    users = filter(lambda u: len(u.ratings) >= k, tqdm(users, desc='Pruning users'))

    if dataset.time_based_pruning:
        # Filter users with ratings spanning less than k days
        # users = filter(lambda u: len(set(t for _, t in u.rating_time)) >= k, users)
        users = filter(lambda u: ((max(rating_times[u.index]) - min(rating_times[u.index])) / (60 * 60 * 24)) >= k, users)

    # Apply all filters
    for d_filter in dataset.filters:
        rating_type = dataset.sentiment_utility[d_filter.sentiment]
        users = filter(lambda u: d_filter.filter_func(len([i for i, r in u.ratings if r == rating_type])), users)

    users = sorted(users, key=lambda u: u.index)  # Sort by index
    post_c = len(users)

    # Create new indices if users have been removed
    change = bool(pre_c - post_c)
    if change:
        # Reassign index
        for i, u in tqdm(enumerate(users), total=len(users), desc='Reassigning user indices'):
            u.index = i

    return users, change


def prune_items(users: List[User], dataset: DatasetConfiguration, k=10) -> Tuple[List[User], bool]:
    item_count = defaultdict(list)

    for user in tqdm(users, desc='Finding rating count of items'):
        for item, r in user.ratings:
            item_count[item].append(r)

    items = filter(lambda i: len(i[1]) >= k, item_count.items())

    # {i for i, c in item_count.items() if c < k}
    # Apply filters
    for d_filter in dataset.filters:
        rating_type = dataset.sentiment_utility[d_filter.sentiment]
        items = filter(lambda i:  d_filter.filter_func(len([r for r in i[1] if r == rating_type])), items)

    prune_set = set(item_count.keys()).difference(dict(items).keys())
    to_prune = bool(prune_set)
    if to_prune:
        for user in tqdm(users, desc='Removing items'):
            user.ratings = [(i, r) for i, r in user.ratings if i not in prune_set]
            if hasattr(user, 'rating_time'):
                user.rating_time = [(i, t) for i, t in user.rating_time if i not in prune_set]

    return users, to_prune


def subsample_users(users: List[User], dataset: DatasetConfiguration):
    logger.info(f'Sampling {dataset.max_users} users.')
    if len(users) <= dataset.max_users:
        change_occured = False
    else:
        if dataset.time_based_sampling:
            # Sort by last user rating
            permutation = sorted(users, key=lambda u: min(t for _, t in u.rating_time), reverse=False)
            permutation = [u.index for u in permutation]
        else:
            permutation = list(range(len(users)))
            random.shuffle(permutation)

        permutation = sorted(permutation[:dataset.max_users])
        users = [users[p] for p in permutation]
        change_occured = True

    return users, change_occured


def create_k_core(users: List[User], dataset: DatasetConfiguration, k=10) -> Dict[int, User]:
    change_occured = True
    iter_count = 0
    logger.info('Going to iteratively prune users or items with too few ratings until convergence (or max 50 iter.)')
    while change_occured:
        logger.info(f'Pruning iter: {iter_count}')
        users, u_change = prune_users(users, dataset, k)
        users, i_change = prune_items(users, dataset, k)

        change_occured = u_change or i_change

        # If no items nor users were removed stop loop with a max of 50
        if iter_count >= 50:
            change_occured = False
        else:
            iter_count += 1

        if not change_occured and dataset.max_users is not None:
            users, change_occured = subsample_users(users, dataset)

    users = {user.index: user for user in users}

    for i, user in users.items():
        user.index = i

    return users


def map_ratings(users: List[User], dataset: DatasetConfiguration) -> List[User]:
    unseen = dataset.sentiment_utility[Sentiment.UNSEEN]
    for user in tqdm(users, desc='Mapping user ratings'):
        # Skip non polarized ratings and add rest to ratings
        mapped = [(i, dataset.ratings_mapping(r)) for i, r in user.ratings]
        indices = [i for i, (_, r) in enumerate(mapped) if r != unseen]
        user.ratings = [mapped[i] for i in indices]
        if hasattr(user, 'rating_time'):
            user.rating_time = [user.rating_time[i] for i in indices]

    if dataset.max_ratings is not None:
        logger.info(f'Sampling {dataset.max_ratings} ratings.')
        rating_matrix = [[u.index, i, t] for u in users for i, t in u.rating_time]
        if dataset.time_based_sampling:
            # Sort by last user rating
            permutation = sorted(range(len(rating_matrix)), key=lambda x: rating_matrix[x][2], reverse=True)
        else:
            permutation = list(range(len(rating_matrix)))
            random.shuffle(permutation)

        permutation = sorted(permutation[:dataset.max_ratings])
        rating_matrix = [rating_matrix[p] for p in permutation]

        # Group by user
        rating_matrix = sorted(rating_matrix, key=lambda x: x[0])
        ratings = defaultdict(list)
        for u, i, t in rating_matrix:
            ratings[u].append(i)

        # Update users
        for u in users:
            if u.index in ratings:
                u.ratings = [(i, r) for i, r in u.ratings if i in ratings[u.index]]
                if hasattr(u, 'rating_time'):
                    u.rating_time = [(i, t) for i, t in u.rating_time if i in ratings[u.index]]
            else:
                u.ratings = []
                if hasattr(u, 'rating_time'):
                   u.rating_time = []

    if dataset.prune_duplicates:
        logger.info('Pruning duplicate ratings')
        for user in tqdm(users, desc='Pruning duplicate ratings'):
            items = {i for i, _ in user.ratings}

            # Last encounter will be added to dictionary.
            if hasattr(user, 'rating_time'):
                times = {it: i for i, (it, t) in sorted(enumerate(user.rating_time), key=lambda x: x[::-1])}
            else:
                times = {it: i for i, it in enumerate(items)}

            # Sort by index
            times = [i for i in sorted(times.values())]

            # Limit to indexing
            user.ratings = [user.ratings[t] for t in times]

            if hasattr(user, 'rating_time'):
                user.rating_time = [user.rating_time[t] for t in times]

    return users


def get_rated_items(users: List[User]) -> Set[int]:
    items = set()

    for user in users:
        items.update([r[0] for r in user.ratings])

    return items


def remove_duplicate_edges(relations: List[Relation]) -> List[Relation]:
    for relation in relations:
        relation.edges = list(set(relation.edges))

    return relations


def num_components(relations: List[Relation]) -> int:
    logger.info('Getting number of connected components')
    edges = []

    for relation in relations:
        edges.extend(relation.edges)

    g = nx.Graph()

    g.add_edges_from(edges)

    return nx.number_connected_components(g)


def prune_relations(entities: Dict[int, Entity], relations: List[Relation]) -> List[Relation]:
    # remove invalid edges
    for relation in relations:
        relation.edges = [(src, dst) for src, dst in relation.edges if src in entities and dst in entities]

    return [r for r in relations if r.edges]  # Remove relations with no edges


def get_n_hop_connections(entities: Dict[int, Entity], relations: List[Relation], n_hops: int)\
        -> Set[int]:
    # The start set is the recommendable entities.
    if n_hops <= 0:
        return {e.index for e in entities.values() if e.recommendable}

    connections = get_n_hop_connections(entities, relations, n_hops-1)

    for relation in relations:
        for src, dst in relation.edges:
            if src in connections:
                connections.add(dst)
            elif dst in connections:
                connections.add(src)

    return connections


def remove_unrated(entities: List[Entity], relations: List[Relation], rated_entities: Set[int]) \
        -> Tuple[List[Entity], List[Relation]]:
    start_len = len(entities)
    # remove item entities without ratings
    entities = filter(lambda e: not e.recommendable or e.index in rated_entities, entities)
    entities = {e.index: e for e in entities}

    relations = prune_relations(entities, relations)

    # Find number of times pointing to or from a recommendable entity.
    connections = get_n_hop_connections(entities, relations, n_hops=2)

    # Ensure entity is connected to at least one rated entity or is recommendable.
    entities = {idx: e for idx, e in entities.items() if e.recommendable or idx in connections}

    relations = prune_relations(entities, relations)

    logger.info(f'Removed {start_len - len(entities)} items from graph due to lack of ratings')

    return list(entities.values()), relations


def prune_entities(dataset, entities: List[Entity], relations: List[Relation], rated_entities: Set[int], min_degree: int = 2) \
        -> Tuple[Dict[int, Entity], Dict[int, Relation]]:

    # entities, relations = remove_unrated(entities, relations, rated_entities)
    entities = {e.index: e for e in entities}

    changed = True
    iter_count = 0
    while changed:
        degree = defaultdict(int)
        logger.info(f'Iteratively pruning entities, iter: {iter_count}, n_entities: {len(entities)}')
        # Get degree
        for relation in relations:
            for src, dst in relation.edges:
                degree[src] += 1
                degree[dst] += 1

        l = len(entities)

        # Store if rated and recommendable, however if not rated but recommendable, discard or if too few edges.
        entities = {i: e for i, e in entities.items() if e.index in rated_entities or
                    (degree.get(i, 0) >= min_degree)}

        if dataset.remove_unseen:
            entities = {i: e for i, e in entities.items() if not e.recommendable or e.index in rated_entities}

        to_prune = bool(l-len(entities))

        if to_prune:
            relations = prune_relations(entities, relations)

            iter_count += 1
        else:
            changed = False

    nc = num_components(relations)
    logger.info(f'Num connected = {nc}')
    assert nc == 1, 'Some models require single connected component. Implement code adding' \
                                           'entities back into the graph.'

    return entities, {r.index: r for r in relations}


def reindex(entities: Dict[int, Entity], relations: Dict[int, Relation], users: Dict[int, User], org_u_map) \
        -> Tuple[Dict[int, Entity], Dict[int, Relation], Dict[int, User]]:
    logger.info('Reindexing entities, relations, and users')

    # Create mapping, sorting by having recommendable entities first then by index.
    rated = {item for user in users.values() for item, _ in user.ratings}
    mapping = {e: i for i, e in enumerate(sorted(entities.keys(), key=lambda e: (e not in rated, e)))}

    # Reindex entities
    for i, e in entities.items():
        e.recommendable = e.index in rated
        e.index = mapping[i]

    entities = {e.index: e for e in sorted(entities.values(), key=lambda x: x.index)}

    # Reindex relations
    for relation in relations.values():
        relation.edges = [(mapping[src], mapping[dst]) for src, dst in relation.edges]

    # Reindex users
    user_mapping = {}
    for i, user in enumerate(users.values()):
        user_mapping[user.index] = i
        user.index = i
        user.ratings = [(mapping[item], rating) for item, rating in user.ratings]
        if hasattr(user, 'rating_time'):
            item_set = [item for item, _ in user.ratings]
            user.rating_time = [(mapping[item], time) for item, time in user.rating_time
                                if item in mapping and mapping[item] in item_set]
    users = {org_u_map[u.original_id]: u for u in sorted(users.values(), key=lambda x: x.index)}

    for entity in entities.values():
        if entity.description is not None and isinstance(entity.description, list):
            if isinstance(entity.description[0], tuple):
                entity.description = [(user_mapping[i], d) for i, d in entity.description if i in user_mapping]

    return entities, relations, users


def run(path, dataset):
    dataset = next(filter(lambda d: d.name == dataset, datasets))
    random.seed(dataset.seed)
    in_path = out_path = join(path, dataset.name)
    users = load_users(in_path)
    org_u_map = {u.original_id: u.index for u in users}
    users = map_ratings(users, dataset)
    users = create_k_core(users, dataset, k=dataset.k_core)

    items = get_rated_items(list(users.values()))

    relations = load_relations(in_path)
    relations = remove_duplicate_edges(relations)

    entities = load_entities(in_path)
    entities, relations = prune_entities(dataset, entities, relations, items)
    entities, relations, users = reindex(entities, relations, users, org_u_map)

    logger.info('Saving the entities, relations and users')
    save_users(out_path, users, fname_extension='processed')
    save_entities(out_path, entities, fname_extension='processed')
    save_relations(out_path, relations, fname_extension='processed')


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.path, args.dataset)
