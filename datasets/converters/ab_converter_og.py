import argparse
import json

import os
from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from shared.entity import Entity
from shared.relation import Relation
from shared.user import User
from shared.utility import valid_dir, save_entities, save_relations, save_users, load_users, load_entities, \
    load_relations

parser = argparse.ArgumentParser()
parser.add_argument('--path', nargs=1, type=valid_dir, default='../amazon-book',
                    help='amazon dataset path')
parser.add_argument('--subsample', action='store_true', help='flag for subsampling to 1m')


def get_element(entity_org_id, entities, name: str, recommendable: bool, original_id: str = None,
                 description: list = None):
    if original_id is None:
        original_id = name

    if original_id in entity_org_id:
        return entity_org_id[original_id]
    else:
        idx = len(entity_org_id)
        entity_org_id[original_id] = idx
        entities[idx] = Entity(idx, name, recommendable, original_id, description)
        return idx


def create_entities_and_relations(path):
    entities = {}
    entity_org_id = {}
    relation_org_id = {'TYPE_OF': 0, 'HAS_CATEGORY': 1, 'SALES_TYPE': 2, 'IS_BRAND': 3}
    triples = set()
    fname = os.path.join(path, 'meta_Books.json')
    n_lines = int(os.popen(f'wc -l {fname}').read().split(' ')[0])
    with open(fname) as f:
        for line in tqdm(f, total=n_lines):
            inner_t = set()
            info = eval(line)
            item = get_element(entity_org_id, entities, info.get('title', ''), True, info.get('asin'),
                               [info.get('description', [])])
            if categories := info.get('category'):
                for i, category in enumerate(reversed(categories)):
                    if i == 0:
                        entity = get_element(entity_org_id, entities, category, False, None, None)
                        inner_t.add((item, relation_org_id['HAS_CATEGORY'], entity))
                    else:
                        entity = get_element(entity_org_id, entities, category, False, None, None)
                        # Note -i gets the previous element of the reversed list
                        entity_2 = get_element(entity_org_id, entities, categories[-i], False, None, None)
                        inner_t.add((entity_2, relation_org_id['TYPE_OF'], entity))

            if rank := info.get('salesRank'):
                l = list(rank.keys())
                assert len(l) == 1
                entity = get_element(entity_org_id, entities, l[0], False, None, None)
                inner_t.add((item, relation_org_id['SALES_TYPE'], entity))

            if brand := info.get('brand'):
                entity = get_element(entity_org_id, entities, brand, False, None, None)
                inner_t.add((item, relation_org_id['IS_BRAND'], entity))

            triples.update(inner_t)

    logger.info(f'Found {len(entities)} entities and {len(triples)} triples')
    df = pd.DataFrame.from_records(list(triples), columns=['h', 'r', 't'])
    df = df.sort_values(by=['r', 'h', 't'])
    relations = {}
    reverse_r = {v: k for k, v in relation_org_id.items()}
    for i, g in df.groupby(by='r'):
        edges = list(g[['h', 't']].to_records(index=False))
        relations[i] = Relation(i, reverse_r[i], edges)

    return entities, relations


def create_users(path, entities, subsample):
    logger.info('Creating users')
    users = {}

    fname = os.path.join(path, 'reviews_Books.json')
    n_lines = int(os.popen(f'wc -l {fname}').read().split(' ')[0])
    ratings = set()
    org_user_mapping = {}
    org_entity_mapping = {e.original_id: e.index for e in entities.values()}
    times = []
    if subsample:
        with open(fname) as f:
            for i, line in enumerate(tqdm(f, total=n_lines)):
                info = json.loads(line)
                times.append(int(info['unixReviewTime']))

        times = sorted(times)
    times = set(times)
    with open(fname) as f:
        for line in tqdm(f, total=n_lines):
            info = json.loads(line)
            org_id = info['reviewerID']
            if org_id in org_user_mapping:
                uid = org_user_mapping[org_id]
            else:
                uid = len(org_user_mapping)
                org_user_mapping[org_id] = uid
                users[uid] = User(org_id, uid)

            time = int(info['unixReviewTime'])
            if (entity := org_entity_mapping.get(info['asin'])) and \
                         (not subsample or time in times):
                if subsample:
                    ratings.add((uid, entity, int(info['overall']), int(info['unixReviewTime'])))
                else:
                    ratings.add((uid, entity, int(info['overall'])))

                if (text := info.get('reviewText')) is not None:
                    if entities[entity].description is not None:
                        entities[entity].description.append(text)
                    else:
                        entities[entity].description = [text]

    no_desc = len([0 for e in entities.values() if e.recommendable and e.description is None])
    if no_desc:
        logger.warning(f'No descriptions found for {no_desc} items')

    columns = ['u', 'i', 'r']
    if subsample:
        columns += ['t']

    df = pd.DataFrame.from_records(list(ratings), columns=columns)

    if subsample:
        df = df.sort_values(by='t')[:1000000]
        df = df[['u', 'i', 'r']]

    logger.info(f'Found {len(users)} user and {len(ratings)} ratings')
    df = df.sort_values(by=['u', 'i', 'r'])
    for i, g in tqdm(df.groupby(by='u'), desc='Assigning ratings to users'):
        r = list(g[['i', 'r']].to_records(index=False))
        users[i].ratings = r

    return users


def _reindex(dictionary):
    result = {}
    reindex = {}
    for _, value in dictionary.items():
        if isinstance(value, User):
            if not value.ratings:
                continue
        elif isinstance(value, Relation):
            if not value.edges:
                continue

        idx = len(result)
        reindex[value.index] = idx
        result[idx] = value
        value.index = idx

    return result, reindex


def _reindex_all(users, entities, relations):
    users, user_reindex = _reindex(users)
    entities, entity_reindex = _reindex(entities)
    relations, relation_reindex = _reindex(relations)

    for d in [users, relations]:
        for v in d.values():
            if isinstance(v, User):
                v.ratings = [(entity_reindex[i], r) for i, r in v.ratings]
            else:
                v.edges = [(entity_reindex[h], entity_reindex[t]) for h, t in v.edges]

    return users, entities, relations


def prune(users, entities, relations):
    liked = {item for user in users.values() for item, _ in user.ratings}
    before = set(entities.keys())
    entities = {idx: e for idx, e in entities.items() if not e.recommendable or idx in liked}
    after = set(entities.keys())

    i = 1
    while before != after:
        before = after
        n_relations = defaultdict(int)

        for idx, relation in tqdm(relations.items(), total=len(relations), desc=f'Iter {i}, pruning'):
            relation.edges = [(h, t) for h, t in relation.edges if h in entities]
            for h, t in relation.edges:
                n_relations[t] += 1

        entities = {idx: e for idx, e in entities.items() if e.recommendable or n_relations[idx] > 0}

        after = set(entities.keys())
        i += 1

    return _reindex_all(users, entities, relations)


def run(path, subsample):
    if subsample:
        out_path = path + '-1m'
        if not os.path.isdir(path):
            raise NotADirectoryError(f'Must create directory at path: {path}')
    else:
        out_path = path

    data_path = os.path.join(path, 'data')
    entities, relations = create_entities_and_relations(data_path)

    # entities = load_entities(out_path)
    # relations = load_relations(out_path)
    # entities = {t.index: t for t in entities}
    # relations = {t.index: t for t in relations}

    users = create_users(data_path, entities, subsample)

    # users = load_users(out_path)
    # users = {t.index: t for t in users}

    if subsample:
        users, entities, relations = prune(users, entities, relations)

    logger.info('Saving data')
    save_entities(out_path, entities)
    save_relations(out_path, relations)
    save_users(out_path, users)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.path, args.subsample)
