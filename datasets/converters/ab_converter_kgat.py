import argparse
import ast
import json
import os

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice
from typing import Dict

import pandas as pd
from loguru import logger
from tqdm import tqdm

from shared.entity import Entity
from shared.relation import Relation
from shared.user import User
from shared.utility import valid_dir, save_entities, save_relations, save_users, load_entities
from datasets.converters.ab_converter_og import prune

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=valid_dir, default='../amazon-book',
                    help='amazon dataset path')


def load_chunk(file_path, start, end, post_func=None):
    data = [l if post_func is None else post_func(l) for i, l in enumerate(islice(open(file_path), start, end))]
    return data


def parse(path, desc, chunk_size=100000):
    try:
        n_lines = int(os.popen(f'wc -l {path}').read().split(' ')[0])
    except OSError:
        raise OSError('Must run ab-(downloader|converter)-(og).py before this file.')
    with ProcessPoolExecutor() as executor:
        futures = []
        start = 0
        for i in range(chunk_size, n_lines, chunk_size):
            futures.append(executor.submit(load_chunk, path, start, i, ast.literal_eval))
            start = i

        for f in tqdm(as_completed(futures), desc=desc, total=len(futures), smoothing=0):
            pass

    for f in tqdm(futures, 'Yielding results', total=len(futures), smoothing=0):
        for d in f.result():
            yield d


def getDF(path, items):
    descriptions = {}
    rating_times = defaultdict(list)
    item_reviews = defaultdict(list)
    for d in parse(path, desc=f'Loading {path}', chunk_size=10000):
        if item := d.get('asin'):
            descriptions[item] = d
            if t := d.get('unixReviewTime'):
                u = d.get('reviewerID')
                rating_times[u].append((item, t))

    for item in tqdm(items):
        if desc := descriptions.get(item):
            if text := desc.get('title'):
                item_reviews[item].append(text)
            if text := desc.get('description'):
                item_reviews[item].append(text)
    return item_reviews


def create_entities(path):
    logger.info('Creating entities')
    d = {'freebaseId': [], 'remap_id': []}
    with open(os.path.join(path, 'entity_list.txt')) as f:
        f.readline() # Skip headers
        for line in f:
            freebase_id, index = line.rsplit(' ', maxsplit=1)
            d['freebaseId'].append(freebase_id)
            d['remap_id'].append(index)

    items = []
    with open(os.path.join(path, 'item_list.txt')) as f:
        f.readline() # Skip headers
        for line in f.read().splitlines():
            items.append(line.split(' ', maxsplit=2))

    df = pd.DataFrame.from_dict(d).astype({'freebaseId': str, 'remap_id': int})
    df_item = pd.DataFrame.from_records(items, columns=['orgId', 'remap_id', 'freebaseId'])\
        .astype({'freebaseId': str, 'remap_id': int, 'orgId': str})

    df = pd.merge(df, df_item, on='freebaseId', how='left')
    df = df.sort_values(by=['orgId', 'freebaseId'])
    descriptions = getDF(os.path.join(path, 'meta_Books.json'), df[['orgId']].dropna().drop_duplicates()['orgId'].tolist())

    entities = {}

    no_desc = 0
    for index, (_, (freebaseId, remap_id, org_id, _)) in tqdm(enumerate(df.iterrows()), total=len(df)):
        recommendable = not pd.isna(org_id)
        if not recommendable or (description := descriptions.get(org_id)) is not None:
            entities[index] = Entity(index, name=freebaseId, original_id=remap_id,
                                 recommendable=recommendable, description=description)
        else:
            no_desc += 1

    if no_desc:
        logger.warning(f'There are {no_desc} items without any descriptions')

    entities =  {i: e for i, e in enumerate(entities.values())}
    for i, e in entities.items():
        e.index = i

    return entities


def create_relations(path, remap_index):
    logger.info('Creating relations')
    df_kg = pd.read_csv(os.path.join(path, 'kg_final.txt'), header=None, names=['head', 'relation', 'tail'],
                            sep=' ')
    df_kg['head'] = df_kg['head'].apply(remap_index.get)
    df_kg['tail'] = df_kg['tail'].apply(remap_index.get)
    before = len(df_kg)
    df_kg = df_kg.dropna()
    diff = before - len(df_kg)

    if diff:
        logger.warning(f'Removed {diff} edges as entities have been removed.')

    df_relation = pd.read_csv(os.path.join(path, 'relation_list.txt'), sep=' ')
    remap_org = df_relation.set_index('remap_id')['org_id'].to_dict()

    relations = {}
    for group, df in df_kg.groupby('relation'):
        uri = remap_org[group]
        edges = df[['head', 'tail']].to_records(index=False).tolist()
        relations[group] = Relation(group, uri, edges, uri)

    return relations


def create_users(path, remap_id):
    logger.info('Creating users')
    df = pd.read_csv(os.path.join(path, 'user_list.txt'), sep=' ')
    user_mapping = df.set_index('remap_id')['org_id'].to_dict()
    reverse_user_mapping = {v: k for k, v in user_mapping.items()}

    item_df = pd.read_csv(os.path.join(path, 'item_list.txt'), sep=' ')
    item_mapping = item_df.set_index('remap_id')['org_id'].to_dict()
    reverse_item_mapping = {v: k for k, v in item_mapping.items()}

    fp = os.path.join(path, 'reviews_Books.json')
    n_lines = int(os.popen(f'wc -l {fp}').read().split(' ')[0])
    # data = [eval(line) for line in tqdm(open(fp), total=n_lines)]
    logger.info('Create df')
    data_df = pd.DataFrame(list(parse(fp, desc='Loading reviews')))

    logger.info('Map item ids')
    data_df['idx'] = data_df['reviewerID'].apply(reverse_user_mapping.get)

    logger.info('Map user ids')
    data_df['asin'] = data_df['asin'].apply(reverse_item_mapping.get)

    logger.info('Drop nans')
    data_df = data_df.dropna()

    users = {}
    for org_id, user_df in tqdm(data_df.groupby('reviewerID'), desc='Creating users', total=len(user_mapping)):
        idx = reverse_user_mapping[org_id]
        ratings = user_df[['asin', 'overall']].to_records(index=False).tolist()
        rating_times = user_df[['asin', 'unixReviewTime']].to_records(index=False).tolist()
        assert idx not in users
        users[idx] = User(org_id, idx, ratings, rating_times)

    return users


def run(path):
    data_path = os.path.join(path, 'data')
    entities = create_entities(data_path)
    remap_index = {e.original_id: e.index for e in entities.values()}
    relations = create_relations(data_path, remap_index)
    users = create_users(data_path, remap_index)

    users, entities, relations = prune(users, entities, relations)

    save_entities(path, entities)
    save_relations(path, relations)
    save_users(path, users)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.path)
