import argparse
from os.path import join

import pandas as pd
from loguru import logger

from shared.entity import Entity
from shared.relation import Relation
from shared.user import User
from shared.utility import load_entities, valid_dir, save_entities, save_relations, save_users
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=valid_dir, default='../mindreader',
                    help='path to store downloaded data')


def convert_to_entities(path):
    logger.info('Converting to Entity')
    df_e = pd.read_csv(join(path, 'entities.csv'))
    df_t = pd.read_csv(join(path, 'triples.csv'), index_col=0)

    entities = {}

    for i, (_, (uri, name, labels)) in enumerate(df_e.sort_values('uri').iterrows()):
        labels = labels.split('|')
        entities[i] = Entity(i, name, 'Movie' in labels, original_id=uri)

    # Reverse
    uri_idx = {e.original_id: i for i, e in entities.items()}

    # Relation index
    r_idx = {r: i for i, r in enumerate(sorted(pd.unique(df_t.relation)))}

    # Map entities to index
    df_t.head_uri = df_t.head_uri.map(lambda x: uri_idx.get(x, None))
    df_t.tail_uri = df_t.tail_uri.map(lambda x: uri_idx.get(x, None))

    relations = {}

    for relation, group in df_t.groupby('relation'):
        idx = r_idx[relation]

        relations[idx] = Relation(idx, relation, group[['head_uri', 'tail_uri']].to_records(index=False).tolist(),
                                  relation)

    return entities, relations


def convert_to_users(path, entities):
    logger.info('Converting to User')
    df_r = pd.read_csv(join(path, 'ratings.csv'), index_col=0)

    uri_idx = {entity.original_id: entity.index for entity in entities}

    df_r.uri = df_r.uri.map(lambda x: uri_idx.get(x, None))
    df_r = df_r[df_r.isItem]  # Remove non-item ratings for now.

    users = {}
    for i, (user, df_ratings) in enumerate(sorted(df_r.groupby('userId'))):
        users[i] = User(user, i, df_ratings[['uri', 'sentiment']].to_records(index=None).tolist())
        if 'timestamp' in df_ratings.columns:
            users[i].rating_time = df_ratings[['uri', 'timestamp']].to_records(index=None).tolist()

    return users


def run(path):
    try:
        raise FileNotFoundError
        entities = load_entities(path)
    except (FileNotFoundError, TypeError):
        entities, relations = convert_to_entities(path)

        save_entities(path, entities)
        save_relations(path, relations)

        entities = list(entities.values())

    users = convert_to_users(path, entities)
    save_users(path, users)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.path)
