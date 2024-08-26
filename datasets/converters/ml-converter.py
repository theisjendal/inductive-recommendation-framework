import argparse
from os.path import join, isfile
from typing import Dict, List

import pandas as pd
from loguru import logger
from tqdm import tqdm

from shared.entity import Entity
from shared.relation import Relation
from shared.user import User
from shared.utility import valid_dir, load_json, save_pickle, save_entities, save_relations, load_entities, save_users

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=valid_dir, default='../movielens',
                    help='path to store downloaded data')
parser.add_argument('--out_path', type=valid_dir, default='../movielens',
                    help='path to store downloaded data')


def convert_to_entities(in_path):
    logger.info('Loading data, be patient.')
    triples = pd.read_csv(join(in_path, 'triples.csv'))
    entities_df = pd.read_csv(join(in_path, 'entities.csv'))
    predicates_df = pd.read_csv(join(in_path, 'predicates.csv'))
    movie_uri = load_json(join(in_path, 'movie_uris.json'))
    movies = {uri for _, uri in movie_uri.items()}

    entities = {}
    entities_df = entities_df.sort_values('uri')
    for i, (_, (uri, label, description, _)) in tqdm(enumerate(entities_df.iterrows()), total=len(entities_df),
                               desc='Creating entities'):
        recommendable = uri in movies
        entities[i] = Entity(i, label, recommendable, uri, description)

    uris = [e.original_id for e in entities.values()]

    # Remove rows with entities without type
    triples = triples[triples.isin(uris).sum(axis=1) == 2]

    reverse_idx = {e.original_id: i for i, e in entities.items()}

    relations = {}
    relation_id = 0
    for relation_uri in tqdm(sorted(triples.p.unique()), desc='Creating relations'):
        _, label, description = predicates_df[predicates_df.uri == relation_uri].iloc[0]
        relation = Relation(relation_id, label, [], original_id=relation_uri)
        for _, (s, _, o) in triples[triples.p == relation_uri].iterrows():
            relation.edges.append((reverse_idx[s], reverse_idx[o]))

        relations[relation_id] = relation
        relation_id += 1

    return entities, relations


def convert_to_users(in_path, entities: List[Entity]):
    movie_uris = load_json(join(in_path, 'movie_uris.json'))
    reverse_idx = {e.original_id: e.index for e in entities}
    movie_idx = {m: reverse_idx[u] for m, u in movie_uris.items() if u in reverse_idx}  # Map imdb to idx

    ratings = pd.read_csv(join(in_path, 'ml-20m', 'ratings.csv'), dtype=str)

    users = ratings.userId.unique().astype(int).tolist()

    assert len(users) == max(users), 'Write code to convert user indices to incremental'

    users = {}
    for user, rating_tuples in tqdm(ratings.groupby('userId'), desc='Creating users'):
        u = User(str(user), int(user), [])

        # Skip ratings without uri
        u.ratings = [(movie_idx[m_id], float(rating)) for _, (_, m_id, rating, _) in rating_tuples.iterrows() if
                     m_id in movie_idx]

        u.rating_time = rating_tuples[['movieId', 'timestamp']].to_records(index=False).tolist()

        users[user] = u

    return users


def run(in_path, out_path):
    entities, relations = convert_to_entities(in_path)

    save_entities(out_path, entities)
    save_relations(out_path, relations)

    entities = list(entities.values())

    entities = load_entities(out_path)

    users = convert_to_users(in_path, entities)
    save_users(out_path, users)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.in_path, args.out_path)