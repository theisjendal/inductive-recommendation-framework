import argparse
import json
import os
import pickle
from os.path import join
from typing import T, List, Dict

import networkx as nx
import numpy as np

import requests
from bs4 import BeautifulSoup
from loguru import logger
from tqdm import tqdm

from configuration.datasets import datasets
from configuration.experiments import experiments
from configuration.features import feature_configurations
from shared.configuration_classes import ExperimentConfiguration
from shared.entity import Entity
from shared.relation import Relation
from shared.user import User


def valid_dir(path):
    """
    Ensure valid path is parsed as argument.
    :param path: path to directory
    :return: error or path
    """
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError('The specified path is not a directory')
    else:
        return path


def valid_file(path):
    """
    Ensure file exists
    :param path: path to file
    :return: error or path
    """
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError('The specified path is not a file')
    else:
        return path


def download_progress(url, out_path, fname):
    """
    Downloads a file with a progress bar for visual clues.
    :param url: url to file
    :param out_path: output directory for file
    :param fname: name of output file
    """
    r = requests.get(url, stream=True)
    total = int(r.headers.get('content-length', 0))
    fname = join(out_path, fname)

    if not os.path.isfile(fname):
        with open(fname, 'wb') as file, \
                tqdm(fname, total=total, unit='iB', unit_scale=True, unit_divisor=1024, desc=f'Downloading: {fname}')\
                        as bar:
            for d in r.iter_content(chunk_size=1024):
                size = file.write(d)
                bar.update(size)

    else:
        logger.warning('Skipping download, already exist.')


def save_entities(out_path, idx_entities: Dict[int, Entity], fname_extension=None):
    _save_pickle(out_path, idx_entities, 'entities', fname_extension)


def load_entities(in_path, fname_extension=None) -> List[Entity]:
    return _load_pickle(in_path, 'entities', fname_extension)


def save_relations(out_path, idx_relations: Dict[int, Relation], fname_extension=None):
    _save_pickle(out_path, idx_relations, 'relations', fname_extension)


def load_relations(in_path, fname_extension=None) -> List[Relation]:
    return _load_pickle(in_path, 'relations', fname_extension)


def save_users(out_path, idx_users: Dict[int, User], fname_extension=None):
    _save_pickle(out_path, idx_users, 'users', fname_extension)


def load_users(in_path, fname_extension=None) -> List[User]:
    return _load_pickle(in_path, 'users', fname_extension)


def _save_pickle(in_path, dictionary: Dict[int, T], name, extension=None):
    fname = f'{name}.pickle' if extension is None else f'{name}_{extension}.pickle'
    path = join(in_path, fname)

    return save_pickle(path, list(dictionary.values()))


def _load_pickle(in_path, name, extension=None) -> List[T]:
    fname = f'{name}.pickle' if extension is None else f'{name}_{extension}.pickle'
    path = join(in_path, fname)

    return load_pickle(path)


def save_pickle(path, j_object):
    with open(path, 'wb') as f:
        pickle.dump(j_object, f)


def load_pickle(path: str) -> T:
    with open(path, 'rb') as f:
        logger.debug('Loading pickle file, may be slow for large files. Be patient.')
        return pickle.load(f)


def save_numpy(path, arr_object):
    np.save(path, arr_object)


def save_json(path, j_object):
    with open(path, 'w') as f:
        json.dump(j_object, f)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def graph_from_relations(relations: List[Relation], directed: bool = False) -> nx.Graph:
    edges = []
    for relation in relations:
        edges.extend(relation.edges)

    g = nx.Graph() if directed else nx.DiGraph()
    g.add_edges_from(edges)

    return g


def get_dataset_configuration(dataset):
    return next(filter(lambda d: d.name == dataset, datasets))


def get_experiment_configuration(experiment) -> ExperimentConfiguration:
    return next(filter(lambda e: e.name == experiment, experiments))


def get_feature_configuration(feature):
    return next(filter(lambda f: f.name == feature, feature_configurations))


def join_paths(*paths):
    result = None

    for path in paths:
        if not result:
            result = path
        else:
            result = os.path.join(result, path)

    return result


def beautify(entities):
    for e in tqdm(entities):
        soup = BeautifulSoup(e.description, 'html.parser')
        t = soup.get_text().strip()
        e.description = t

    return entities


def is_debug_mode():
    return next(iter(logger._core.handlers.values())).levelno < 20


def get_ranks(users, model, meta, sg, rank_all=False, item_index=None):
    assert rank_all == bool(item_index)

    if not rank_all:
        ranking = users.get_ranking(sg.get_seed(), meta)

        # Get predictions
        predictions = model.predict(users.index, users.ratings, ranking.to_list(neg_samples=None))

        # Then produce a ranked list
        ranked_list = list(sorted(ranking.to_list(), key=lambda item: (predictions.get(item, 0), item), reverse=True))

        ranked_lists = [(ranking, ranked_list)]
    else:
        predictions = model.predict_all([(user.index, user.ratings) for user in users])

        assert predictions.shape == (len(users), len(meta.items)), f'Expected predictions to be of shape ' \
                                                                   f'{(len(users), len(meta.items))} got ' \
                                                                   f'{predictions.shape}.'

        rankings = [user.get_ranking(sg.get_seed(), meta) for user in users]

        ranked_lists = [(r, list(sorted(r.to_list(neg_samples=None), key=lambda item: (predictions[i][item_index[item]], item),
                                        reverse=True)))
                        for i, r in enumerate(rankings)]

    return ranked_lists