import argparse
import json
import os
import pickle
import shutil
import sys
import time
from copy import deepcopy
from typing import List

import dgl
import torch
from loguru import logger
import numpy as np

import configuration.experiments
from configuration.features import feature_conf_names
from configuration.models import dgl_models
from models.dgl_recommender_base import RecommenderBase
from shared.configuration_classes import FeatureConfiguration
from shared.efficient_validator import Validator
from shared.experiments import Dataset, Fold, Experiment
from shared.meta import Meta
from shared.seed_generator import SeedGenerator
from shared.utility import valid_dir, get_experiment_configuration, get_feature_configuration

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=valid_dir, help='path to datasets')
parser.add_argument('--out_path', type=valid_dir, help='path to store results')
parser.add_argument('--experiments', nargs='+', choices=configuration.experiments.experiment_names,
                    type=str, help='name of experiment')
parser.add_argument('--include', nargs='+', choices=dgl_models.keys(), help='models to include')
parser.add_argument('--test_batch', default=None, type=int, help='predict in batches, with default being one user at a time')
parser.add_argument('--seed', default=42, type=int, help='seed for reproducibility')
parser.add_argument('--state', action='store_true', help='use state of model if it exists')
parser.add_argument('--debug', action='store_true', help='enable debug mode')
parser.add_argument('--folds', nargs='*', default=None, help='Folds to run')
parser.add_argument('--workers', default=4, type=int, help='Number of workers')
parser.add_argument('--parameter', choices=configuration.experiments.experiment_names, default=None,
                    help='supply another experiment where method should get parameters from - affects all models')
parser.add_argument('--other_model', choices=dgl_models.keys(), default=None,
                    help='supply another method which the where parameters should be located - cannot include multiple '
                         'models')
parser.add_argument('--feature_configuration', nargs='+', choices=feature_conf_names, default=['processed'],
                    help='feature_configuration to use.')
parser.add_argument('--prune', action='store_true', help='remove stored results')
parser.add_argument('--parallel', action='store_true', help='run in parallel')


def _get_parameter_path(parameter_base, model_name):
    return os.path.join(parameter_base, f'parameters_{model_name}.pickle')


def _get_model_path(output_path, fold, model_name):
    path = os.path.join(output_path, fold.experiment.name, model_name)
    if not os.path.exists(path):
        logger.debug(f'Making dirs for path {os.path.abspath(path)}')
        os.makedirs(path, exist_ok=True)  # Never true
    return path


def get_model_name(name, feature_name, require_features, other_parameter: Fold = None, other_state: Fold = None, other_model: str = None):
    return '_'.join(filter(lambda x: x is not None, [name,
                                                     # Below lines returns None or names.
                                                     '_'.join([f'features_{f.name}' for f in feature_name]) if require_features else None,
                                                     other_state and 'state_' + other_state.experiment.name,
                                                     other_parameter and 'parameter_' + other_parameter.experiment.name,
                                                     other_model and 'model_' + other_model]))


def _get_state(out_path, fold: Fold, model_name: str):
    state_path = os.path.join(out_path, f'{fold.name}_{model_name}_state.pickle')

    if not os.path.exists(state_path):
        return None
    else:
        return pickle.load(open(state_path, 'rb'))


def _instantiate_model(model_name: str, meta: Meta, seed_generator: SeedGenerator, fold: Fold,
                       feature_configurations: List[FeatureConfiguration], workers: int,
                       out_path, other_fold, other_model, train=True, parallel=False):
    kwargs = deepcopy(dgl_models[model_name])

    kwargs.update(
        {
            'meta': meta,
            'seed_generator': seed_generator,
            'use_cuda': kwargs.get('use_cuda', False),  # Ensure there is a cuda variable
            'workers': workers,
            'fold': fold,
            'other_fold': other_fold,
            'parallelized': parallel,
            'train': train
        }
    )

    # Load graph and related information
    if train:
        graph_instances = ['train', 'validation']
    else:
        graph_instances = ['train', 'test']

    path = fold.data_loader.path
    graph_names = kwargs['graphs']
    graphs = []
    infos = []
    for g_name in graph_names:
        gs = [dgl.load_graphs(os.path.join(path, f'{t}_{g_name}.dgl'))[0][0] for t in graph_instances]
        for g in gs:
            if 'recommendable' in g.ndata and isinstance(g.ndata['recommendable'], torch.Tensor):
                g.ndata['recommendable'] = g.ndata['recommendable'].type(torch.bool)
        graphs.append(gs)
        infos.append([dgl.data.utils.load_info(os.path.join(path, f'{t}_{g_name}.dgl.info')) for t in graph_instances])

    kwargs['graphs'] = graphs
    kwargs['infos'] = infos
    kwargs['sherpa_types'] = kwargs.get('sherpa_types', [])

    # Get class and instantiate
    recommender = kwargs.pop('model')
    model_path = _get_model_path(out_path, fold if other_fold is None else other_fold,
                                 model_name if not other_model else other_model)
    writer_path = _get_model_path(out_path, fold, model_name)
    instance = recommender(**kwargs, parameter_path=model_path, writer_path=writer_path)  # type: RecommenderBase

    if instance.require_features:
        logger.info('Using extracted features')
        features = {}
        for feature_configuration in feature_configurations:
            if feature_configuration.require_ratings:
                feature_path = os.path.join(fold.experiment.path, fold.name)
            else:
                feature_path = os.path.join(fold.experiment.path)

            f = []
            for name in graph_instances:
                ffn = f'features_{feature_configuration.name}_{name}.npy'
                f.append(np.load(os.path.join(feature_path, ffn), mmap_mode='r'))
            features[feature_configuration.feature_span] = f

        # Only require modification for methods that uses multiple features as input.
        if len(features) == 1:
            features = list(features.values())[0]

        instance.set_features(features)

    # Join model name, excluding
    model_name = get_model_name(model_name, feature_configurations, instance.require_features,
                                other_parameter=other_fold,
                                other_model=other_model)
    instance.name = model_name

    return instance


def _write_extra_info(out_path, fold: Fold, info, model: RecommenderBase):
    with open(os.path.join(out_path, f'{fold.name}_{model.name}_info.json'), 'w') as f:
        json.dump(info, f)

    with open(os.path.join(out_path, f'{fold.name}_{model.name}_state.pickle'), 'wb') as f:
        pickle.dump(model.get_state(), f)


def _run_model(model: RecommenderBase, validator):
    train_start = time.time()
    model.set_seed()

    model.fit(validator)
    train_end = time.time()

    meta_info = {'training_time': train_end - train_start, 'best_score': float(model._best_score)}

    return model, meta_info


def run_fold(out_path: str, model_names, fold: Fold, feature_configurations: List[FeatureConfiguration], seed: int,
             workers: int, test_batch=None, use_state=False, other_fold=None, other_model=None, prune=False,
             parallel=False):
    meta = fold.data_loader.meta()
    validator = Validator(fold.data_loader.path, test_batch, meta, full=True)

    for name in model_names:
        logger.info(f'Running model: {name}')
        model_outpath = _get_model_path(out_path, fold, name)
        if prune and os.path.exists(model_outpath):
            shutil.rmtree(model_outpath)

        model = _instantiate_model(name, meta, SeedGenerator(seed), fold, feature_configurations, workers, out_path,
                                   other_fold, other_model, parallel=parallel)

        if use_state:
            state = _get_state(model_outpath, fold, model.name)
            model.set_state(state)
        else:
            model.set_state(None)

        model, meta_info = _run_model(model, validator)

        # Only save if model have been trained without using previously saved state.
        if not use_state:
            _write_extra_info(model_outpath, fold, meta_info, model)


def run():
    args = parser.parse_args()
    args = vars(args)

    if not args.pop('debug'):
        logger.remove()
        logger.add(sys.stderr, level='INFO')

    data_path, out_path, experiments, models, test_batch, seed, state, folds, workers, experiment_parameter, \
        model_parameter, feature_names, prune, parallel = args.values()

    if experiment_parameter is not None and model_parameter is not None:
        assert len(models) == 1, "Parameter arguments are not compatible with multiple models"

    experiment_configs = [e for e in configuration.experiments.experiments if e.name in experiments]
    feature_configuration = [get_feature_configuration(feature_name) for feature_name in feature_names]

    if experiment_parameter:
        # Get experiment configuration for experiment used as parameters.
        e_conf_param = get_experiment_configuration(experiment_parameter)
        other_dataset = Dataset(os.path.join(data_path, e_conf_param.dataset.name), [experiment_parameter])
        e_conf_param = next(other_dataset.experiments())
    else:
        e_conf_param = None

    datasets = set([e.dataset.name for e in experiment_configs])
    for dataset in datasets:
        dataset = Dataset(os.path.join(data_path, dataset), experiments)
        for experiment in dataset.experiments():
            if e_conf_param is None:
                iterator = experiment.folds()
            else:
                iterator = zip(experiment.folds(), e_conf_param.folds())
            logger.info(f'Running experiment: {experiment}')
            for fold in iterator:
                if e_conf_param is None:
                    fold, other_fold = fold, None
                else:
                    fold, other_fold = fold

                if folds is None or fold.name in folds:
                    logger.info(f'Running fold: {fold}')
                    run_fold(out_path, models, fold, feature_configuration, seed, workers, test_batch=test_batch,
                             use_state=state, other_fold=other_fold, other_model=model_parameter, prune=prune,
                             parallel=parallel)
                else:
                    logger.warning(f'Skipping fold: {fold.name}')


if __name__ == '__main__':
    run()

