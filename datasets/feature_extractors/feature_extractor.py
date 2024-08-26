import argparse
import os.path
from os.path import join

from sklearn.preprocessing import StandardScaler

from configuration.experiments import experiment_names
from datasets.feature_extractors.anchor_extractor import AnchorFeatureExtractor
from datasets.feature_extractors.complex_extractor import ComplEXFeatureExtractor
from datasets.feature_extractors.feature_extraction_base import FeatureExtractionBase
from datasets.feature_extractors.graphsage_feature_extractor import GraphSAGEFeatureExtractor
from datasets.feature_extractors.idcf_feature_extractor import IDCFFeatureExtractor
from datasets.feature_extractors.simple_feature_extractor import SimpleFeatureExtractor
from shared.configuration_classes import FeatureConfiguration
from shared.enums import ExperimentEnum, FeatureEnum
from shared.utility import valid_dir, save_numpy, save_pickle, get_experiment_configuration, get_feature_configuration

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=valid_dir, help='path to datasets')
parser.add_argument('--experiment', choices=experiment_names, help='Experiment to generate features from')
parser.add_argument('--other_experiment', default=None, choices=experiment_names,
                    help='For a cold start setting features might need to be based on the warm-start equivalent.'
                         'When scaling or similar will use this instead of experiment. Assumes item and user indices to'
                         'be equivalent. For example for key and query users, where we assume warm-start users to'
                         'be key and cold-start users to be query.')
parser.add_argument('--feature_configuration', default='graphsage', help='feature configuration name')
parser.add_argument('--cuda', action='store_true', help='use cuda, default false')
parser.add_argument('--skip_extracted', action='store_true', help='skip extraction if already extracted')


def scale_features(experiment, feature_configuration, features, feature_extractor: FeatureExtractionBase, data_path, fold):
    if experiment.experiment == ExperimentEnum.TEMPORAL:
        meta, train_users = feature_extractor.load_experiment(data_path, experiment, limit=['meta', 'train'],
                                                               fold=fold)
        indices = []
        if feature_configuration.feature_span in [FeatureEnum.ITEMS, FeatureEnum.ALL, FeatureEnum.ENTITIES]:
            train_items = set(i for u in train_users for i, _ in u.ratings)
            indices.extend(list(train_items))
        if feature_configuration.feature_span in [FeatureEnum.DESC_ENTITIES, FeatureEnum.ALL, FeatureEnum.ENTITIES]:
            indices.extend(meta.entities)
        if feature_configuration.feature_span in [FeatureEnum.USERS, FeatureEnum.ALL]:
            train_users = set(u.index for u in train_users)
            indices.extend(list(train_users))

        indices = list(sorted(set(indices)))
        X = features[indices]
    else:
        raise ValueError('Invalid experiment')

    scaler = StandardScaler().fit(X)
    features = scaler.transform(features)

    return features, scaler


def run_feature_extraction(data_path, feature_extractor: FeatureExtractionBase,
                           feature_configuration: FeatureConfiguration, fold, experiment, other_experiment=None):

    features = feature_extractor.extract_features(data_path, feature_configuration, experiment, other_experiment, fold)

    if not feature_extractor.return_for_all_settings:
        features = [features] * 3

    scaler = None
    if feature_configuration.scale:
        tmp_features = []
        for feature in features:
            if scaler is not None:
                f, scaler = scale_features(experiment, feature_configuration, feature, feature_extractor, data_path, fold)
            else:
                f = scaler.transform(feature)
            tmp_features.append(f)
        features = tmp_features

    return features, scaler


def get_feature_extractor(feature_configuration: FeatureConfiguration, use_cuda):
    if feature_configuration.extractor == 'simple':
        extractor = SimpleFeatureExtractor
    elif feature_configuration.extractor == 'graphsage':
        extractor = GraphSAGEFeatureExtractor
    elif feature_configuration.extractor == 'idcf':
        extractor = IDCFFeatureExtractor
    elif feature_configuration.extractor == 'complex':
        extractor = ComplEXFeatureExtractor
    elif 'anchor' in feature_configuration.extractor:
        extractor = AnchorFeatureExtractor
    else:
        raise NotImplementedError()

    return extractor(use_cuda=use_cuda, **feature_configuration.kwargs)


def _save(path, features, scaler, feature_name):
    if scaler is not None:
        save_pickle(join(path, f'feature_{feature_name}_meta.pickle'), {'scaler': scaler})

    save_numpy(join(path, f'features_{feature_name}.npy'), features)


def save(path, features, scaler, feature_name, feature_extractor=None):
    # If no cold-start items, features can be scaled using this method.
    if feature_extractor is not None:
        if feature_extractor.return_for_all_settings:
            assert feature_extractor.name_suffix is not None, ('Must define order of returned embeddings. Norm is '
                                                               'train, validation, test.')
            suffix = feature_extractor.name_suffix
        else:
            suffix = ['train', 'validation', 'test']

        for i, (f, n) in enumerate(zip(features, suffix)):
            _save(path, f, scaler, f'{feature_name}_{n}')
    else:
        _save(path, features, scaler, feature_name)


def run(path, experiment_name, f_conf_name, cuda, other_experiment=None, skip_extracted=False):
    experiment = get_experiment_configuration(experiment_name)
    if other_experiment is not None:
        other_experiment = get_experiment_configuration(other_experiment)
    else:
        other_experiment = None

    f_conf = get_feature_configuration(f_conf_name)
    extractor = get_feature_extractor(f_conf, cuda)

    # If extractor requires ratings, it needs to iterate all folds and the features are store under each fold.
    # If not, we just use the first fold (range(1) ~= [0]).
    out_path = os.path.join(path, experiment.dataset.name, experiment.name)
    for i in range(experiment.folds if f_conf.require_ratings else 1):
        if skip_extracted:
            skip = False
            for file in os.listdir(out_path):
                if f'features_{f_conf.name}' in file:
                    skip = True
                    break

            if skip:
                print('Skipping extraction as features already exist.')
                continue

        features, scaler = run_feature_extraction(path, extractor, f_conf, i, experiment, other_experiment)

        # If we do not require ratings save to experiment folder instead of fold folder (in experiment).
        out_path_i = out_path if not f_conf.require_ratings else os.path.join(out_path, f'fold_{i}')
        save(out_path_i, features, scaler, f_conf.name, extractor)


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.path, args.experiment, args.feature_configuration, args.cuda, args.other_experiment, args.skip_extracted)
