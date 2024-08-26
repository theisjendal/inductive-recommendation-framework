import argparse
from os.path import join
from typing import List, Dict

import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler

from configuration.experiments import experiment_names
from datasets.feature_extractors import feature_extractor
from datasets.feature_extractors.feature_extraction_base import FeatureExtractionBase
from shared.configuration_classes import FeatureConfiguration, ExperimentConfiguration
from shared.enums import FeatureEnum
from shared.utility import valid_dir, get_feature_configuration, get_experiment_configuration

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=valid_dir, help='path to datasets')
parser.add_argument('--experiment', choices=experiment_names, help='Experiment to generate features from')
parser.add_argument('--feature_configurations', nargs=3, help='feature configuration names. Combines the first and '
                                                                'second feature configuration and saves using the '
                                                                'thirds name. Note features must be extracted for the'
                                                                'two first prior to running this program.')

PHASES = ['train', 'validation', 'test']


def load(path, experiment: ExperimentConfiguration, confs: List[FeatureConfiguration], fold: str) \
        -> Dict[FeatureConfiguration, np.ndarray]:
    features = {}
    for conf in confs:
        fs = []
        for phase in PHASES:
            if conf.require_ratings:
                in_path = join(path, experiment.dataset.name, experiment.name, fold)
            else:
                in_path = join(path, experiment.dataset.name, experiment.name)

            in_path = join(in_path, f'features_{conf.name}_{phase}.npy')
            fs.append(np.load(in_path))
        features[conf] = fs
    return features


def run(path, experiment: ExperimentConfiguration, conf_a: FeatureConfiguration, conf_b: FeatureConfiguration,
        conf_out: FeatureConfiguration, fold: str = None):

    require_ratings = conf_a.require_ratings or conf_b.require_ratings
    assert require_ratings == conf_out.require_ratings, ('If any of input configurations require ratings output config '
                                                         'must do so as well.')

    features = load(path, experiment, [conf_a, conf_b], fold)
    meta, = FeatureExtractionBase.load_experiment(path, experiment, limit=['meta'])

    if conf_out.feature_span == FeatureEnum.ENTITIES:
        dim = sum([feat[0].shape[-1] if isinstance(feat, list) else feat.shape[-1] for feat in features.values()])
        new_features = [np.zeros((len(meta.entities), dim)) for _ in range(3)]
        start = 0
        for conf, feature in sorted(features.items(), key=lambda x: (x[0].feature_span, x[0].name)):
            if conf.feature_span == FeatureEnum.ITEMS:
                f_start = 0
                f_end = len(meta.items)
            elif conf.feature_span == FeatureEnum.DESC_ENTITIES:
                f_start = len(meta.items)
                f_end = None
            elif conf.feature_span == FeatureEnum.ENTITIES:
                f_start = 0
                f_end = None
            else:
                raise NotImplementedError()

            end = start + feature[0].shape[-1]
            for i in range(3):
                if isinstance(feature, list):
                    new_features[i][f_start:f_end, start:end] = feature[i]
                else:
                    new_features[i][f_start:f_end, start:end] = feature

            start = end
    else:
        raise NotImplementedError()

    scaler = None
    if conf_out.scale:
        logger.warning('Scaling both features, ensure that the features are not already scaled.')
        nf, scaler = feature_extractor.scale_features(experiment, conf_out, new_features[0], FeatureExtractionBase, path, 0)
        nfs = [nf]
        for i in range(1, 3):
            nfs.append(scaler.transform(new_features[i]))

        new_features = nfs

    for i, phase in enumerate(PHASES):
        if fold is not None:
            out_path = join(path, experiment.dataset.name, experiment.name, fold)
        else:
            out_path = join(path, experiment.dataset.name, experiment.name)
        feature_extractor.save(out_path, new_features[i], scaler, f'{conf_out.name}_{phase}')


if __name__ == '__main__':
    args = parser.parse_args()
    experiment = get_experiment_configuration(args.experiment)
    conf_a, conf_b, conf_out = [get_feature_configuration(c) for c in args.feature_configurations]
    iterate = any([conf.require_ratings for conf in [conf_a, conf_b, conf_out]])
    if iterate:
        for fold in range(experiment.folds):
            fold = f'fold_{fold}'
            run(args.path, experiment, conf_a, conf_b, conf_out, fold)
    else:
        run(args.path, experiment, conf_a, conf_b, conf_out)