import argparse
import random
from os import makedirs
from os.path import join, isdir

from sklearn.model_selection import KFold

from configuration.experiments import experiment_names
from datasets.partitioners import temporal_splitting
from shared.enums import ExperimentEnum
from shared.meta import Meta
from shared.seed_generator import SeedGenerator
from shared.utility import valid_dir, load_entities, load_users, load_relations, save_pickle, \
    get_experiment_configuration

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=valid_dir, help='Path to datasets')
parser.add_argument('--experiment', choices=experiment_names, help='Experiment to create.')


def run(in_path, experiment_name):
    # Get experiment
    experiment = get_experiment_configuration(experiment_name)

    # Initialise rand
    sg = SeedGenerator(experiment.seed)
    random.seed(sg.get_seed())

    in_path = join(in_path, experiment.dataset.name)
    path = join(in_path, experiment.name)

    if not isdir(path):
        makedirs(path)

    entities = load_entities(in_path, fname_extension='processed')
    users = load_users(in_path, fname_extension='processed')
    relations = load_relations(in_path, fname_extension='processed')

    kf = None

    meta = Meta(entities, users, relations, sentiment_utility=experiment.dataset.sentiment_utility)
    save_pickle(join(path, 'meta.pickle'), meta)

    if experiment.experiment == ExperimentEnum.TEMPORAL:
        folds = temporal_splitting.fold_data_iterator
    else:
        raise ValueError('Invalid experiment')

    fold = 0
    iter_folds = iter(folds(path, experiment, kf, entities, users, relations, sg))
    for train in iter_folds:
        fold_name = f'fold_{fold}'
        complete_path = join(path, fold_name)

        if not isdir(complete_path):
            makedirs(complete_path)

        # folds first yields train, afterwards, validation and then test. We therefore do not need to store them in
        # memory for long.
        save_pickle(join(complete_path, 'train.pickle'), train)
        save_pickle(join(complete_path, 'validation.pickle'), next(iter_folds))
        save_pickle(join(complete_path, 'test.pickle'), next(iter_folds))

        fold += 1


if __name__ == '__main__':
    args = parser.parse_args()
    run(args.path, args.experiment)
