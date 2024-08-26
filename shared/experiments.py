import os
from os.path import join
from typing import List

from shared.meta import Meta
from shared.utility import load_pickle


class DataLoader:
    def __init__(self, path, meta_path):
        self.path = path
        self._meta_path = meta_path

    def _get_path(self, file):
        return join(self.path, file)

    def training(self) -> list:
        return load_pickle(self._get_path('train.pickle'))

    def testing(self) -> list:
        return load_pickle(self._get_path('test.pickle'))

    def validation(self) -> list:
        return load_pickle(self._get_path('validation.pickle'))

    def meta(self, recommendable_only: bool = False) -> Meta:
        meta = load_pickle(self._meta_path)
        meta.recommendable_only = recommendable_only

        return meta


class Fold:
    def __init__(self, parent, path, meta_path):
        file_seen = {file: False for file in ['train.pickle', 'test.pickle', 'validation.pickle']}

        if not os.path.exists(path):
            raise IOError(f'Fold path {path} does not exist')

        for file in os.listdir(path):
            if file in file_seen:
                file_seen[file] = True

        if not all(file_seen.values()):
            raise IOError(f'Fold path {path} is missing at least one required file')

        self.experiment = parent
        self.name = os.path.basename(path)
        self.data_loader = DataLoader(path, meta_path)

    def __str__(self):
        return f'{self.experiment}/{self.name}'


class Experiment:
    def __init__(self, parent, path):
        if not os.path.exists(path):
            raise IOError(f'Experiment path {path} does not exist')

        self.path = path
        self.dataset = parent
        self.name = os.path.basename(path)
        self.fold_paths = []
        self.meta_path = join(path, 'meta.pickle')

        if not os.path.isfile(self.meta_path):
            raise IOError(f'Experiment meta is missing at {self.meta_path}')

        for directory in os.listdir(path):
            full_path = os.path.join(path, directory)
            if not os.path.isdir(full_path):
                continue

            self.fold_paths.append(full_path)

        if not self.fold_paths:
            raise RuntimeError(f'Experiment path {path} contains no splits')

        self.fold_paths = sorted(self.fold_paths)

    def __str__(self):
        return f'{self.dataset}/{self.name}'

    def folds(self):
        for path in self.fold_paths:
            yield Fold(self, path, self.meta_path)


class Dataset:
    def __init__(self, path: str, experiments: List[str]):
        if not os.path.exists(path):
            raise IOError(f'Dataset path {path} does not exist')

        self.name = os.path.basename(path)
        self.experiment_paths = []

        for item in os.listdir(path):
            full_path = join(path, item)

            if not os.path.isdir(full_path) or experiments and item not in experiments:
                continue

            self.experiment_paths.append(full_path)

        if not self.experiment_paths:
            raise RuntimeError(f'Dataset path {path} contains no experiments')

    def __str__(self):
        return self.name

    def experiments(self):
        for path in self.experiment_paths:
            yield Experiment(self, path)
