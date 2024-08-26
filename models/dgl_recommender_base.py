import itertools
import os.path
import pickle
import time
from copy import deepcopy
from fcntl import flock

import dgl
import numpy.random
import sherpa
import torch
from loguru import logger
from sherpa import Trial
from torch.utils.tensorboard import SummaryWriter

import random

import numpy as np

from datasets.feature_extractors.feature_extractor import get_feature_extractor
from models.utility import SuccessiveHalvingWrapper
from shared.efficient_validator import Validator
from shared.enums import RecommenderEnum
from shared.experiments import Fold
from shared.filelock import FileLock
from shared.meta import Meta
from shared.seed_generator import SeedGenerator
from shared.utility import get_feature_configuration, get_experiment_configuration


class RecommenderBase:
    def __init__(self, recommender_type: RecommenderEnum, meta: Meta, seed_generator: SeedGenerator, use_cuda, graphs, infos, hyperparameters, fold: Fold,
                 sherpa_types, workers, parameter_path, writer_path, features=False, gridsearch=False, ablation_parameter=None,
                 feature_configuration=None, other_fold=None, parallelized=False, batch_size=4096, train=True):
        """
        Baseclass used by all methods. Contain base fit method for running methods.
        :param recommender_type: Type of recommender.
        :param meta: Contains meta information about the dataset, e.g. users, items, entities.
        :param seed_generator: Used to generate and set seeds. Does not effect sherpa optimizer.
        :param use_cuda: To use cuda or not.
        :param graphs: Graphs to be used in training.
        :param infos: Information about the graphs.
        :param hyperparameters: Hyperparameters of the model.
        :param sherpa_types: Mapping of hyperparameters to sherpa types.
        :param workers: Number of workers used for sampling.
        :param parameter_path: Path to store model trials and similar.
        :param features: To use pre-calculated features or not.
        :param gridsearch: To use gridsearch or not.
        :param ablation_parameter: Used for ablation studies. Changes one parameter to another value.
        """
        self.name = ''
        self.recommender_type = recommender_type
        self.meta = meta
        self.seed_generator = seed_generator
        self.use_cuda = use_cuda
        self.fold = fold
        self.graphs = graphs
        self.infos = infos
        self._hyperparameters = hyperparameters
        self.ablation_parameter = ablation_parameter
        self._sherpa_types = sherpa_types
        self.workers = workers
        self._other_fold = other_fold
        self._gridsearch = gridsearch
        self.parallelized = parallelized

        self.device = torch.device('cuda:0') if self.use_cuda else torch.device('cpu')
        self.batch_size = batch_size

        self._random = random.Random(seed_generator.get_seed())
        self.require_features = features
        self.trials_per_parameter = 64
        self._max_epochs = 1000
        self._min_epochs = 2
        self.asha_r = 2
        self.asha_eta = 2
        self.asha_s = 0
        self._features = None
        self._feature_configuration = feature_configuration
        self.parameter_path = parameter_path
        self._full_parameter_path = f'{self.parameter_path}/parameters.states'
        self.writer_path = writer_path
        self._model = None
        self._optimizer = None
        self._scheduler = None
        self._parameters = None
        self._state = None
        self._seed = seed_generator.get_seed()

        self._study = None
        self._sherpa_parameters = None
        self._trial = None
        self._epoch_resource = {}
        self._no_improvements = 0
        self._best_state = None
        self._best_score = -1
        self._last_score = -1
        self._best_epoch = 0
        self.summary_writer = None
        if self._gridsearch:
            self._min_epochs = 0
            self._eval_intermission = 5
            self._early_stopping = 10
        else:
            self._eval_intermission = 5
            self._early_stopping = 10

        self.__start_time = time.time()
        self.train = train

        sherpa.rng = numpy.random.RandomState(self._seed)
        self.set_seed()

    def _create_parameters(self):
        parameters = []
        for key, value in self._hyperparameters.items():
            t = self._sherpa_types[key]
            if t in ['continuous', 'discrete']:
                min_v = np.min(value)
                max_v = np.max(value)

                if key in ['learning_rates', 'weight_decays', 'autoencoder_weights', 'deltas', 'anchors',
                           'disentanglement_weight', 'l2_loss', 'auxiliary_weight', 'trans_weights']:
                    scale = 'log'
                else:
                    scale = 'linear'

                if t == 'continuous':
                    parameters.append(sherpa.Continuous(name=key, range=[min_v, max_v], scale=scale))
                else:
                    parameters.append(sherpa.Discrete(name=key, range=[min_v, max_v], scale=scale))
            elif t == 'choice':
                parameters.append(sherpa.Choice(name=key, range=value))
            else:
                raise ValueError(f'Invalid sherpa type or not implemented: {t}')

        self._sherpa_parameters = parameters

    def _parameter_combinations(self):
        has_next = True
        self._create_parameters()
        while has_next:
            with FileLock(self._full_parameter_path + '.lock'):
                # Get study may be created by other thread.
                self._load()

                if self._study is None:
                    n_params = len([p for p in self._hyperparameters if len(self._hyperparameters[p]) > 1])
                    if self._gridsearch:
                        algorithm = sherpa.algorithms.GridSearch()
                    else:
                        logger.debug(f'Number of parameters: {n_params}, '
                                     f'approx trials: {self.trials_per_parameter * n_params}')
                        algorithm = SuccessiveHalvingWrapper(
                            r=self.asha_r, R=self.trials_per_parameter * n_params, eta=self.asha_eta, s=0,
                            max_finished_configs=1
                            # r=1, R=27, eta=3, s=0,  max_finished_configs=1
                        )

                    study = sherpa.Study(parameters=self._sherpa_parameters, algorithm=algorithm, lower_is_better=False,
                                         disable_dashboard=True)
                    self._study = study

                # If parallelized, only run one trial.
                try:
                    trial = self._study.next()
                except StopIteration:
                    has_next = False
                    continue

                if self.invalid_configuration(trial.parameters):
                    self._study.add_observation(trial=trial, iteration=0, objective=0)
                    self._study.finalize(trial=trial, status='FAILED')
                    self._save()
                    continue
                else:
                    self._save()

            if self.parallelized:
                has_next = False
                yield trial
            else:
                yield trial

    def _create_model(self, trial):
        raise NotImplementedError

    def _epoch_range(self, trial):
        n_epochs = trial.parameters.get('resource', self._max_epochs)

        if not self._epoch_resource:
            self._epoch_resource[n_epochs] = 0
        elif n_epochs not in self._epoch_resource:
            m = max(self._epoch_resource)
            self._epoch_resource[n_epochs] = m + self._epoch_resource[m]
            if self._epoch_resource[m] == 0:
                self._epoch_resource[n_epochs] += self._min_epochs

        init_epoch = self._epoch_resource[n_epochs]
        end_epoch = n_epochs + init_epoch
        if init_epoch == 0:
            end_epoch += self._min_epochs

        # Set timer
        self.__start_time = time.time()

        return init_epoch, end_epoch

    def _to_report(self, start_epoch, end_epoch, cur_epoch, validator, trial, loss=None, **kwargs):
        n_epochs = end_epoch - start_epoch
        eval_intermission = self._eval_intermission
        for eval_intermission in reversed(range(1, self._eval_intermission+1)):
            count = n_epochs // eval_intermission + 1
            if count >= 5:
                break

        if (cur_epoch-start_epoch) % eval_intermission == 0 or cur_epoch == end_epoch-1:
            score = self._last_score
            timed = False
            t_time = time.time() - self.__start_time
            if self._no_improvements < self._early_stopping:
                st = time.time()
                self._inference(**kwargs)
                i_time = time.time() - st
                st = time.time()
                score = validator.validate(self)
                v_time = time.time() - st
                timed = True
                self._last_score = score
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('score/ndcg', score, cur_epoch)
                self.summary_writer.add_scalar('statistics/train_time', t_time, cur_epoch)
                if timed:
                    self.summary_writer.add_scalar('statistics/inference_time', i_time, cur_epoch)
                    self.summary_writer.add_scalar('statistics/validation_time', v_time, cur_epoch)
            self._on_epoch_end(trial, score, cur_epoch, loss=loss)

    def _on_epoch_end(self, trial, score, epoch, loss=None):
        if trial is not None:
            end_time = time.time()
            context = {'time': end_time - self.__start_time} if self._early_stopping > self._no_improvements else {}
            self.summary_writer.add_scalar('statistics/time', end_time-self.__start_time, epoch)

            if loss:  # Assume loss is never zero.
                try:
                    loss = loss.item()  # try to convert, ignore if not of tensor type
                except AttributeError:
                    pass

                context['loss'] = loss

            with FileLock(self._full_parameter_path + '.lock'):
                self._load()
                self._study.add_observation(trial=trial, iteration=epoch,
                                            objective=score, context=context)
                self._save()
            self.__start_time = end_time

        if epoch < self._min_epochs:
            pass  # ensure model runs for at least min epochs before storing best state
        elif score > self._best_score:
            self._best_state = deepcopy(self._model.state_dict())
            self._best_score = score
            self._best_epoch = epoch
            self._no_improvements = 0
        else:
            self._no_improvements += 1

        if self._no_improvements < self._early_stopping:
            name = f'{self.name}_{trial.id}' if trial is not None else self.name
            logger.debug(f'Validation for {name}: {score}, best epoch: {self._best_epoch}, '
                         f'current epoch: {epoch}')

    def _save(self):
        """
        Saves current state and parameters, should be used after each tuning.
        :param score: score for state and parameters.
        """
        with open(self._full_parameter_path, 'wb') as f:
            state = {'study': self._study, 'resources': self._epoch_resource}
            pickle.dump(state, f)

    def _get_best_trial(self, load_state):
        with FileLock(self._full_parameter_path + '.lock'):
            self._load()
            result = self._study.get_best_result()
        self.set_parameters(result)
        if load_state:
            result['load_from'] = result.get('save_to', str(result.get('Trial-ID')))
        else:
            result['load_from'] = ""
        return Trial(result.pop('Trial-ID'), result)

    def _sherpa_load(self, trial):
        load_from = trial.parameters.get('load_from', '')

        if load_from != "":
            p = os.path.join(self.parameter_path, load_from) + '.trial'
            logger.info(f"Loading model from {p}")

            # Loading model
            checkpoint = torch.load(p, map_location=self.device)

            self._model.load_state_dict(checkpoint['model'])
            if self._optimizer is not None:
                self._optimizer.load_state_dict(checkpoint['optimizer'])
            if self._scheduler is not None:
                self._scheduler.load_state_dict(checkpoint['scheduler'])
            self._best_score = checkpoint['score']
            self._last_score = checkpoint.get('last_score', self._best_score)
            self._best_epoch = checkpoint.get('epoch', 0)
            self._best_state = checkpoint['state']
            self._no_improvements = checkpoint['iter']
        else:
            self._best_score = 0
            self._best_state = None
            self._no_improvements = 0
            self._best_epoch = 0

    def _sherpa_save(self, trial):
        torch.save({
            'model': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict() if self._optimizer is not None else None,
            'scheduler': self._scheduler.state_dict() if self._scheduler is not None else None,
            'score': self._best_score,
            'last_score': self._last_score,
            'state': self._best_state,
            'epoch': self._best_epoch,
            'iter': self._no_improvements
        }, os.path.join(self.parameter_path, trial.parameters.get('save_to', str(trial.id))) + '.trial')

    def _load(self):
        """
        Loads best state and parameters from temporary directory if it exists.
        """
        fname = self._full_parameter_path
        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                state = pickle.load(f)

            self._study = state.get('study', None)
            self._epoch_resource = state.get('resources', {})

    def _parameter_to_string(self, parameters):
        if isinstance(parameters, str):
            return parameters
        elif isinstance(parameters, tuple) or isinstance(parameters, int):
            return str(parameters)
        elif isinstance(parameters, dict):
            return {k: self._parameter_to_string(v) for k,v in parameters.items()}
        elif parameters is None:
            return ''
        else:
            return f'{parameters:.2E}'

    def _get_ancestor(self, identifier):
        df = self._study.results
        df = df[df['Status'] == 'COMPLETED'].set_index('Trial-ID')
        t = df.loc[identifier]
        if t.load_from == '':
            return identifier
        else:
            return self._get_ancestor(int(t.load_from))

    def fit(self, validator: Validator):
        change = False

        for trial in self._parameter_combinations():
            logger.info(f"Trial:\t{trial.id}")
            if trial.parameters.get('load_from', '') != '':
                a = self._get_ancestor(int(trial.parameters['load_from']))
            else:
                a = trial.id
            self.summary_writer = SummaryWriter(log_dir=os.path.join(self.writer_path, str(a)))
            
            self.set_parameters(trial.parameters)  # Set parameters
            # UVA will create an error if the graph is already pinned
            for i, g_set in enumerate(self.graphs):
                for g in g_set:
                    if g.is_pinned():
                        g.unpin_memory_()

            self.set_seed()
            param_str = self._parameter_to_string(trial.parameters)
            logger.debug(f'{self.name} parameter: {param_str}')
            self._create_model(trial)  # initialize model with parameters given by trial

            with FileLock(self._full_parameter_path + '.lock'):
                self._load()
                first, final = self._epoch_range(trial)
                self._save()
            logger.info(f'Epochs {first} - {final}')

            status = 'COMPLETED'
            error = None
            try:
                self._fit(validator, first, final, trial=trial)  # train
            except Exception as e:
                status = 'FAILED'
                error = e
            finally:
                # save trial and state of study (sherpa study)
                with FileLock(self._full_parameter_path + '.lock'):
                    self._load()
                    if 'Trial-ID' not in self._study.results or trial.id not in self._study.results['Trial-ID'].values:
                        self._study.add_observation(trial=trial, iteration=first, objective=self._best_score)
                    self._study.finalize(trial=trial, status=status)
                    self._save()

                self._sherpa_save(trial)

            if error is not None:
                raise error

            change = True
            self.summary_writer.add_hparams(trial.parameters, {'score': self._best_score})
            self.summary_writer.close()

        # If parallelized, exit after one trial.
        if change and self.parallelized:
           exit(0)

        trial = self._get_best_trial(load_state=change)

        # Run with ablation parameter
        if self.ablation_parameter:
            trial.parameters.update(self.ablation_parameter)

        logger.info(f'Using best parameters {trial.parameters}')
        self.set_parameters(trial.parameters)
        self._create_model(trial)
        first = self._epoch_range(trial)[-1] if change else 0

        # If no state is available or a change have occurred, find state.
        if change or self.get_state() is None:

            # Train in we can still train, i.e. both max epochs and early stopping criterion haven't been reached.
            if self._no_improvements < self._early_stopping and first < self._max_epochs:
                if self.summary_writer is None:
                    # Use different directory for final training
                    self.summary_writer = SummaryWriter(log_dir=os.path.join(self.writer_path,
                                                                             str(trial.id) + '_final'))
                else:
                    self.summary_writer.log_dir = SummaryWriter(log_dir=os.path.join(self.writer_path, str(trial.id)))
                self._fit(validator, first, self._max_epochs)

            self.set_state(self._best_state)

        # Should occur for all torch methods.
        if self.get_state() is not None:
            self._model.load_state_dict(self.get_state())

    def _fit(self, validator: Validator, first_epoch, final_epoch=1000, trial=None):
        raise NotImplementedError()

    def _inference(self, **kwargs):
        raise NotImplementedError()

    def predict_all(self, users: np.array) -> np.array:
        """
        Score all items given users
        :param users: user ids to predict for
        :return: an matrix of user-item scores
        """
        raise NotImplementedError()

    def set_features(self, features):
        """
        Sets entity features either pretrained or extracted.
        :param features as ndarray, possibly memory mapped.
        :return: None
        """
        self._features = features

    def set_seed(self):
        """
        Seed all random generators, e.g., random, numpy.random, torch.random...
        """
        # dgl.random.seed(self._seed)
        # dgl.seed(self._seed)
        torch.random.manual_seed(self._seed)
        torch.manual_seed(self._seed)
        torch.cuda.manual_seed(self._seed)
        np.random.seed(self._seed)
        random.seed(self._seed)
        # torch.backends.cudnn.determinstic = True
        # torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)

    def set_parameters(self, parameters):
        """
        Set all model parameters, often hyperparameters.
        """
        raise NotImplementedError()

    def set_optimal_parameter(self, parameters):
        self._parameters = parameters

    def get_parameters(self) -> dict:
        """
        Get model parameters or hyperparameters.
        :return: Dictionary of parameters
        """
        return self._parameters

    def set_state(self, state):
        """
        Set the model state.
        :param state: State that a model can use for retraining or prediction.
        """
        self._state = state

    def get_state(self):
        """
        Return state of model.
        :return: state of model.
        """
        return self._state

    def get_features(self, feature_kwargs, extract_kwargs):
        if self._feature_configuration is not None:
            f_conf = get_feature_configuration(self._feature_configuration)
            f_conf.kwargs.update(feature_kwargs)
            anchor_path = os.path.join(self.parameter_path, 'anchor')

            os.makedirs(anchor_path, exist_ok=True)

            extension = '' if not feature_kwargs else '_' + '_'.join(map(str, itertools.chain(*feature_kwargs.items())))
            f_path = os.path.join(anchor_path, f_conf.name + extension + '.npy')
            #todo create lock
            #todo out path should be fold_0 instead of results?
            if os.path.isfile(f_path):
                features = np.load(f_path)
            else:
                e_conf = get_experiment_configuration(self.fold.experiment.name)
                d_path = self.fold.data_loader.path
                d_path = d_path.rsplit('/', 3)[0]  # ignore /datset/experiment/fold
                fold_num = self.fold.name.rsplit('_', 1)[1]

                f_extractor = get_feature_extractor(f_conf, self.use_cuda)
                features = f_extractor.extract_features(d_path, f_conf, e_conf, fold=fold_num)
                np.save(f_path, features)

            if self.train:
                return features[0], features[1]
            else:
                return features[0], features[2]
        else:
            return self._features

    def invalid_configuration(self, parameters):
        return False