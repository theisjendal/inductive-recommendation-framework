import concurrent
import itertools
import os
import sys
import pickle
import subprocess
from concurrent.futures import ThreadPoolExecutor

from shared.filelock import FileLock
from shared.utility import get_experiment_configuration

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--datasets', nargs='+', help='datasets to run',
                    default=['ml_1m_temporal'])
parser.add_argument('--methods', nargs='+', help='methods to run',
                    default=["ginrec", "pinsage", "graphsage", "random", "toppop", "bpr", "ppr", "ppr-cf",
                             "inmo", "idcf"])
parser.add_argument('--features', nargs='+', help='features to run', default=['comsage'])
parser.add_argument('--results_path', nargs=1, help='path to store results', default='./results')
parser.add_argument('--folds', nargs='+', help='folds to run', default=['fold_0'])
parser.add_argument('--gpus', nargs='+', help='GPU ids to run on, same value can be inserted multiple times',
                    default=['0', '1', '2', '3'])
parser.add_argument('--num_workers', nargs=1, help='number of workers to run', default=4)

args = parser.parse_args()
g_datasets = args.datasets
g_methods = args.methods
dataset_feature_list = args.features
RESULTS = args.results_path
g_folds = args.folds
g_gpus = args.gpus
WORKERS = args.num_workers

DATASET_PARAMS = {'feats': dataset_feature_list}


def runner(fold=None, dataset=None, method=None, gpu=None):
    gpu = gpu if gpu is not None else 0

    features = dataset.get('feats', '')
    dataset = dataset['dataset']

    experiment = get_experiment_configuration(dataset)

    workers = WORKERS if method != 'igmc' else 8  # IGMC is very slow and therefore needs more workers
    str_arg = ""

    str_arg += f"CUDA_VISIBLE_DEVICES={gpu} python3 train/dgl_trainer.py --data ./datasets --out_path {RESULTS} "\
              f"--experiments {experiment.name} --include {method} --test_batch 1024 --debug --workers={workers} "\
              f"--folds {fold} --feature_configuration {features} --parallel"

    p = subprocess.Popen(str_arg, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
    for line in p.stdout:
        print(line)

    p.wait()

    return gpu


def has_next(path_to_states):
    try:
        with FileLock(path_to_states + '.lock'):
            if os.path.isfile(path_to_states):
                with open(path_to_states, 'rb') as f:
                    state = pickle.load(f)

                study = state.get('study', None)
                if study is not None:
                    study.next()
        return True
    except StopIteration:
        return False


def method_runner(fold, dataset, method):
    futures = []
    first = True
    ngpus = len(g_gpus)
    os.makedirs(os.path.join(RESULTS, dataset['dataset'], method), exist_ok=True)
    parameter_path = os.path.join(RESULTS, dataset['dataset'], method, 'parameters.states')
    with ThreadPoolExecutor(max_workers=ngpus) as e:
        while has_next(parameter_path):

            # should only be false on first iteration
            if first:
                # start process on each gpu. Zip ensures we do not iterate more than num gpus or combinations.
                for gpu in g_gpus:
                    futures.append(e.submit(runner, fold, dataset, method, gpu))

                first = False
            else:
                # Check if any completed
                completed = list(filter(lambda x: futures[x].done(), range(len(futures))))

                # if any process is completed start new on same gpu; otherwise, wait for one to finish
                if completed and has_next(parameter_path):
                    f = futures.pop(completed[0])
                    gpu = f.result()
                    futures.append(e.submit(runner, fold, dataset, method, gpu))
                else:
                    concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

    concurrent.futures.wait(futures)


def run():
    params = []
    for i in range(len(g_datasets)):
        d_params = {'dataset': g_datasets[i]}
        for key, value in DATASET_PARAMS.items():
            if value:
                d_params[key] = value[i]
        params.append(d_params)

    combinations = list(itertools.product(g_folds, params, g_methods))
    for combination in combinations:
        method_runner(*combination)


if __name__ == '__main__':
    run()