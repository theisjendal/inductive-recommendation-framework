import concurrent
import itertools
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor
from shared.utility import get_experiment_configuration

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--datasets', nargs='+', help='datasets to run',
                    default=['ml_1m_temporal', 'ab_temporal', 'yk_temporal'])
parser.add_argument('--methods', nargs='+', help='methods to run',
                    default=["ginrec", "pinsage", "graphsage", "random", "toppop", "bpr", "ppr", "ppr-cf",
                             "inmo", "idcf"])
parser.add_argument('--features', nargs='+', help='features to run', default=['comsage', 'comsage', 'comsage'])
parser.add_argument('--results_path', nargs=1, help='path to store results', default='./results')
parser.add_argument('--folds', nargs='+', help='folds to run', default=['fold_0'])
parser.add_argument('--gpus', nargs='+', help='gpus to run', default=['0', '1', '2', '3'])
parser.add_argument('--eval', action='store_true', help='flag for training')
parser.add_argument('--skip', action='store_true', help='flag for skipping trained folds')
parser.add_argument('--state', nargs='+', help='state to run', default=[])
parser.add_argument('--parameter', nargs='+', help='parameter to run', default=['', 'ml_1m_temporal', 'ml_1m_temporal'])
parser.add_argument('--value', nargs='+', help='parameter value to run', default=['', '1', '1'])
parser.add_argument('--other_model', nargs='+', help='other model to run', default=[])
parser.add_argument('--model_value', nargs='+', help='other model value to run', default=[])

args = parser.parse_args()
g_datasets = args.datasets
g_methods = args.methods
dataset_feature_list = args.features
RESULTS = args.results_path
g_folds = args.folds
g_gpus = args.gpus
TRAINING = not args.eval
STATE_LIST = args.state
PARAMETER_LIST = args.parameter
PARAMETER_VALUES = args.value
OTHER_MODEL = args.other_model
MODEL_VALUE = args.model_value
SKIP_TRAINED_FOLDS = args.skip

DATASET_PARAMS = {'feats': dataset_feature_list, 'params': PARAMETER_LIST, 'states': STATE_LIST,
                  'values': PARAMETER_VALUES}


def runner(folds=None, datasets=None, methods=None, gpu=None):
    gpu = gpu if gpu is not None else 0

    features = datasets.get('feats', '')
    state = datasets.get('states', '')
    parameter = datasets.get('params', '')
    value = datasets.get('values', '')
    datasets = datasets['dataset']

    experiment = get_experiment_configuration(datasets)

    workers = 4 if methods != 'igmc' else 8  # IGMC is very slow and therefore needs more workers
    if TRAINING:
        model_name = '_'.join(filter(lambda x: bool(x), [methods, features, state, parameter, OTHER_MODEL]))
        p = os.path.join(RESULTS, experiment.name, methods)
        if SKIP_TRAINED_FOLDS and os.path.isdir(p) and len(list(
                filter(lambda x: x.endswith('state.pickle') and model_name in x,
                       os.listdir(p)))):
            print(f'Skipping {p} with gpu {gpu}')
            return gpu

        str_arg = (f"CUDA_VISIBLE_DEVICES={gpu} python3 train/dgl_trainer.py --data ./datasets --out_path {RESULTS} "
                   f"--experiments {experiment.name} --include {methods} --test_batch 1024 --workers={workers} "
                   f"--folds {folds} --feature_configuration {features}")
        str_arg += f" --parameter {parameter}" if parameter else ""
        str_arg += f" --other_model {OTHER_MODEL}" if OTHER_MODEL else ""
    else:

        str_arg = f"CUDA_VISIBLE_DEVICES={gpu} python3 evaluate/dgl_evaluator.py --data ./datasets --results_path {RESULTS} "\
                    f"--experiments {experiment.name} --include {methods} --test_batch 1024 --workers {workers} "\
                    f"--folds {folds} --feature_configuration {features}"
        str_arg += f" --parameter {parameter}" if parameter else ""
        str_arg += f" --parameter_usage {value}" if value else ""
        str_arg += f" --other_model {OTHER_MODEL}" if OTHER_MODEL else ""
        str_arg += f" --other_model_usage {MODEL_VALUE}" if MODEL_VALUE else ""

    p = subprocess.Popen(str_arg, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
    for line in p.stdout:
        print(line)

    p.wait()

    return gpu


def run():
    ngpus = len(g_gpus)

    params = []
    for i in range(len(g_datasets)):
        d_params = {'dataset': g_datasets[i]}
        for key, value in DATASET_PARAMS.items():
            if value:
                d_params[key] = value[i]
        params.append(d_params)

    combinations = list(itertools.product(g_folds, params, g_methods))

    futures = []
    first = True
    with ThreadPoolExecutor(max_workers=ngpus) as e:
        while combinations:
            # should only be false on first iteration
            if first:
                # start process on each gpu. Zip ensures we do not iterate more than num gpus or combinations.
                for _, gpu in list(zip(combinations, g_gpus)):
                    futures.append(e.submit(runner, *combinations.pop(0), gpu))

                first = False
            else:
                # Check if any completed
                completed = list(filter(lambda x: futures[x].done(), range(len(futures))))

                # if any process is completed start new on same gpu; otherwise, wait for one to finish
                if completed:
                    f = futures.pop(completed[0])
                    gpu = f.result()
                    futures.append(e.submit(runner, *combinations.pop(0), gpu))
                else:
                    concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)

    concurrent.futures.wait(futures)


if __name__ == '__main__':
    run()