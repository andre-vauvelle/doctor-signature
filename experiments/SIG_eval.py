import math
import os

import numpy as np
import hyperopt
from allennlp.nn import Activation
from hyperopt import fmin, hp
from hyperopt.mongoexp import MongoTrials
from hyperopt.pyll import scope
from itertools import permutations

from definitions import MODEL_DIR
from experiments import ALL_AUGMENTATION_OPTIONS
from experiments.SIG import hyperopt_objective
import warnings
import argparse

from src.omni.functions import load_json, save_json

parser = argparse.ArgumentParser()
parser.add_argument("config", help="absolute path to config json file", type=str)
args = parser.parse_args()

# see https://stackoverflow.com/a/40846742
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

if __name__ == '__main__':
    curr_db_name = 'hyperopt'
    max_evals = 1

    # Define the search space
    space = load_json(args.config)

    # Set mongo trail name
    trials = MongoTrials('mongo://bigtop:27017/{}/jobs'.format(curr_db_name),
                         exp_key=space['name'] + '_eval')
    print('using exp_key: {}'.format(space['name'] + '_eval'))

    print('Pending on workers to connect ..')
    print('db = ', curr_db_name)
    argmin = fmin(fn=hyperopt_objective,
                  space=space,
                  algo=hyperopt.tpe.suggest,
                  max_evals=max_evals,
                  trials=trials,
                  verbose=True)
    best_acc = 1 - trials.best_trial['result']['loss']
    save_json(trials.best_trial['result'], os.path.join(MODEL_DIR, 'configs', 'results', space['name'] + '_eval.json'))
    print(trials.best_trial['result'])
