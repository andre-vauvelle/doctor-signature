import debugpy

debugpy.listen(5678)
import math
import numpy as np

import hyperopt
from allennlp.nn import Activation
from hyperopt import fmin, hp
from hyperopt.mongoexp import MongoTrials
from hyperopt.pyll import scope
from itertools import permutations

from experiments import ALL_AUGMENTATION_OPTIONS
from experiments.SIG import hyperopt_objective

import warnings


# see https://stackoverflow.com/a/40846742
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

if __name__ == '__main__':
    for p in [0.025]:
        name = 'SIG'
        version = 2.3
        max_evals = 100
        curr_db_name = 'hyperopt'

        # Define the search space
        space = {
            "name": name,
            "version": version,
            "all_code_types": True,
            'logsig': True,
            'sig_depth': 2,
            'add_time': True,
            "use_timestamps": True,
            "t_max": hp.uniform('t_max', 0, 1), # not to explode activations
            "t_scale": hp.uniform('t_scale', 86400, 604800), # days and weeks
            'leadlag': True,
            "split_paths": False,
            "min_count": 5,
            "batch_size": 128,
            "d_embedding": scope.int(hp.quniform('d_embedding', 16, 64, 1)),
            "epochs": 20,
            "lr": hp.loguniform('lr', np.log(1e-5), np.log(1e-2)),
            "wd": hp.loguniform('wd', np.log(1e-7), np.log(1e-4)),
            "patience": 5,
            "feedforward_num_layers": hp.uniformint('feedforward_num_layers', 1, 2),
            "embedding_dropout_p": 0,
            "verbose": True,
            # "testing_subsample_size": 1000
            "feedforward_hidden_dims": scope.int(hp.quniform('feedforward_hidden_dims', 32, 256, 4)),
            "feedforward_activations": "relu",
            "feedforward_dropout": hp.uniform('feedforward_dropout', 0, 0.7),
            "training_proportion": p,
            "evaluate_on_test": False
        }

        space.update({
            "name": "{name}_{version}_logsig{logsig}_sigdepth{sig_depth}_leadlag{leadlag}_addtime_{add_time}_timestamps{use_timestamps}_allcode{all_code_types}_trainprop{training_proportion}".format_map(
                space)})

        # Set mongo trail name
        trials = MongoTrials('mongo://bigtop:27017/{}/jobs'.format(curr_db_name),
                             exp_key=space['name'])
        print('using exp_key: {}'.format(space['name']))

        print('Pending on workers to connect ..')
        print('db = ', curr_db_name)
        argmin = fmin(fn=hyperopt_objective,
                      space=space,
                      algo=hyperopt.tpe.suggest,
                      max_evals=max_evals,
                      trials=trials,
                      verbose=True)
        best_acc = 1 - trials.best_trial['result']['loss']

        print('best val acc=', best_acc, 'params:', argmin)
