import os
import numpy as np

import hyperopt
from hyperopt.mongoexp import MongoTrials
from hyperopt.pyll import scope
from hyperopt import hp, fmin, Trials

from experiments.BoWLR_sklearn import hyperopt_objective

base = os.path.basename(__file__)
base_dir = os.path.dirname(__file__)
experiment_name = os.path.splitext(base)[0]
save_at = os.path.join(base_dir, experiment_name)

if __name__ == '__main__':
    name = 'BoWLR_W2V_sklearn'
    version = 2.1
    max_evals = 100
    all_code_types = True
    if all_code_types:
        name = name + '_all_code_types'
    if os.getenv('LOCAL'): 
        trials = Trials()
    else:
        trials = MongoTrials('mongo://localhost:27017/hyperopt/jobs',
                             exp_key='%s_v%s' % (name, version))
    print('using exp key %s_v%s' % (name, version))

    space = {
        "name": name,
        "version": version,
        'all_code_types': all_code_types,
        'batch_size': 128,
        'min_count': 5,
        'w2v': True,
        "d_embedding": scope.int(hp.quniform("d_embedding", 16, 80, 8)),  # Genism works well on multiple of 4
        'w2v_epochs': 150,
        "w2v_window": scope.int(hp.quniform("w2v_window", 5, 30, 5)),
        "w2v_mincount": 5,
        "w2v_sampling_method": 0,
        'C': hp.loguniform('C', np.log(1e-6), np.log(1e-1)),
        'verbose': False,
        'max_iter': 1000,
        'penalty': 'l2'
    }

    argmin = fmin(fn=hyperopt_objective,
                  space=space,
                  algo=hyperopt.tpe.suggest,
                  max_evals=max_evals,
                  trials=trials,
                  verbose=True)
    best_acc = 1 - trials.best_trial['result']['loss']

    print('best val acc=', best_acc, 'params:', argmin)
