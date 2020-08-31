import os
import logging
import sys

import torch
import numpy as np
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from sklearn.metrics import roc_auc_score, recall_score, average_precision_score, f1_score
from sklearn.model_selection import cross_validate, StratifiedKFold

from definitions import MODEL_DIR
from src.data import generate_ml_data
from src.features.feature_pipeline import FeaturePipeline, EmbedTransform, OneHotEmbedTransform, PoolTransform
from src.omni.functions import save_pickle

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

base = os.path.basename(__file__)
base_dir = os.path.dirname(__file__)
experiment_name = os.path.splitext(base)[0]
save_at = os.path.join(base_dir, experiment_name)

ex = Experiment(experiment_name)
my_url = '127.0.0.1:27017'  # Or <server-static-ip>:<port> if running on server
curr_db_name = experiment_name
ex.captured_out_filter = apply_backspaces_and_linefeeds

LOCAL = os.environ.get('LOCAL')
TEST = True
K_FOLDS = 5

if LOCAL:
    ex.observers.append(FileStorageObserver(save_at))
    logger.info('saving file observer at {}'.format(save_at))
else:
    ex.observers.append(MongoObserver(url=my_url,
                                      db_name=curr_db_name))
    logger.info('saving in mongodb {}'.format(curr_db_name))


@ex.config
def tsee_logreg_config():
    """Tsee configation for logistic regresssion with sklearn"""
    name = "BoWLR_sklearn"
    version = -1
    all_code_types = False
    C = 0.010590518372202755
    w2v = False
    d_embedding = 100
    w2v_window = 5
    w2v_epochs = 150
    evaluate_on_test = True
    max_iter=100


@ex.capture()
def run(
        all_code_types,
        C,
        w2v,
        d_embedding,
        w2v_window,
        w2v_epochs,
        w2v_sampling_method=0,
        solver="lbfgs",
        penalty="l2",
        verbose=True,
        min_count=5,
        w2v_mincount=5,
        batch_size=512,
        max_iter=100,
        dataset_path=None,
        evaluate_on_test=False,
        name="BoWLR_sklearn"
):
    train_generator_tqdm, test_generator_tqdm, vocab = generate_ml_data(all_code_types, min_count, batch_size,
                                                                        verbose=verbose, dataset_path=dataset_path,
                                                                        split_paths=False)
    steps = [
        ('EmbedTransform', EmbedTransform(vocab,
                                          all_code_types,
                                          d_embedding,
                                          w2v_window,
                                          w2v_mincount,
                                          w2v_sampling_method,
                                          w2v_epochs) if w2v else OneHotEmbedTransform(vocab)),
        ('poolTransform', PoolTransform(
            pooling_funcs=[lambda x, dim: torch.max(x, dim=dim)[0], torch.mean, torch.sum] if w2v else [torch.sum])),
    ]
    pipeline = FeaturePipeline(steps)

    model = LogisticRegression(solver=solver, max_iter=max_iter, penalty=penalty, C=C, verbose=verbose)

    # Run model with 5k fold cross validation
    X, y = pipeline.transform(train_generator_tqdm)
    X = preprocessing.scale(X.numpy())

    skf = StratifiedKFold(n_splits=K_FOLDS)

    metrics = cross_validate(model, X=X, y=y.numpy(), cv=skf,
                             scoring=('roc_auc', 'f1_weighted', 'precision_weighted', 'recall_weighted',
                                      'average_precision'), return_estimator=True)

    # Get mean and std for metrics
    metric_names = ['test_roc_auc', 'test_f1_weighted', 'test_average_precision', 'fit_time']
    for m in metric_names:
        metrics.update({m + '_mean': float(np.mean(metrics[m]))})
        metrics.update({m + '_std': float(np.std(metrics[m]))})
        ex.log_scalar(m + '_mean', float(np.mean(metrics[m])))
        ex.log_scalar(m + '_std', float(np.std(metrics[m])))

    if evaluate_on_test:
        model = metrics['estimator'][0]
        save_pickle(model, os.path.join(MODEL_DIR, name) + '.dill')
        X, y = pipeline.transform(test_generator_tqdm)
        y_proba = model.decision_function(X)
        y_pred = model.predict(X)
        metrics['average_precision'] = average_precision_score(y, y_proba, average='weighted')
        metrics['auc'] = roc_auc_score(y, y_proba, average='weighted')

    return metrics


@ex.main
def main():
    return run()


if __name__ == '__main__':
    ex.run_commandline()

##### HyperOpt support #####################
from hyperopt import STATUS_OK


# noinspection PyUnresolvedReferences

def hyperopt_objective(params):
    config = {}

    if type(params) == dict:
        params = params.items()

    for (key, value) in params:
        if key in ['fc_dim']:
            value = int(value)
        config[key] = value
    run = ex.run(config_updates=config, )
    err = run.result
    return {'loss': 1 - np.mean(err['test_roc_auc_mean']), 'status': STATUS_OK}

##### End HyperOpt support #################
