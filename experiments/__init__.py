import os
from itertools import product

from hyperopt.mongoexp import MongoTrials
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

from src.data import generate_ml_data
from src.features.augmentations import LeadLag, CumulativeSum, AddTime
from src.features.feature_pipeline import FeaturePipeline

import numpy as np

from hyperopt import STATUS_OK, STATUS_FAIL, Trials, fmin, tpe
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


def reformat_metrics(metrics_by_fold, ex):
    # Convert metrics into dict of lists from list of dicts
    metrics = {k: [dic[k] for dic in metrics_by_fold] for k in metrics_by_fold[0]}
    # Convert numpy types to natives
    for k, values in metrics.items():
        if type(values[0]).__module__ == np.__name__:
            metrics[k] = [float(v) for v in metrics[k]]

    # Get mean and std for metrics
    metric_names = ['best_validation_auc', 'best_validation_average_precision', 'best_validation_f1']
    for m in metric_names:
        metrics.update({m + '_mean': float(np.mean(metrics[m]))})
        metrics.update({m + '_std': float(np.std(metrics[m]))})
        # Get mean and std for metrics
        ex.log_scalar(m + '_mean', float(np.mean(metrics[m])))
        ex.log_scalar(m + '_std', float(np.std(metrics[m])))

    return metrics


def init_augmentations(augmentations, use_timestamps=False, t_max=999999999, t_scale=1):
    i_augmentations = []
    for a in augmentations:
        if a == 'add_time':
            i_augmentations.append(AddTime(t_max, t_scale, use_timestamps=use_timestamps))
        if a == 'leadlag':
            i_augmentations.append(LeadLag())
        if a == 'cumsum':
            i_augmentations.append(CumulativeSum())
    return i_augmentations


def update_dims(augmentations, d_embedding):
    d_embedding_updated = d_embedding
    for augmentation in augmentations:
        if augmentation == 'leadlag':
            d_embedding_updated = d_embedding_updated * 2
        if augmentation == 'add_time':
            d_embedding_updated += 1
    return d_embedding_updated


# All possible ordering including empty set
AUGMENTATIONS_OPTIONS = ['leadlag', 'cumsum']
_ALL_AUGMENTATION_OPTIONS = list(
    product(product(product(AUGMENTATIONS_OPTIONS), AUGMENTATIONS_OPTIONS), AUGMENTATIONS_OPTIONS))
ALL_AUGMENTATION_OPTIONS = [list(dict.fromkeys([a, b, c]).keys()) for ((a,), b), c in _ALL_AUGMENTATION_OPTIONS]
ALL_AUGMENTATION_OPTIONS.append([])
