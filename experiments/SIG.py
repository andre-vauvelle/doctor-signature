import logging
import numpy as np
from datetime import datetime
import os
import sys
import uuid

import torch
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, FeedForward
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.nn import Activation
from allennlp.training import Trainer, GradientDescentTrainer
from allennlp.training.util import evaluate
from allennlp.data.dataloader import BatchSampler, DataLoader

from experiments import reformat_metrics, init_augmentations, update_dims
from src.data import StratifiedKFold

from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from torch import nn, optim
from torch.utils.data import Subset

import pandas as pd

from src.data import generate_ml_data

from definitions import ROOT_DIR, TENSORBOARD_DIR, MODEL_DIR
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from sacred.utils import apply_backspaces_and_linefeeds
from src.models.train_model import BaseModel, SignatureEncoder, train_model
from src.features.augmentations import AddTime, LeadLag, CumulativeSum
from src.omni.functions import save_pickle, save_json

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)
# Set GLOBAL vars
LOCAL = os.environ.get('LOCAL')
TEST = False
if TEST:
    logger.info('TESTING')

base = os.path.basename(__file__)
base_dir = os.path.dirname(__file__)
experiment_name = os.path.splitext(base)[0]
save_at = os.path.join(base_dir, experiment_name)

ex = Experiment(experiment_name)
# ex.observers.append(FileStorageObserver(save_at))
my_url = 'bigtop:27017'  # Or <server-static-ip>:<port> if running on server
curr_db_name = experiment_name
ex.captured_out_filter = apply_backspaces_and_linefeeds

if LOCAL:
    ex.observers.append(FileStorageObserver(save_at))
    logger.info('saving at {}'.format(save_at))
else:
    ex.observers.append(MongoObserver(url=my_url,
                                      db_name=curr_db_name))
    logger.info('saving at {}'.format(curr_db_name))

K_FOLDS = 5


@ex.config
def config():
    """configation for SIG"""
    name = "SIG"
    version = -1
    all_code_types = True
    d_embedding = 32
    min_count = 5
    batch_size = 128
    verbose = True
    epochs = 20
    lr = 0.001
    wd = 0.00001
    sig_depth = 2
    logsig = True
    run_name = 'SIG'
    testing_subsample_size = None
    patience = 10
    feedforward_num_layers = 1
    feedforward_hidden_dims = 128
    feedforward_activations = 'relu'  # use Activation.by_name('linear')()
    feedforward_dropout = 0.4
    dataset_path = None

    embedding_dropout_p = 0.5
    split_paths = False

    # Augmentations
    add_time = False
    t_max = 999999999
    t_scale = 1
    use_timestamps = False
    leadlag = False

    training_proportion = 1
    evaluate_on_test = False
    tensorboard_log = False

    if TEST:
        epochs = 1
        sig_depth = 2
        batch_size = 256
        testing_subsample_size = 10000


def init_sig(vocab, d_embedding, embedding_dropout_p, sig_depth,
             logsig, all_code_types, feedforward_num_layers, feedforward_hidden_dims, feedforward_activations,
             feedforward_dropout, leadlag, add_time, t_max, t_scale, use_timestamps, split_paths):
    # Init feedward params
    feedforward_hidden_dims = [feedforward_hidden_dims] * feedforward_num_layers
    feedforward_activations = [Activation.by_name(feedforward_activations)()] * feedforward_num_layers
    feedforward_dropout = [feedforward_dropout] * feedforward_num_layers

    # Needed for final layer
    feedforward_num_layers += 1
    feedforward_hidden_dims.append(1)
    feedforward_activations.append(Activation.by_name('linear')())
    feedforward_dropout.append(0)

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size(),
                                embedding_dim=d_embedding)

    # Handle Augmentations
    augmentations = []
    if add_time:
        augmentations.append('add_time')
    if leadlag:
        augmentations.append('leadlag')
    d_embedding_updated = update_dims(augmentations, d_embedding)
    i_augmentations = init_augmentations(augmentations, use_timestamps=use_timestamps, t_max=t_max, t_scale=t_scale)

    # Embedder maps the input tokens to the appropriate embedding matrix
    word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": token_embedding})

    # Encoder takes path of (N, L, C) and encodes into state vector
    # encoder = BagOfEmbeddingsEncoder(embedding_dim=d_embedding)
    encoder: Seq2VecEncoder = SignatureEncoder(input_dim=d_embedding_updated,
                                               depth=sig_depth,
                                               logsig=logsig)

    classifier_feedforward: FeedForward = FeedForward(
        input_dim=encoder.get_output_dim() * 3 if (all_code_types and split_paths) else encoder.get_output_dim(),
        num_layers=feedforward_num_layers,
        hidden_dims=feedforward_hidden_dims,
        activations=feedforward_activations,
        dropout=feedforward_dropout
    )

    model = BaseModel(
        vocab,
        word_embeddings,
        encoder,
        classifier_feedforward,
        augmentations=i_augmentations,
        embedding_dropout_p=embedding_dropout_p
    )

    return model


@ex.capture()
def run(
        all_code_types,
        d_embedding,
        embedding_dropout_p,
        min_count,
        batch_size,
        verbose,
        epochs,
        lr,
        wd,
        logsig,
        sig_depth,
        run_name,
        patience,
        add_time,
        leadlag,
        t_scale,
        t_max,
        use_timestamps,
        feedforward_num_layers,
        feedforward_hidden_dims,
        feedforward_activations,
        feedforward_dropout,
        training_proportion=1,
        testing_subsample_size=None,
        split_paths=False,
        tensorboard_log=False,
        evaluate_on_test=True):
    """Run the experiment for either cross validation or testing"""

    dataset, dataset_test, vocab = generate_ml_data(all_code_types, min_count, batch_size,
                                                    verbose=verbose, allen_mode=True,
                                                    dataset_path=None,
                                                    training_proportion=training_proportion,
                                                    testing_subsample_size=testing_subsample_size,
                                                    split_paths=split_paths)

    logger.info("Using k-fold cross validation")
    # Allen kfold
    metrics_by_fold = []
    cross_validator = StratifiedKFold(n_splits=K_FOLDS, shuffle=True)

    n_splits = cross_validator.get_n_splits(dataset)

    for fold_index, (train_indices, validation_indices) in enumerate(cross_validator(dataset)):
        logger.info(f"Fold {fold_index}/{n_splits - 1}")
        train_dataset = Subset(dataset, train_indices, )
        validation_dataset = Subset(dataset, validation_indices)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True)
        if tensorboard_log or evaluate_on_test:
            serialization_dir = os.path.join(
                TENSORBOARD_DIR,
                run_name,
                str(uuid.uuid4()),
                str(fold_index))
        else:
            serialization_dir = None

        model = init_sig(vocab, d_embedding, embedding_dropout_p,
                         sig_depth,
                         logsig, all_code_types, feedforward_num_layers, feedforward_hidden_dims,
                         feedforward_activations,
                         feedforward_dropout, leadlag, add_time, t_max, t_scale,
                         use_timestamps,
                         split_paths)
        if torch.cuda.is_available():
            cuda_device = 0
            model = model.cuda(cuda_device)
            logger.info('USING CUDA GPU')
        else:
            cuda_device = -1

        fold_metrics, model = train_model(model, lr, wd, train_loader, validation_loader, patience, epochs, cuda_device,
                                          serialization_dir)
        if serialization_dir is not None:
            ex.add_artifact(os.path.join(serialization_dir, 'best.th'))  # Add file location to sacred log

        metrics_by_fold.append(fold_metrics)

        if evaluate_on_test:
            if serialization_dir is None:
                raise Exception('serialization_dir needed to load best model from validation')
            test_dataloader = DataLoader(dataset=dataset_test, batch_size=batch_size,
                                         shuffle=True)  # Held out test data
            metrics = evaluate(model, test_dataloader, cuda_device)
            return metrics
        torch.cuda.empty_cache()

    metrics = reformat_metrics(metrics_by_fold, ex)
    return metrics


@ex.main
def main():
    return run()


if __name__ == '__main__':
    ex.run_commandline()

##### HyperOpt support #####################
from hyperopt import STATUS_OK, STATUS_FAIL


# noinspection PyUnresolvedReferences

def hyperopt_objective(params):
    config = {}

    try:
        if type(params) == dict:
            params = params.items()

        for (key, value) in params:
            if key in ['fc_dim']:
                value = int(value)
            config[key] = value
        run = ex.run(config_updates=config, )
        err = run.result

        if config['evaluate_on_test']:
            result = {'loss': 1, 'status': STATUS_OK}
            result.update(err)
            return result
        return {'loss': 1 - np.mean(err['best_validation_auc']), 'status': STATUS_OK}
    except Exception as e:
        return {'status': STATUS_FAIL,
                'exception': str(e)}

##### End HyperOpt support #################
