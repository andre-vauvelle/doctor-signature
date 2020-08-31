import os
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from allennlp.data.vocabulary import Vocabulary

from definitions import DATA_DIR, MODEL_DIR
from src.omni.functions import load_pickle, save_pickle

import multiprocessing

THREAD_COUNT = multiprocessing.cpu_count()


def normalise_standardised(x: torch.Tensor, dim=1):
    '''
    Normalise a pytorch tensor along a particular dimention
    :param x: the tensor to be normalised
    :param dim: the dim to do it along
    :return:
    '''
    x_mean = x.mean(dim=dim)
    x_std = x.std(dim=dim)
    return (x - x_mean) / x_std


def normalise_rescale(x: torch, dim=1):
    """
    Normalise a pytorch tensor along a particular dimention
    :param x: the tensor to be normalised
    :param dim: the dim to do it along
    :return:
    """
    x_max = torch.max(x, dim=dim)[0]  # min and max return tuple with (values, indices)
    x_min = torch.min(x, dim=dim)[0]
    return (x - x_max) / (x_max - x_min)


def get_wordvectors(vocab,
                    data_set_path='processed/train_valid_test/train.dill',
                    size=100,
                    window=5,
                    min_count=5,
                    sg=0,
                    epochs=50,
                    init_only=False,
                    **kwargs) -> np.ndarray:
    """
    Rapid gensim implementation of word2vec
    :param vocab:
    :param data_set_path:
    :param size:
    :param window:
    :param min_count:
    :param workers:
    :param sg: 0 for CBOW and 1 for SG
    :param init_only: if True, we won't run word vectors rather just get xiavier init
    :return: numpy array of values with dimention (vocab.size x size)
    """
    if init_only:
        embedding = torch.empty((vocab.get_vocab_size(), size))
        return torch.nn.init.xavier_uniform_(embedding).numpy()

    # Check if it has been run already
    cache_path = os.path.join(
        MODEL_DIR,
        'embeddings',
        '{size}_{window}_{sg}_{data_set_path}_epochs_{epochs}.dill'.format_map(
            {
                'size': size,
                'window': window,
                'sg': sg,
                'data_set_path': '_'.join(data_set_path.replace('.dill', '').split('/')),
                'epochs': epochs
            }))
    if os.path.exists(cache_path):
        print('Using cached embeddings {}'.format(cache_path))
        word_vectors = load_pickle(cache_path, use_dill=True)
    else:
        print('No embedding cache found {}. Training from scratch.'.format(cache_path))
        data_set = load_pickle(DATA_DIR + data_set_path, use_dill=True)
        sequences = data_set['x']
        # time_sequences = data_set['t']

        model = Word2Vec(size=size,
                         window=window,
                         min_count=min_count,
                         workers=THREAD_COUNT,
                         sg=sg)
        model.build_vocab(sentences=sequences)
        model.train(sequences, total_examples=model.corpus_count, epochs=epochs)
        word_vectors = pd.Series(dict(zip(model.wv.vocab.keys(), model.wv.vectors)))
        save_pickle(word_vectors, cache_path)

    # Fill embedding matrix in the correct order
    # TODO: should I use xaiver here?
    correct_order = pd.Series(vocab.get_token_to_index_vocabulary())
    missing_tokens = set(correct_order.index) - set(word_vectors.index)
    random_missing = np.random.normal(loc=0, scale=1, size=(len(missing_tokens), size))
    word_vectors_needed = word_vectors[word_vectors.index.isin(correct_order.index)]
    correct_order.loc[word_vectors_needed.index] = word_vectors_needed
    correct_order.loc[missing_tokens] = list(random_missing)

    embeddings = np.stack(correct_order.values)
    return embeddings


if __name__ == '__main__':
    from src.models.train_model import EHRDatasetReader

    batch_size = 128
    min_count = 1
    max_vocab_size = 10000
    solver = "lbfgs"
    max_iter = 10_0000
    penalty = 'l2'
    C = 1.0
    w2v = True
    d_embedding = 10
    w2v_window = 5
    w2v_mincount = 5
    w2v_sampling_method = 0
    signatures = False
    sig_depth = 3
    verbose = True
    dataset_path = 'processed/train_valid_test_c4/'

    reader = EHRDatasetReader()
    train_dataset = reader.read(os.path.join(dataset_path, "train.dill"))
    validation_dataset = reader.read(os.path.join(dataset_path, "train.dill"))

    # Only use the train set to learn an oov token
    vocab = Vocabulary.from_instances(
        train_dataset + validation_dataset,
        max_vocab_size=max_vocab_size,
        min_count={"tokens": min_count},
    )

    word_vectors = get_wordvectors(
        vocab,
        data_set_path="processed/train_embeddings/train_embeddings_full.dill",
        size=d_embedding,
        window=w2v_window,
        min_count=w2v_mincount,
        workers=3,
        sg=w2v_sampling_method,
    )
