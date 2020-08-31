import copy
import math
import os
import random

import allennlp
import torch
from typing import Iterator, Dict, List, Tuple

from allennlp.training import Trainer
from torch.utils.data import Subset, Dataset

from numpy.random.mtrand import RandomState
from allennlp.common import util as common_util

from allennlp.data.vocabulary import Vocabulary

from allennlp.common.tqdm import Tqdm
from allennlp.nn.util import get_text_field_mask
from allennlp.common import Registrable
from overrides import overrides
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import BaseCrossValidator

from src.features import Transformer
from src.features.event_embedding_features import get_wordvectors
from src.features.signatures.compute import SignatureTransform
from src.data.reader import EHRDatasetReader
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union
from allennlp.data import Instance, Vocabulary, DatasetReader, DataLoader
import logging

logger = logging.getLogger(__name__)


# Taking a lot from unfinished https://github.com/allenai/allennlp/blob/e1723d4cd5514064d56783a2137b12c286b613bb/allennlp/training/cross_validation.py
class CrossValidator(Registrable, BaseCrossValidator):
    default_implementation = "k_fold"

    def __call__(
            self, instances: Sequence[Instance]
    ) -> Iterator[Tuple[Sequence[int], Sequence[int]]]:
        groups, labels = self._labels_groups(instances)
        return super().split(instances, labels, groups=groups)

    @overrides
    def split(self, instances: Sequence[Instance]) -> Iterator[Tuple[Sequence[int], Sequence[int]]]:
        return self(instances)

    @overrides
    def get_n_splits(self, instances: Sequence[Instance]) -> int:
        groups, labels = self._labels_groups(instances)
        return super().get_n_splits(instances, labels, groups=groups)

    @staticmethod
    def _labels_groups(
            instances: Sequence[Instance],
    ) -> Tuple[Optional[Sequence[Union[str, int]]], Optional[Sequence[Any]]]:
        labels = [1 if instance["label"].label == 'True' else 0 for instance in instances] if "label" in instances[
            0] else None
        groups = (
            [instance["_group"] for instance in instances] if "_group" in instances[0] else None
        )
        return groups, labels


@CrossValidator.register("stratified_k_fold")
class StratifiedKFold(CrossValidator, sklearn.model_selection.StratifiedKFold):
    def __init__(
            self,
            n_splits: int = 5,
            shuffle: bool = False,
            random_state: Optional[Union[int, RandomState]] = None,
    ) -> None:
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def generate_ml_data(all_code_types: bool, min_count, batch_size, n_epochs=1,
                     dataset_path=None, max_vocab_size=99_999, allen_mode=False, training_proportion=1,
                     testing_subsample_size=None,
                     split_paths=True, max_sentence_length=100, verbose=False):
    """
    Get a generator that will yield nicely tokenised data and a vocab to match!

    :param all_code_types: if true we will include PRIMDX_SECDX_PROC data if false just PRIMDX
    :param min_count: Minimum count of a code that should be given its own token
    :param batch_size: What is the batch size to process our features, higher and we get done faster but more memory...
    :param dataset_path: If not none will override the defaults set by all_code_types
    :param max_vocab_size: maxmium size of vocab. This should not be an issue unless something broke so set very high
    :param split_paths: this will split the paths into each code type
    :param max_sentence_length: cap on the max number of codes in a sequence
    :param verbose:
    :return:
    """
    if all_code_types:
        code_types = ('diag_icd10', 'oper4', 'secondary_icd10')
        dataset_path = os.path.join('processed',
                                    'train_test_PRIMDX_SECDX_PROC') if dataset_path is None else dataset_path
    else:
        code_types = ('diag_icd10',)
        dataset_path = os.path.join('processed', 'train_test_PRIMDX') if dataset_path is None else dataset_path

    reader = EHRDatasetReader(required_code_types=code_types, testing_subsample_size=testing_subsample_size,
                              split_paths=split_paths, max_sentence_length=max_sentence_length)
    dataset = reader.read(os.path.join(dataset_path, "train.dill"))
    dataset_test = reader.read(os.path.join(dataset_path, "test.dill"))

    # Only use the train set to learn an oov token
    vocab = Vocabulary.from_instances(
        dataset,
        max_vocab_size=max_vocab_size,
        min_count={"tokens": min_count},
    )

    # Index both training and test with only training vocab. Testing vocab unseen at prediction time
    dataset.index_with(vocab)
    dataset_test.index_with(vocab)

    if training_proportion < 1:
        data_size = len(dataset)
        sample_idx = random.sample(range(int(data_size)), int(data_size * training_proportion))
        dataset = Subset(dataset, sample_idx)

    if allen_mode:
        return dataset, dataset_test, vocab

    # Get tqdm for the training batches
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, )
    batch_generator = iter(data_loader)
    generator_tqdm = Tqdm.tqdm(batch_generator, total=len(data_loader))

    # For test data
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True, )
    batch_generator_test = iter(data_loader_test)
    generator_tqdm_test = Tqdm.tqdm(batch_generator_test, total=len(data_loader_test))

    return generator_tqdm, generator_tqdm_test, vocab
