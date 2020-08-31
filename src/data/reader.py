import logging
import os
from typing import Iterator, List, Dict, Optional, Callable
import torch
import json
import numpy as np
from overrides import overrides
from allennlp.data import Instance
from allennlp.data.fields import TextField, ArrayField, LabelField, MetadataField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

from src.omni.functions import load_pickle
import itertools

from definitions import DATA_DIR

USE_GPU = torch.cuda.is_available()

logger = logging.getLogger(__name__)


class EHRDatasetReader(DatasetReader):
    """
    DatasetReader for EHR data stored in dill files by get_sequences
    """

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None,
                 max_sentence_length=100, required_code_types=('diag_icd10',),
                 testing_subsample_size=None, split_paths=True):
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_sentence_length = max_sentence_length
        self.required_code_types = required_code_types
        # self.tokenize_and_preprocess = tokenize_and_preprocess
        self.testing_subsample_size = testing_subsample_size
        self.split_paths = split_paths

    @overrides
    def text_to_instance(self, code_tokens: Dict[str, List[Token]], code_timestamps: Dict[str, np.ndarray],
                         label: str, eid: Optional[int] = None) -> Instance:

        label_field = LabelField(label=label)
        fields = {"label": label_field}

        if self.split_paths:
            for code_type, tokens in code_tokens.items():
                token_field = TextField(tokens, self.token_indexers)
                fields['sequence_' + code_type] = token_field
            for code_type, timestamps in code_timestamps.items():
                timestamps_field = ArrayField(timestamps)
                fields['timestamps_' + code_type] = timestamps_field
        else:
            # flatten
            timestamps = np.concatenate(list(code_timestamps.values()))
            tokens = sum(code_tokens.values(), [])  # slight hack to get flat list

            token_field = TextField(tokens, self.token_indexers)
            timestamps_field = ArrayField(np.array(timestamps))

            # TODO: using these keys here but should be more general. Will need to change split paths
            fields['sequence_diag_icd10'] = token_field
            fields['timestamps_diag_icd10'] = timestamps_field

        # if eid:
        eid_field = MetadataField(metadata=eid)  # Used for debugging etc
        fields["eid"] = eid_field

        return Instance(fields)

    @staticmethod
    def _tokenize_and_preprocess(sequence):
        # TODO: add bias token?
        # sequence = [code if type(code) == str else '<MISSING>' for code in sequence]
        return sequence

    @overrides
    def _read(self, data_set_path: str) -> Iterator[Instance]:

        while not os.path.exists(os.path.join(DATA_DIR, data_set_path)):
            cont = input(
                'Can\'t find file {}. Do you want to try again? y/n'.format(
                    DATA_DIR + data_set_path))
            if cont == 'n':
                raise FileNotFoundError
        data_set = load_pickle(os.path.join(DATA_DIR, data_set_path), use_dill=True)

        sequences = data_set['x']
        labels = data_set['y']
        timestamp_sequences = data_set['t']
        eid_list = data_set['eid']

        if self.testing_subsample_size is not None:
            # TODO: update to allow for data ablation
            # Sample
            sequences_sample = sequences[:self.testing_subsample_size]
            # Get the unique sequences
            stringed_dicts = [json.dumps(d) for d in sequences_sample]
            unique_dicts = set(stringed_dicts)
            unique_indexes = [stringed_dicts.index(u) for u in unique_dicts]
            sequences = [sequences[i] for i in unique_indexes]
            labels = [labels[i] for i in unique_indexes]
            timestamp_sequences = [timestamp_sequences[i] for i in unique_indexes]
            eid_list = [eid_list[i] for i in unique_indexes]
            logger.info(
                '{} Subsampled but only {} unique \n new class balence {}'.format(len(stringed_dicts), len(sequences),
                                                                                  sum([l == 'True' for l in
                                                                                       labels]) / len(labels)))

        for eid, sequence, timestamps, label in zip(eid_list, sequences, timestamp_sequences, labels):
            missing_code_types = set(self.required_code_types) - set(sequence.keys())
            for code_type in missing_code_types:
                sequence.update({code_type: ['<MISSING>', ]})
                timestamps.update({code_type: [0, ]})

            for code_type, codes in sequence.items():
                tokens = [Token(code) for code in self._tokenize_and_preprocess(codes)][:self.max_sentence_length]
                sequence.update({code_type: tokens})
            for code_type, code_timestamp in timestamps.items():
                code_timestamp = np.array(code_timestamp)[:self.max_sentence_length]
                timestamps.update({code_type: code_timestamp})

            yield self.text_to_instance(sequence, timestamps, label, eid)

