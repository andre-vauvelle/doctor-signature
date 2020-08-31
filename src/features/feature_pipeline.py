import os
import torch
from typing import Iterator, Dict, List, Tuple

from allennlp.data.vocabulary import Vocabulary
# from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.common.tqdm import Tqdm
from allennlp.nn.util import get_text_field_mask

from sklearn.preprocessing import StandardScaler

from src.features import Transformer
from src.features.event_embedding_features import get_wordvectors
from src.features.signatures.compute import SignatureTransform
from src.data.reader import EHRDatasetReader


class EmbedTransform(Transformer):
    """
    Embeds all sequences from a list of N arrays of tokens length L to path of shape [N, L, C].

    C is the dimension of the applied embedding array [C_one, C] where C_one the number of unique tokens.
    """

    def __init__(self, vocab, all_code_types, d_embedding, w2v_window, w2v_mincount, w2v_sampling_method, epochs,
                 w2v_datapath=None):

        if all_code_types:
            w2v_datapath = os.path.join('processed', 'train_embeddings',
                                        'train_embeddings_full_diag_icd10_secondary_icd10_oper4.dill') if w2v_datapath is None else w2v_datapath
        else:
            w2v_datapath = os.path.join('processed', 'train_embeddings',
                                        'train_embeddings_full_diag_icd10.dill') if w2v_datapath is None else w2v_datapath

        self.embedding = get_wordvectors(
            vocab,
            data_set_path=w2v_datapath,
            size=d_embedding,
            window=w2v_window,
            min_count=w2v_mincount,
            sg=w2v_sampling_method,
            epochs=epochs
        )

    def transform(self, sequence):
        tokens = sequence['tokens']['tokens']
        active_tokens_mask = get_text_field_mask(sequence)

        embedding_t = torch.from_numpy(self.embedding)
        embedded = embedding_t[tokens]  # Select the correct vector for each token
        embedded = torch.einsum('nlc, nl -> nlc', embedded, active_tokens_mask)
        return embedded

    def generate(self, batch_group: Dict[str, Dict]) -> Dict[str, Dict]:
        for name, sequence in batch_group.items():
            if name.startswith('sequence'):
                sequence = batch_group[name]
                embedded = self.transform(sequence)
                batch_group[name].update({'embedded': embedded})

        return batch_group


class OneHotEmbedTransform(Transformer):
    """
    One hot embeddings fairly self explanatory
    """

    def __init__(self, vocab):
        self.vocab = vocab

    def transform(self, sequence: Dict['str', torch.Tensor]) -> torch.Tensor:
        tokens = sequence['tokens']['tokens']
        active_tokens_mask = get_text_field_mask(sequence)
        one_hot = torch.eye(self.vocab.get_vocab_size())
        embedded = torch.einsum(
            "ijk, ij -> ijk", one_hot[tokens], active_tokens_mask
        )  # apply mask
        return embedded

    def generate(self, batch_group: Dict[str, Dict]) -> Dict[str, Dict]:
        for name, sequence in batch_group.items():
            if name.startswith('sequence'):
                sequence = batch_group[name]
                embedded = self.transform(sequence)
                batch_group[name].update({'embedded': embedded})
        return batch_group


class PoolTransform(Transformer):
    """
    Applies a list of torch operators over the length of the path and concatenates them.

    Requires 'embedded' to have been added to batch_group['sequence']

    Embedded is a path of shape [N, L, C]
    """

    def __init__(self, pooling_funcs: list):
        self.pooling_funcs = pooling_funcs

    def transform(self, sequence) -> torch.Tensor:
        embedded = sequence['embedded']
        pooled_embedded = [f(embedded, dim=1) for f in self.pooling_funcs]
        features = torch.cat(pooled_embedded, dim=-1)
        return features

    def generate(self, batch_group: Dict[str, Dict]) -> Dict[str, Dict]:
        for name, sequence in batch_group.items():
            if name.startswith('sequence'):
                features = self.transform(sequence)
                batch_group[name].update({'features': features})

        return batch_group


class FeaturePipeline:
    def __init__(self, steps: List[Tuple[str, type(Transformer)]], scale: bool = False):
        """
        Args:
            steps: List of tuples where each tuple element of the list is of the form (name, Transformer).
                   'name' is the name of the transformation step
                   'transformer' is a class containing a Transformer.
            scale: A boolean which scales the final
        """
        self.steps = steps
        self.scale = scale

    def transform(self, generator_tqdm: Iterator):
        """

        :param generator_tqdm: A generator which is will return batches of token sequences
        :return:
        """
        X_store = []
        y_store = []
        for batch_group in generator_tqdm:
            # Do the first transform first
            name, transformer = self.steps[0]
            output_group = transformer.generate(batch_group)
            # Then loop through every other transform in the pipeline series
            for name, transformer in self.steps[1:]:
                output_group = transformer.generate(output_group)

            patient_features_store = []
            for name, feature in output_group.items():
                if name.startswith('sequence'):
                    patient_features_store.append(output_group[name]['features'])
            # feature tensors have rows as patients columns as features
            patient_features = torch.cat(patient_features_store, axis=1)  # cat each feature for one patient

            X_store.append(patient_features)
            y_store.append(output_group['label'])

        X = torch.cat(X_store)

        if self.scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        y = torch.cat(y_store)
        return X, y
