from typing import Dict

import torch

from src.features import Transformer
import signatory

TERMS_LIMIT = 2_000_000


class SignatureTransform(Transformer):
    def __init__(self, depth, logsig=False, basepoint=True):
        self.depth = depth
        self.logsig = logsig
        self.basepoint = basepoint

    def transform(self, sequence) -> torch.Tensor:
        embedded = sequence['embedded']

        sig_func = signatory.logsignature if self.logsig else signatory.signature

        terms = signatory.logsignature_channels(
            embedded.shape[-1], self.depth) if self.logsig else signatory.signature_channels(
            embedded.shape[-1], self.depth)
        if terms >= TERMS_LIMIT:
            raise ImportError("Number of signature terms is greater than {}..".format(TERMS_LIMIT))

        signature_terms = sig_func(path=embedded, depth=self.depth, basepoint=self.basepoint)
        return signature_terms

    def generate(self, batch_group: Dict[str, Dict]) -> Dict[str, Dict]:
        for name, sequence in batch_group.items():
            if name.startswith('sequence'):
                sequence = batch_group[name]
                signature_terms = self.transform(sequence)
                batch_group[name].update({'features': signature_terms})
        return batch_group
