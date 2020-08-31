import logging
import sys
from typing import Dict, List, Union
import torch
import signatory

import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from allennlp.common.util import gpu_memory_mb
from allennlp.modules import FeedForward
from allennlp.training.metrics import BooleanAccuracy
from allennlp.training.trainer import GradientDescentTrainer
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder

from src.features.augmentations import AddTime, LeadLag, CumulativeSum
from src.models.custom_metrics import F1Sklearn, AveragePrecisionSklearn, AucSklearn

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

TERMS_LIMIT = 1_000_000


class BaseModel(Model):
    """
    Base structure for all NN models
    """

    def __init__(self,
                 vocab,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 augmentations: List[Union[AddTime, LeadLag, CumulativeSum]] = [],
                 embedding_dropout_p=0.4):
        """

        :param vocab: from allennlp vocab class
        :param embedding: a dictionary of codes and pretrained embeddings
        """
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.augmentations = augmentations
        self.encoder = encoder
        self.classifier_feedforward = classifier_feedforward
        self.loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        # The embedder maps the input tokens to the appropriate embedding matrix
        self.embedding_dropout_p = embedding_dropout_p
        self.metrics = {'accuracy': BooleanAccuracy(),
                        'auc': AucSklearn(),
                        'f1_score': F1Sklearn(),
                        'average_precision': AveragePrecisionSklearn()}

    def channel_dropout(self, embeddings):
        """
        Input embeddings of shape [N,L,C] where N is the number of batches and L is the length of the path and C is
        the number of channels.

        Masking is consistent within each batch

        Dropout2d is used which takes in tensor of dim3 but applies dropout to dim 1.

        SO we:
        * reshape to [N,C,L]
        * get the mask
        * reshape back to [N,L,C]

        """
        embeddings = embeddings.permute(0, 2, 1)
        embeddings = f.dropout2d(embeddings, self.embedding_dropout_p, training=self.training)
        embeddings = embeddings.permute(0, 2, 1)
        return embeddings

    # def get_embedding_path(self,sequence: Dict[str, torch.Tensor]) -> torch.Tensor:

    def get_state(self, sequence: Dict[str, torch.Tensor], timestamps: torch.Tensor = None) -> torch.Tensor:
        """Takes in a sequence and returns the state after applying word embeddings and augmentations"""
        embeddings = self.word_embeddings(sequence)
        mask = get_text_field_mask(sequence)
        for augmentation in self.augmentations:
            sequence.update({'embedded': embeddings})
            if str(type(augmentation)) == "<class 'src.features.augmentations.AddTime'>":
                embeddings = augmentation.transform(sequence, timestamps)
            else:
                embeddings = augmentation.transform(sequence)
        if self.embedding_dropout_p != 0:
            embeddings = self.channel_dropout(embeddings)
        state = self.encoder(embeddings, mask)
        return state

    def forward(self,
                sequence_diag_icd10: Dict[str, torch.Tensor], eid,
                label: torch.Tensor = None,
                sequence_secondary_icd10: Dict[str, torch.Tensor] = None,
                sequence_oper4: Dict[str, torch.Tensor] = None,
                timestamps_diag_icd10: torch.Tensor = None,
                timestamps_secondary_icd10: torch.Tensor = None,
                timestamps_oper4: Dict[str, torch.Tensor] = None
                ):

        states = []

        state_PRIMDX = self.get_state(sequence_diag_icd10, timestamps_diag_icd10)
        states.append(state_PRIMDX)

        class_logits = self.classifier_feedforward(torch.cat(states, dim=-1))

        output = {"class_logits": class_logits}
        out = self.sigmoid(class_logits)

        self.metrics['accuracy'](out.round(), label.unsqueeze(dim=1).to(dtype=torch.float32))
        self.metrics['auc'](out.squeeze(dim=-1), label.to(dtype=torch.float32))
        self.metrics['f1_score'](out.squeeze(dim=-1).round(), label.to(dtype=torch.float32))
        self.metrics['average_precision'](out.squeeze(dim=-1), label.to(dtype=torch.float32))
        output["loss"] = self.loss(class_logits, label.unsqueeze(dim=1).to(dtype=torch.float32))
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {'accuracy': self.metrics['accuracy'].get_metric(reset=reset)}
        metrics.update({
            'average_precision': self.metrics['average_precision'].get_metric(reset=reset),
            'f1': self.metrics['f1_score'].get_metric(reset=reset),
            'auc': self.metrics['auc'].get_metric(reset=reset)})
        for (gpu_num, memory) in gpu_memory_mb().items():
            metrics.update({'gpu_batch_' + str(gpu_num) + '_memory_MB': memory})

        return metrics


class SignatureEncoder(Seq2VecEncoder):
    """
    Input shape: ``(batch_size, sequence_length, input_dim)``; output shape:
    ``(batch_size, output_dim)``.

    """

    def __init__(self, input_dim, depth, logsig, basepoint=True):
        super().__init__()

        self.basepoint = basepoint
        self.depth = depth
        self.logsig = logsig
        self.sig_func = signatory.logsignature if self.logsig else signatory.signature
        self.terms = signatory.logsignature_channels(
            input_dim, self.depth) if self.logsig else signatory.signature_channels(
            input_dim, self.depth)
        if self.terms >= TERMS_LIMIT:
            raise ImportError("Number of signature terms {} is greater than {}..".format(self.terms, TERMS_LIMIT))

    def forward(self, input, mask):
        """mask required but not used"""
        signature_terms = self.sig_func(path=input, depth=self.depth, basepoint=self.basepoint)
        output = signature_terms
        return output

    def get_input_dim(self) -> int:
        """
        Returns the dimension of the vector input for each element in the sequence input
        to a `Seq2VecEncoder`. This is `not` the shape of the input tensor, but the
        last element of that shape.
        """
        return self.input_dim

    def get_output_dim(self) -> int:
        """
        Returns the dimension of the final vector output by this `Seq2VecEncoder`.  This is `not`
        the shape of the returned tensor, but the last element of that shape.
        """
        return self.terms


def train_model(model, lr, wd, train_loader, validation_loader, patience, epochs, cuda_device, serialization_dir):
    """Train an initialized model"""

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    trainer = GradientDescentTrainer(model=model,
                                     data_loader=train_loader,
                                     validation_data_loader=validation_loader,
                                     optimizer=optimizer,
                                     patience=patience,
                                     num_epochs=epochs,
                                     cuda_device=cuda_device,
                                     serialization_dir=serialization_dir)

    fold_metrics = trainer.train()
    # Save embedding weights for visualization
    # n = word_embeddings.token_embedder_tokens.weight.item()
    # pd.DataFrame(n).to_csv(os.path.join(TENSORBOARD_DIR, run_name, 'model_weights.tsv'),
    #                       header=None, index=None, sep='\t')

    return fold_metrics, model
