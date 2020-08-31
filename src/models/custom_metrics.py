import torch
from sklearn import metrics

from allennlp.training.metrics import Auc


class AucSklearn(Auc):
    """
    The AUC Metric measures the area under the receiver-operating characteristic
    (ROC) curve for binary classification problems.
    """

    def __init__(self, average='weighted', positive_label=1):
        super().__init__()
        self.average = average
        self._positive_label = positive_label
        self._all_predictions = torch.FloatTensor()
        self._all_gold_labels = torch.LongTensor()

    def get_metric(self, reset: bool = False):
        if self._all_gold_labels.shape[0] == 0:
            return 0.5
        auc = metrics.roc_auc_score(self._all_gold_labels.cpu().numpy(),
                                    self._all_predictions.cpu().numpy(),
                                    average=self.average)
        if reset:
            self.reset()
        return auc


class F1Sklearn(Auc):
    """
    Using sklearn to calculate F1
    """

    def __init__(self, average='weighted', positive_label=1):
        super(F1Sklearn, self).__init__()
        self.average = average
        self._positive_label = positive_label
        self._all_predictions = torch.FloatTensor()
        self._all_gold_labels = torch.LongTensor()

    def get_metric(self, reset: bool = False):
        f1_score = metrics.f1_score(self._all_gold_labels.cpu().numpy(),
                                    self._all_predictions.cpu().numpy(),
                                    pos_label=self._positive_label,
                                    average=self.average)
        if reset:
            self.reset()
        return f1_score


class AveragePrecisionSklearn(Auc):
    """
    Using sklearn to calculate F1
    """

    def __init__(self, average='weighted', positive_label=1):
        super(AveragePrecisionSklearn, self).__init__()
        self.average = average
        self._positive_label = positive_label
        self._all_predictions = torch.FloatTensor()
        self._all_gold_labels = torch.LongTensor()

    def get_metric(self, reset: bool = False):
        average_precision = metrics.average_precision_score(self._all_gold_labels.cpu().numpy(),
                                                            self._all_predictions.cpu().numpy(),
                                                            pos_label=self._positive_label,
                                                            average=self.average)
        if reset:
            self.reset()
        return average_precision
