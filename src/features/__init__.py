from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import torch


class Transformer(ABC):
    @abstractmethod
    def transform(self, sequence: Dict['str', torch.Tensor]) -> torch.Tensor:
        pass

    @abstractmethod
    def generate(self, batch_group: Dict[str, Dict]) -> Dict[str, Dict]:
        return batch_group
