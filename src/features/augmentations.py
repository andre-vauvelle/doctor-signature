import torch
from typing import Dict
from src.features import Transformer


# TODO: add Pen off?

class AddTime(Transformer):
    """Add time component to an embedding
    For a path of shape [N, L, C] this adds a time channel to be placed at the first index so that C <- C + 1.

    Parameterised by function if use timestamps is True. (Applied element wise)
    math::
        f(\Delta T)=
        \begin{cases}
        T_{scale} log(\Delta T + 1),   & \text{if} f(\Delta T) \leq T_{max}\
        T_max,                     & \text{otherwise}
        \end{cases}

    Otherwise a time features evenly spaced across [0,1] is used
    """

    def __init__(self, t_max=99999, t_scale=1, use_timestamps=False):
        """

        :param t_max: T_max hyper param cap on max time diff
        :param t_scale: Scale hyperparam
        :param use_timestamps: if disabled use standard [0,1] scaling without time stamps
        """
        self.use_timestamps = use_timestamps
        self.t_max = t_max
        self.t_scale = t_scale

    def transform(self, sequence, timestamp_sequence=None):

        embedded = sequence['embedded']
        N, L = embedded.shape[0], embedded.shape[1]
        if self.use_timestamps:
            timestamps = timestamp_sequence
            time_feature = self.time_scaling(timestamps, self.t_max, self.t_scale).unsqueeze(-1)
        else:
            # Time scaled to 0, 1
            time_feature = torch.linspace(0, 1, L).repeat(N, 1).view(N, L, 1)

        embedded = torch.cat((time_feature.to(dtype=embedded.dtype, device=embedded.device), embedded), -1)
        return embedded

    def generate(self, batch_group: Dict[str, Dict]) -> Dict[str, Dict]:
        # Batch and length dim
        for name, sequence in batch_group.items():
            if name.startswith('sequence'):
                sequence = batch_group[name]
                if self.use_timestamps:
                    time_name = name.replace('sequence', 'timestamps')
                    timestamp_sequence = batch_group[time_name]
                else:
                    timestamp_sequence = None
                embedded = self.transform(sequence, timestamp_sequence)
                batch_group[name].update({'embedded': embedded})
        return batch_group

    @staticmethod
    def time_scaling(timestamps, t_max, t_scale):
        time_feature = t_scale * torch.log(timestamps + 1)  # +1 allows 0 time difference...
        # zero_time_difference_ind = torch.isinf(time_feature)
        return time_feature.clamp(max=t_max)


class LeadLag(Transformer):
    """Applies the leadlag transformation to each path.
    doubles the number of channels with a lead and lag set of channels
    Example:
            [1, 2, 3] -> [[1, 1], [2, 1], [2, 2], [3, 2], [3, 3]]
    """

    def transform(self, sequence) -> torch.Tensor:
        embedded = sequence['embedded']
        embedded_repeat = embedded.repeat_interleave(2, dim=1)
        # Split out lead and lag
        lead = embedded_repeat[:, 1:, :]
        lag = embedded_repeat[:, :-1, :]

        # Combine
        embedded_leadlag = torch.cat((lead, lag), 2)
        return embedded_leadlag

    def generate(self, batch_group: Dict[str, Dict]) -> Dict[str, Dict]:
        # Interleave
        for name, sequence in batch_group.items():
            if name.startswith('sequence'):
                sequence = batch_group[name]
                embedded_leadlag = self.transform(sequence)
                batch_group[name].update({'embedded': embedded_leadlag})
        return batch_group


class CumulativeSum(Transformer):
    """Cumulative sum transform. """

    def transform(self, sequence: Dict['str', torch.Tensor]) -> torch.Tensor:
        embedded = sequence['embedded']
        embedded_cumsum = torch.cumsum(embedded, 1)
        return embedded_cumsum

    def generate(self, batch_group: Dict[str, Dict]) -> Dict[str, Dict]:
        for name, sequence in batch_group.items():
            if name.startswith('sequence'):
                sequence = batch_group[name]
                embedded_cumsum = self.transform(sequence)
                batch_group[name].update({'embedded': embedded_cumsum})
        return batch_group
