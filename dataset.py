import math
import random
from typing import Dict, Tuple, List

import torch
from torch.utils.data import Dataset


class CmapssRandomCropDataset(Dataset):
    def __init__(
        self,
        sequences_by_unit: Dict[int, torch.Tensor],
        targets_by_unit: Dict[int, torch.Tensor],
        samples_per_epoch: int,
        l_min: int = 20,
        l_max: int = 200,
        end_bias_alpha: float = 5.0,
        end_bias_beta: float = 2.0,
    ):
        self.sequences_by_unit = sequences_by_unit
        self.targets_by_unit = targets_by_unit
        self.unit_ids = list(sequences_by_unit.keys())
        self.samples_per_epoch = samples_per_epoch
        self.l_min = l_min
        self.l_max = l_max
        self.end_bias_alpha = end_bias_alpha
        self.end_bias_beta = end_bias_beta

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float, int]:
        unit_id = random.choice(self.unit_ids)

        x = self.sequences_by_unit[unit_id]
        y = self.targets_by_unit[unit_id]

        t_max = x.shape[0] - 1

        p = random.betavariate(self.end_bias_alpha, self.end_bias_beta)
        t = int(p * t_max)

        l_high = min(self.l_max, t + 1)

        if l_high <= self.l_min:
            l = l_high
        else:
            u = random.random()
            l = int(math.exp(math.log(self.l_min) + u * (math.log(l_high) - math.log(self.l_min))))

        start = t - l + 1

        seq = x[start : t + 1]
        target = float(y[t].item())

        length = seq.shape[0]
        return seq, target, length

class CmapssFullDataset(Dataset):
    def __init__(self, sequences_by_unit, targets_by_unit):
        self.sequences_by_unit = sequences_by_unit
        self.targets_by_unit = targets_by_unit
        self.unit_ids = sorted(sequences_by_unit.keys())

    def __len__(self):
        return len(self.unit_ids)

    def __getitem__(self, idx):
        unit_id = self.unit_ids[idx]
        seq = self.sequences_by_unit[unit_id]
        target = self.targets_by_unit[unit_id]

        length = int(seq.shape[0])
        return seq, float(target), length
    



class CmapssLastWindowDataset(Dataset):
    def __init__(self, sequences_by_unit, targets_by_unit, window_len: int):
        self.sequences_by_unit = sequences_by_unit
        self.targets_by_unit = targets_by_unit
        self.window_len = window_len
        self.unit_ids = sorted(sequences_by_unit.keys())

    def __len__(self):
        return len(self.unit_ids)

    def __getitem__(self, idx):
        unit_id = self.unit_ids[idx]
        seq = self.sequences_by_unit[unit_id]
        target = self.targets_by_unit[unit_id]

        if self.window_len is not None and seq.shape[0] > self.window_len:
            seq = seq[-self.window_len:]

        length = int(seq.shape[0])
        return seq, float(target), length