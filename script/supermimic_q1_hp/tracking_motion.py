"""Preserve reference features and repair only transition-start bin ownership."""

import torch

from .motion_base import BIN_FRAMES, CONTROL_DT
from .motion_base import MotionLib as BaseMotionLib
from .transition_bins import transition_bin_layout


class MotionLib(BaseMotionLib):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = transition_bin_layout(self.clip_len.tolist(), BIN_FRAMES)
        for name, values in layout.items():
            setattr(self, name, torch.tensor(values, dtype=torch.long, device=self.device))
        self.num_bins = len(layout["bin_start"])
