"""Differentiable collision-sole height from the existing Q1 full-body FK."""

import torch
from torch import nn


class TensorSole(nn.Module):
    def __init__(self, geometry, body_names, device):
        super().__init__()
        if len(set(body_names)) != len(body_names):
            raise ValueError("Ambiguous kinematics body names")
        self.body_count = len(body_names)
        self.body_ids = [body_names.index(foot.body) for foot in geometry.feet]
        self.contract = geometry.identity()
        for index, foot in enumerate(geometry.feet):
            self.register_buffer(f"points_{index}", torch.tensor(foot.points, device=device, dtype=torch.float32))
            self.register_buffer(f"radii_{index}", torch.tensor(foot.radii, device=device, dtype=torch.float32))

    def forward(self, positions, rotations):
        if (positions.shape[-2:] != (self.body_count, 3)
                or rotations.shape[-3:] != (self.body_count, 3, 3)
                or positions.shape[:-2] != rotations.shape[:-3]):
            raise ValueError("Full-body FK position/rotation dimensions differ from their body names")
        values = []
        for index, body in enumerate(self.body_ids):
            points = getattr(self, f"points_{index}").to(rotations)
            radii = getattr(self, f"radii_{index}").to(rotations)
            z = rotations[..., body, 2, :] @ points.T
            z = z + positions[..., body, 2, None] - radii
            values.append(z.amin(dim=-1))
        return torch.stack(values, dim=-1)


def swing_z_correction(ankle, reference_ankle, sole, reference_sole, swing):
    if (ankle.shape[-1:] != (2,) or any(value.shape != ankle.shape for value in
            (reference_ankle, sole, reference_sole, swing)) or swing.dtype != torch.bool):
        raise ValueError("Expected aligned two-foot Z values and a boolean swing mask")
    ankle_score = (-((ankle - reference_ankle) / .04).square()).exp()
    sole_score = (-((sole - reference_sole) / .04).square()).exp()
    # The original XY half and .02 reward weight are retained exactly.
    return .02 * .5 * ((sole_score - ankle_score) * swing).sum(-1) / swing.sum(-1).clamp(min=1)
