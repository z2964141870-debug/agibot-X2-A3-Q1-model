"""Bounded latent residuals conditioned on already-causal robot history."""

import copy

import torch
from torch import nn

from student_models import current_student


ARMS = ("zero_history", "causal_history")


class ProprioResidualStudent(nn.Module):
    def __init__(self, base, arm, latent_dim):
        super().__init__()
        if arm not in ARMS or type(latent_dim) is not int or latent_dim <= 0:
            raise ValueError("Require the registered history arm and actual latent width")
        self.arm = arm
        self.encoder = copy.deepcopy(base.encoder).eval().requires_grad_(False)
        self.decoder = copy.deepcopy(base.decoder).eval().requires_grad_(False)
        self.adapter = nn.Sequential(nn.Linear(930, 128), nn.ELU(), nn.Linear(128, latent_dim))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        reference = next(self.decoder.parameters())
        self.adapter.to(device=reference.device, dtype=reference.dtype)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.decoder.eval()
        return self

    def latent(self, normalized_history, goal):
        if (normalized_history.ndim != 2 or normalized_history.shape[1] != 930
                or goal.shape != (len(normalized_history), 10, 226)):
            raise ValueError("Use paired current human and 930 normalized robot-history features")
        base = self.encoder(goal)
        history = normalized_history if self.arm == "causal_history" else torch.zeros_like(normalized_history)
        delta = torch.tanh(self.adapter(history))
        return base+(1.-base.abs())*delta

    def forward(self, normalized_history, goal):
        return self.decoder(self.latent(normalized_history, goal), normalized_history)

    def trainable_parameters(self):
        return self.adapter.parameters()


def residual_student(native, payload, arm):
    base = current_student(native, payload["current_statistics"])
    base.encoder.load_state_dict(payload["encoder"], strict=True)
    return ProprioResidualStudent(base, arm, native.policy.cfg.latent)


def zero_adapter(state):
    return (set(state) == {"0.weight", "0.bias", "2.weight", "2.bias"}
        and not bool(torch.count_nonzero(state["2.weight"]))
        and not bool(torch.count_nonzero(state["2.bias"])))
