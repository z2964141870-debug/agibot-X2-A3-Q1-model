"""Sparse real-joint bridge for a new MorphMimic Q1 integration.

This is not installed into the frozen upstream Trainer. Missing canonical slots
have no native index: every gather/scatter uses only the explicit valid pairs.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor


Q1_MISSING = frozenset([
    "waist_pitch_joint",
    *[f"{side}_wrist_{axis}_joint" for side in ("left", "right")
      for axis in ("roll", "pitch", "yaw")],
])


@dataclass(frozen=True)
class SparseJointMap:
    canonical_names: tuple[str, ...]
    native_names: tuple[str, ...]
    native_for_slot: tuple[int | None, ...]
    signs: tuple[float, ...]
    offsets: tuple[float, ...]

    def __post_init__(self):
        count = len(self.canonical_names)
        if not count or any(len(x) != count for x in
                            (self.native_for_slot, self.signs, self.offsets)):
            raise ValueError("Canonical names, indices, signs and offsets must align")
        if len(set(self.canonical_names)) != count or not self.native_names:
            raise ValueError("Require unique canonical and nonempty native names")
        if len(set(self.native_names)) != len(self.native_names):
            raise ValueError("Native joint names must be unique")
        real = [i for i in self.native_for_slot if i is not None]
        if not real or len(set(real)) != len(real):
            raise ValueError("Real joint mappings must be nonempty and injective")
        if any(type(i) is not int or not 0 <= i < len(self.native_names) for i in real):
            raise ValueError("Real native indices must be in range; -1 is invalid")
        if any(s not in (-1.0, 1.0) for s in self.signs):
            raise ValueError("Joint signs must be +1 or -1")
        if not all(math.isfinite(o) for o in self.offsets):
            raise ValueError("Joint offsets must be finite")
        if any(o != 0 for i, o in zip(self.native_for_slot, self.offsets) if i is None):
            raise ValueError("Absent slots must have zero offset")

    @classmethod
    def q1(cls, canonical_names, native_names):
        canonical_names, native_names = tuple(canonical_names), tuple(native_names)
        if len(canonical_names) != 29 or len(native_names) != 22:
            raise ValueError("Expected MorphMimic29 and physical Q1 22 joints")
        if set(canonical_names) - set(native_names) != Q1_MISSING:
            raise ValueError("Unexpected missing Q1 joint names")
        if set(native_names) - set(canonical_names):
            raise ValueError("Unknown physical Q1 joint")
        return cls(canonical_names, native_names,
                   tuple(native_names.index(n) if n in native_names else None
                         for n in canonical_names),
                   (1.0,) * 29, (0.0,) * 29)

    def _pairs(self, ref):
        pairs = [(s, n) for s, n in enumerate(self.native_for_slot) if n is not None]
        slots = torch.tensor([s for s, _ in pairs], device=ref.device, dtype=torch.long)
        native = torch.tensor([n for _, n in pairs], device=ref.device, dtype=torch.long)
        signs = ref.new_tensor([self.signs[s] for s, _ in pairs])
        offsets = ref.new_tensor([self.offsets[s] for s, _ in pairs])
        return slots, native, signs, offsets

    def mask(self, ref):
        return torch.tensor([i is not None for i in self.native_for_slot],
                            device=ref.device, dtype=torch.bool)

    def _width(self, x, n):
        if x.ndim == 0 or x.shape[-1] != n:
            raise ValueError(f"Expected last dimension {n}, got {tuple(x.shape)}")

    def canonical(self, native, *, delta=False):
        self._width(native, len(self.native_names))
        slots, indices, signs, offsets = self._pairs(native)
        out = native.new_zeros(*native.shape[:-1], len(self.canonical_names))
        out[..., slots] = signs * (native[..., indices] - (0 if delta else offsets))
        return out

    def native(self, canonical, default=None, *, delta=False):
        self._width(canonical, len(self.canonical_names))
        slots, indices, signs, offsets = self._pairs(canonical)
        if default is None or delta:
            out = canonical.new_zeros(*canonical.shape[:-1], len(self.native_names))
        else:
            self._width(default, len(self.native_names))
            out = default.to(canonical).expand(
                *canonical.shape[:-1], len(self.native_names)).clone()
        out[..., indices] = signs * canonical[..., slots] + (0 if delta else offsets)
        return out

    def project(self, canonical):
        self._width(canonical, len(self.canonical_names))
        return torch.where(self.mask(canonical), canonical, torch.zeros_like(canonical))

    def limits(self, native_limits):
        if native_limits.shape != (len(self.native_names), 2):
            raise ValueError("Expected native joint limits [J, 2]")
        slots, indices, signs, offsets = self._pairs(native_limits)
        bounds = signs[:, None] * (native_limits[indices] - offsets[:, None])
        out = native_limits.new_zeros(len(self.canonical_names), 2)
        out[slots] = torch.sort(bounds, dim=-1).values
        return out

    def state(self, state):
        self._width(state, 240)
        return torch.cat((self.project(state[..., :29]),
                          self.project(state[..., 29:58]), state[..., 58:]), dim=-1)

    def proprio(self, observation):
        self._width(observation, 93)
        return torch.cat((self.project(observation[..., :29]),
                          self.project(observation[..., 29:58]),
                          observation[..., 58:64],
                          self.project(observation[..., 64:93])), dim=-1)

    def context(self, context):
        self._width(context, 33)
        return torch.cat((context[..., :1], self.project(context[..., 1:30]),
                          context[..., 30:]), dim=-1)

    def goal(self, goal):
        self._width(goal, 670)
        frames = goal.reshape(*goal.shape[:-1], 10, 67)
        out = torch.cat((self.project(frames[..., :29]),
                         self.project(frames[..., 29:58]), frames[..., 58:]), dim=-1)
        return out.flatten(-2)

    def joint_rmse(self, predicted, target):
        self._width(predicted, len(self.canonical_names))
        self._width(target, len(self.canonical_names))
        slots, _, _, _ = self._pairs(predicted)
        return (predicted[..., slots] - target[..., slots]).square().mean(-1).sqrt()


def morphology_with_presence(actuator_terms, geometry, presence):
    """Fit a fixed training-population scale and return [robots, 154].

    Input groups are positive action scale/kp/kd/effort [R,4,29] where present,
    geometry [R,9] and bool presence [R,29]. Absence never enters log or bounds.
    Persist the returned bounds with the run; do not refit at deployment.
    """
    if actuator_terms.ndim != 3 or actuator_terms.shape[1:] != (4, 29):
        raise ValueError("Expected actuator terms [R, 4, 29]")
    robots = actuator_terms.shape[0]
    if geometry.shape != (robots, 9) or presence.shape != (robots, 29):
        raise ValueError("Geometry and presence must match the training robots")
    if presence.dtype != torch.bool or not robots or not presence.any(-1).all():
        raise ValueError("Every robot needs a nonempty boolean joint mask")
    valid_act = presence[:, None, :].expand_as(actuator_terms)
    if not torch.isfinite(geometry).all() or not (geometry > 0).all():
        raise ValueError("Geometry must be finite and positive")
    if not torch.isfinite(actuator_terms[valid_act]).all() or not (
            actuator_terms[valid_act] > 0).all():
        raise ValueError("Present actuator terms must be finite and positive")
    safe = torch.where(valid_act, actuator_terms, torch.ones_like(actuator_terms))
    raw = torch.cat((safe.log().flatten(1), geometry), dim=-1)
    valid = torch.cat((valid_act.flatten(1), torch.ones_like(geometry, dtype=torch.bool)),
                      dim=-1)
    lo = torch.where(valid, raw, torch.full_like(raw, float("inf"))).amin(0)
    hi = torch.where(valid, raw, torch.full_like(raw, -float("inf"))).amax(0)
    any_valid = valid.any(0)
    lo = torch.where(any_valid, lo, torch.zeros_like(lo))
    hi = torch.where(any_valid, hi, torch.zeros_like(hi))
    span = hi - lo
    changing = span > 1e-12 * torch.maximum(torch.ones_like(hi), hi.abs())
    scale = 2 * (raw - lo) / torch.where(changing, span, torch.ones_like(span)) - 1
    scale = torch.where(valid & changing, scale, torch.zeros_like(scale))
    return torch.cat((scale, presence.to(raw)), dim=-1), {"min": lo, "max": hi}
