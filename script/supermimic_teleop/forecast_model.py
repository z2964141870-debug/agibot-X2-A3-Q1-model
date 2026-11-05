"""Small causal GRU specification and training-only forecast normalization.

Torch is imported only when constructing a model in an admitted runtime. A zero
head predicts zero physical residuals (hold); label means are not subtracted.
This module neither acquires compute resources nor launches any training.
"""

from dataclasses import dataclass

import numpy as np

from forecast_data import TrainingMoments
from forecast_geometry import FEATURE_DIM, TARGET_DIM, encode_history, encode_targets, immutable


SCHEMA = "q1_history_yaw_root_body_so3_v1"


@dataclass(frozen=True)
class ForecastSpec:
    history_frames: int = 25
    horizon_frames: int = 10
    hidden_size: int = 128

    def __post_init__(self):
        if (self.history_frames != 25 or self.horizon_frames != 10
                or type(self.hidden_size) is not int or not 32 <= self.hidden_size <= 256):
            raise ValueError("Require the frozen25-history/10-horizon schema and a bounded32..256 GRU")

    def to_dict(self):
        return dict(schema=SCHEMA, history_frames=self.history_frames,
            horizon_frames=self.horizon_frames, hidden_size=self.hidden_size,
            feature_dim=FEATURE_DIM, target_dim=TARGET_DIM, layers=1,
            bidirectional=False, dropout=0., initial_physical_residual="zero_hold")


class NormalizationFit:
    def __init__(self):
        self.inputs, self.labels = TrainingMoments(), TrainingMoments()
        self.sources = set()

    def update(self, window):
        if window.split != "train":
            raise ValueError("Only training windows may fit model normalization")
        history = encode_history(window.input_positions, window.input_quaternions)
        targets = encode_targets(window.target_positions, window.target_quaternions, history.anchor)
        self.inputs.update(history.features, split=window.split)
        self.labels.update(targets, split=window.split)
        self.sources.add(window.relative)

    def freeze(self):
        inputs, labels = self.inputs.parameters(), self.labels.parameters()
        # Physical floors are fixed in advance: 1cm for positions, 0.05rad for rotations.
        floors = np.array([.01]*72+[.05]*72)
        return ForecastNormalization(inputs["mean"], inputs["scale"],
            np.maximum(labels["scale"], floors), inputs["count"], labels["count"],
            tuple(sorted(self.sources)))


@dataclass(frozen=True)
class ForecastNormalization:
    input_mean: np.ndarray
    input_scale: np.ndarray
    target_scale: np.ndarray
    input_count: int
    target_count: int
    fitted_sources: tuple

    def __post_init__(self):
        for name, shape in (("input_mean", (FEATURE_DIM,)), ("input_scale", (FEATURE_DIM,)),
                            ("target_scale", (TARGET_DIM,))):
            values = immutable(getattr(self, name))
            if values.shape != shape or (name.endswith("scale") and np.any(values <= 0)):
                raise ValueError("Invalid forecast normalization dimensions or positive scales")
            object.__setattr__(self, name, values)
        if (type(self.input_count) is not int or self.input_count <= 0
                or type(self.target_count) is not int or self.target_count <= 0
                or not self.fitted_sources or len(set(self.fitted_sources)) != len(self.fitted_sources)):
            raise ValueError("Require actual training counts and distinct fitted source identities")

    def inputs(self, features):
        values = np.asarray(features, dtype=np.float64)
        if not values.ndim or values.shape[-1] != FEATURE_DIM or not np.isfinite(values).all():
            raise ValueError("Invalid finite history features")
        return ((values-self.input_mean)/self.input_scale).astype(np.float32)

    def targets(self, labels):
        values = np.asarray(labels, dtype=np.float64)
        if not values.ndim or values.shape[-1] != TARGET_DIM or not np.isfinite(values).all():
            raise ValueError("Invalid finite future labels")
        return (values/self.target_scale).astype(np.float32)

    def physical(self, normalized):
        values = np.asarray(normalized, dtype=np.float64)
        if not values.ndim or values.shape[-1] != TARGET_DIM or not np.isfinite(values).all():
            raise ValueError("Invalid finite normalized forecasts")
        return values*self.target_scale

    def to_dict(self):
        return dict(schema=SCHEMA, fitted_split="train", input_mean=self.input_mean.tolist(),
            input_scale=self.input_scale.tolist(), target_scale=self.target_scale.tolist(),
            target_mean_subtracted=False, input_count=self.input_count,
            target_count=self.target_count, fitted_sources=list(self.fitted_sources))

    @classmethod
    def from_dict(cls, record, allowed_train_sources):
        if (record.get("schema") != SCHEMA or record.get("fitted_split") != "train"
                or record.get("target_mean_subtracted") is not False
                or not set(record["fitted_sources"]) <= set(allowed_train_sources)):
            raise ValueError("Normalizer includes another schema or a held-out source")
        return cls(*(record[key] for key in ("input_mean", "input_scale", "target_scale",
            "input_count", "target_count")), tuple(record["fitted_sources"]))


def make_model(spec=ForecastSpec()):
    import torch
    from torch import nn

    class CausalForecastGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.recurrent = nn.GRU(FEATURE_DIM, spec.hidden_size, batch_first=True,
                                    num_layers=1, bidirectional=False, dropout=0.)
            self.head = nn.Linear(spec.hidden_size, spec.horizon_frames*TARGET_DIM)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, history):
            if history.ndim != 3 or tuple(history.shape[1:]) != (spec.history_frames, FEATURE_DIM):
                raise ValueError("Model accepts only[batch,25,216] visible history")
            _, hidden = self.recurrent(history)
            return self.head(hidden[-1]).reshape(len(history), spec.horizon_frames, TARGET_DIM)

    return CausalForecastGRU()
