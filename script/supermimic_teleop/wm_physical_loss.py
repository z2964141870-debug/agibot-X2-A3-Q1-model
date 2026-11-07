"""A frozen WM supplies physical loss; its command-memory tail is excluded."""

import time

import torch


SETTING_KEYS = ("maximum_updates", "batch_size", "learning_rate", "seed",
                "gradient_norm_limit", "validation_every_updates")


def matched_settings(plan, control_plan):
    settings = {key: plan[key] for key in SETTING_KEYS}
    if any(settings[key] != control_plan["fit"][key] for key in SETTING_KEYS):
        raise ValueError("Physical fit differs from the published control settings")
    settings["physical_weight"] = .1
    return settings


class PhysicalLoss:
    def __init__(self, model, canonicalize, deadline_ns, maximum_calls=4096):
        self.model, self.canonicalize = model, canonicalize
        self.deadline_ns, self.maximum_calls, self.calls = deadline_ns, maximum_calls, 0
        self.present = model.joint_presence
        self.mask = model.feature_presence.detach()
        self.scale = model.normalizer.std.detach()
        if (model.cfg.bf16 is not False or any(p.requires_grad for p in model.parameters())
                or self.present.shape != (29,) or self.present.dtype != torch.bool
                or self.mask.shape != (268,) or self.mask.dtype != torch.bool or not self.mask.any()
                or self.scale.shape != (268,) or not torch.isfinite(self.scale).all()
                or not (self.scale > 0).all()):
            raise ValueError("Require the frozen FP32 model and its actual physical feature statistics")

    def features(self, state, action, context):
        if self.calls >= self.maximum_calls or time.monotonic_ns() >= self.deadline_ns:
            raise RuntimeError("Physical distillation WM call or wall budget reached")
        if (state.ndim != 2 or state.shape[1] != 269 or action.shape != (len(state), 29)
                or context.shape != (len(state), 62) or state.requires_grad):
            raise ValueError("Use the saved detached state and unscaled canonical29 action")
        self.calls += 1
        action = torch.where(self.present, action, 0.)
        prediction = self.model.step(state, action, context)
        if prediction.shape != state.shape or not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite or incorrectly shaped physical prediction")
        anchor = self.model.frame(state)
        anchor = anchor._replace(pos_xy=anchor.pos_xy.detach(), quat=anchor.quat.detach())
        features = self.canonicalize(prediction[..., :240], anchor)/self.scale
        if features.shape != (len(state), 268):
            raise ValueError("Canonical physical features must exclude the command-memory tail")
        return features[..., self.mask]

    def targets(self, state, teacher_action, context):
        with torch.no_grad():
            return self.features(state, teacher_action.detach(), context).detach()

    def loss(self, state, action, context, teacher_features):
        features = self.features(state, action, context)
        if (features.shape != teacher_features.shape or teacher_features.requires_grad
                or not torch.isfinite(features).all() or not torch.isfinite(teacher_features).all()):
            raise ValueError("Require finite detached same-state teacher physical features")
        return (features-teacher_features).square().mean()
