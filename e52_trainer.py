"""E52 three-arm finetune trainers: E45 full-state resume, optional projection.

Arm A keeps the original Adam displacement and the original teacher hard gate.
Arms B/C resume identically, then replace interception with recording, and arm B
additionally projects every Adam displacement with the E50 teacher first-order
non-increasing half-space operator before it commits.
"""

import copy
import hashlib
import json
import os
from pathlib import Path

import torch

from script.a3.checkpoint_store import atomic_json
from script.a3.cloth_curve_trainer import ClothCurveTrainer
from script.a3.optimizer_metadata_contract import canonical_groups


def canonical_optimizer_state_sha256(state):
    blob = json.dumps({k: {n: (list(map(float, t.reshape(-1)[:4].tolist())) if torch.is_tensor(t) else repr(t))
                           for n, t in v.items()} for k, v in sorted(state["state"].items(), key=lambda kv: int(kv[0]))},
                      sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def canonical_optimizer_sha256(state):
    blob = json.dumps({
        "groups": [dict((k, repr(v)) for k, v in g.items()) for g in state["param_groups"]],
        "state": {k: {n: (list(map(float, t.reshape(-1)[:4].tolist())) if torch.is_tensor(t) else repr(t))
                      for n, t in v.items()} for k, v in sorted(state["state"].items(), key=lambda kv: int(kv[0]))},
    }, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


class E52ResumeBase(ClothCurveTrainer):
    """Restores the exact E45 step4 optimizer state after the vendor warm start.

    The vendor warm start verifies exact actor/critic weights and asserts an
    empty optimizer; the E45 Adam moments are reinstated right after that check.
    """

    def load_checkpoint(self, checkpoint_path, resume=False):
        result = super().load_checkpoint(checkpoint_path, resume=False)
        self._e52_restore_optimizer()
        return result

    def _e52_restore_optimizer(self):
        source = Path(os.environ["YUANQI_E52_SOURCE_CHECKPOINT"])
        payload = torch.load(source, map_location="cpu", weights_only=False)
        saved = payload["optimizer_state_dict"]
        if int(payload["state"].global_step) != 4:
            raise ValueError("E45 source checkpoint step changed")
        arm = os.environ.get("YUANQI_E52_ARM", "?")
        pre = [{k: v for k, v in g.items() if k != "params"} for g in self.optimizer.param_groups]
        self.optimizer.load_state_dict(copy.deepcopy(saved))
        if arm == "C":
            for group, keep in zip(self.optimizer.param_groups, pre):
                for key, value in keep.items():
                    if key in ("lr", "initial_lr") and group.get(key) != value:
                        group[key] = value
        steps = {int(k): int(v.get("step", 0)) for k, v in self.optimizer.state_dict()["state"].items()}
        if any(v != 16 for v in steps.values()):
            raise ValueError("Restored Adam moments are not the E45 step-16 state: " + repr(sorted(set(steps.values()))))
        current_groups = canonical_groups([{k: v for k, v in g.items() if k != "params"}
                                           for g in self.optimizer.param_groups])
        payload_groups = canonical_groups([{k: v for k, v in g.items() if k != "params"}
                                           for g in saved["param_groups"]])
        differences = [(a, b) for a, b in zip(current_groups, payload_groups)
                       for pair in zip(a, b) if pair[0] != pair[1]]
        if arm == "C":
            if any(k != "lr" and k != "initial_lr" for pair in differences for k, _ in pair):
                raise ValueError("Arm C group metadata differs beyond the registered LR dose")
        elif differences:
            raise ValueError("Restored optimizer group metadata differs from the E45 payload")
        receipt = dict(
            arm=arm,
            source_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            optimizer_state_sha256=canonical_optimizer_state_sha256(self.optimizer.state_dict()),
            payload_optimizer_state_sha256=canonical_optimizer_state_sha256(saved),
            adam_state_steps=sorted(set(steps.values())),
            param_groups=current_groups,
        )
        if receipt["optimizer_state_sha256"] != receipt["payload_optimizer_state_sha256"]:
            raise ValueError("Restored Adam moments differ from the E45 payload")
        directory = Path(os.environ["YUANQI_RESUME_LOG_DIR"])
        atomic_json(directory / "step0_optimizer_restore.json", receipt)


class E52ArmATrainer(E52ResumeBase):
    """Arm A: original recipe; the base class raises at the teacher budget."""

    def _e52_restore_optimizer(self):
        if os.environ.get("YUANQI_E52_ARM") != "A":
            raise ValueError("Arm A trainer selected without arm A environment")
        super()._e52_restore_optimizer()


class ProjectionAdamW(torch.optim.AdamW):
    """Shadow-step the native AdamW proposal, project it, commit unconditionally."""

    def __init__(self, params, *, teacher_gradients, record):
        super().__init__(params)
        self._teacher_gradients = teacher_gradients
        self._record = record
        self._ordinal = 0

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Projection Adam requires no closure")
        groups, live, starts, shadows = [], [], [], []
        for group in self.param_groups:
            if any(group.get(k, False) for k in ("amsgrad", "maximize", "capturable", "differentiable")):
                raise ValueError("Unsupported AdamW mode")
            clones = []
            for param in group["params"]:
                if param.dtype != torch.float32 or param.grad is None or not torch.isfinite(param.grad).all():
                    raise ValueError("Missing or nonfinite FP32 gradient")
                clone = torch.nn.Parameter(param.detach().clone(), requires_grad=True)
                clone.grad = param.grad.detach().clone()
                clones.append(clone)
                live.append(param)
                starts.append(param.detach().clone())
                shadows.append(clone)
            groups.append(dict(group, params=clones))
        before = copy.deepcopy(self.state_dict())
        shadow = torch.optim.AdamW(groups)
        shadow.load_state_dict(copy.deepcopy(before))
        grads = self._teacher_gradients()
        if len(grads) != len(live):
            raise ValueError("Teacher gradient coverage differs from optimizer parameters")
        shadow.step()
        rows, committed = [], True
        projected = []
        for index, (start, value, grad) in enumerate(zip(starts, [p.detach().clone() for p in shadows], grads)):
            displacement = value - start
            gnorm = float(grad.detach().double().pow(2).sum().sqrt())
            if gnorm == 0.0:
                dot, scale, zero_teacher = 0.0, 0.0, True
                new_value = value
            else:
                dot = float((grad.detach().double() * displacement.double()).sum())
                scale = max(0.0, dot) / float(grad.detach().double().pow(2).sum())
                new_value = value - scale * grad
                zero_teacher = False
            projected.append(new_value)
            rows.append(dict(index=index, lr=groups[0].get("lr") if groups else None,
                             displacement_norm=float(displacement.double().pow(2).sum().sqrt()),
                             projected_norm=float((new_value - start).double().pow(2).sum().sqrt()),
                             teacher_dot=dot, projection_scale=scale, zero_teacher_gradient=zero_teacher))
        try:
            next_state = copy.deepcopy(shadow.state_dict())
            self.load_state_dict(next_state)
            for param, value in zip(live, projected):
                param.copy_(value)
        except BaseException:
            for param, value in zip(live, starts):
                param.copy_(value)
            self.load_state_dict(before)
            raise
        self._ordinal += 1
        self._record(dict(ordinal=self._ordinal, committed=committed, projections=rows))
        return None


class E52ResumeTrainer(E52ResumeBase):
    """Arms B/C: restore, then record the teacher gate instead of intercepting."""

    def create_optimizer(self):
        result = super().create_optimizer()
        arm = os.environ.get("YUANQI_E52_ARM")
        if arm not in ("B", "C"):
            raise ValueError("Resume trainer requires arm B or C")
        old = self.optimizer
        if type(old) is not torch.optim.AdamW:
            raise ValueError("Expected native AdamW from the vendor recipe")
        groups = [dict(g, params=list(g["params"])) for g in old.param_groups]
        if arm == "B":
            self.optimizer = ProjectionAdamW(
                groups, teacher_gradients=self._e52_teacher_gradients, record=self._e52_record_projection)
        else:
            self.optimizer = old
        return result

    def _e52_teacher_gradients(self):
        policy = self.accelerator.unwrap_model(self.model).policy
        with torch.enable_grad():
            means = torch.cat([policy.forward({n: v[i:i + 4] for n, v in self._cloth_obs.items()},
                                              is_training=False) for i in range(0, 16, 4)], dim=0)
            loss = (means - self._cloth_targets).square().mean()
            params = [p for p in policy.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, params, allow_unused=False)
        table = {id(p): g.detach() for p, g in zip(params, grads)}
        aligned = []
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                gradient = table.get(id(param))
                aligned.append(gradient if gradient is not None else torch.zeros_like(param))
        return aligned

    def _e52_record_projection(self, row):
        directory = Path(os.environ["YUANQI_RESUME_LOG_DIR"])
        with open(directory / "projection_steps.jsonl", "a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()

    def _after_optimizer_step(self, optimizer, args, kwargs):
        policy = self.accelerator.unwrap_model(self.model).policy
        with torch.no_grad():
            means = torch.cat([policy.forward({n: v[i:i + 4] for n, v in self._cloth_obs.items()},
                                              is_training=False) for i in range(0, 16, 4)], dim=0)
        difference = means - self._cloth_targets
        from script.a3.cloth_constrained_trainer import parameter_digest
        row = dict(optimizer_step=self._optimizer_step_ordinal,
                   teacher_action_rmse=float(difference.square().mean().sqrt()),
                   teacher_action_max_abs=float(difference.abs().max()),
                   retention_loss_before_step=self._cloth_last_retention_loss,
                   frozen_parameter_digest=parameter_digest(self._cloth_frozen),
                   gate="RECORDED_NOT_INTERCEPTED")
        self._cloth_records.append(row)
        self._save_cloth_receipt()
        return None
