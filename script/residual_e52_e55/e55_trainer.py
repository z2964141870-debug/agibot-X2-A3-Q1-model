"""E55 trainer: E53 slot-interleave machinery reused verbatim, plus the residual head.

Geometry: the hydra anchor term func returns the native 64-wide slot (58 future
q/qd + 6 ori) plus the 6 arm channels = 70/slot, flat 2462. The E53 torch.load
interleave pads 035 580->640 with zero columns, so the base net stays bit-equal
to 035 at t=0 (cloth anchor guard passes); the channels ride INSIDE the
observation, which keeps PPO replay batches aligned automatically.

Only the residual head is trainable: it slices each 70-wide slot down to the
first 58 (reference q/qd block), concatenates the slot's trailing 6 channel
dims (the arm variable), and adds a zero-init residual on the actor output.
"""

import os
import sys
import sys as _sys
import torch

# E55 P1.1 amendment — 断D (lead ruling 2026-10-20, solidified in e55_trainer.py):
# the cached-token frozen-check ("every other Actor parameter frozen") is
# allowlisted for e55_head.* ONLY, because the residual action head is
# post-token and the sampled-token ABI protection does not apply to it.
# Location per lead ruling: NOT cloth_cached_tokens.py (campaign shared file
# stays untouched during the battle; frozen-check semantics are shared by
# other cells). This block is the material for the post-battle upstream
# proposal to the shared-layer owner.
def _e55_patch_cached_distribution():
    for _name in ("script.a3.cloth_cached_tokens", "cloth_cached_tokens"):
        _mod = _sys.modules.get(_name)
        if _mod is None or getattr(_mod, "_e55_allow_head", False):
            continue
        _orig = getattr(_mod, "cached_distribution", None)
        if _orig is None:
            continue
        try:
            from script.a3.cloth_constrained_trainer import DECODER as _dec
        except Exception:
            continue
        # Full-body replacement (sync source: script/a3/cloth_cached_tokens.py
        # cached_distribution @ 2026-10-19): the wrapped variant re-entered the
        # original strict check. Allowlist differs only in the frozen-check.
        import torch as _torch
        def _allowed(policy, obs, tokens, actions, _dec=_dec):
            if any(p.requires_grad for name, p in policy.named_parameters()
                   if not (name.startswith(_dec) or name.startswith("e55_head"))):
                raise ValueError("Sampled token reuse requires every other Actor parameter frozen")
            if tokens.requires_grad or tokens.shape != (*actions.shape[:2], 2, 32):
                raise ValueError("Detached sampled token ABI changed")
            if not _torch.isfinite(tokens).all():
                raise ValueError("Nonfinite sampled tokens")
            mean = policy.actor_module.forward_with_external_tokens(obs, tokens)
            if mean.shape != actions.shape:
                raise ValueError("External-token decoder action ABI changed")
            std = policy.get_std
            std = _torch.where(_torch.isfinite(std), std.clamp_min(1e-6), _torch.full_like(std, 1e-6))
            policy.distribution = _torch.distributions.Normal(mean, std)
            return dict(action_mean=policy.action_mean, action_std=policy.action_std,
                logprobs=policy.get_actions_log_prob(actions), entropy=policy.entropy)
        _mod.cached_distribution = _allowed
        _mod._e55_allow_head = True
        print("[e55-fix] cached_distribution allowlist e55_head installed", flush=True)
        return

_e55_patch_cached_distribution()

from script.a3.cloth_curve_trainer import ClothCurveTrainer
import e55_residual


from script.a3.cloth_constrained_trainer import ClothConstrainedTrainer, parameter_digest


class E55RecordOnlyDriftMixin:
    """E53 amendment carried to E55 verbatim: the E-series 0.05/0.5 anchor drift
    budget is record-only (E52 arm-A interception made arms untrainable; lead
    mandated teacher MSE record-only). E55 trains only the residual head, but the
    head's action delta is part of the measured policy output, so interception
    would stop the very thing the experiment trains. Frozen-parameter/buffer
    integrity checks are kept."""
    def _after_optimizer_step(self, optimizer, args, kwargs):
        super(ClothConstrainedTrainer, self)._after_optimizer_step(optimizer, args, kwargs)
        self._assert_frozen()
        if os.environ.get("YUANQI_E55_PROBE") and self._optimizer_step_ordinal <= 2:
            policy_p = self.accelerator.unwrap_model(self.model).policy
            for name, param in policy_p.named_parameters():
                if name.startswith("e55_head."):
                    g = param.grad
                    print("[e55-probe] grad %s none=%s norm=%s" % (
                        name, g is None, "None" if g is None else round(float(g.float().norm()), 10)), flush=True)
            if self._optimizer_step_ordinal >= 2:
                print("[e55-probe] STOP after step", self._optimizer_step_ordinal, flush=True)
                sys.stdout.flush()
                os._exit(0)
        policy = self.accelerator.unwrap_model(self.model).policy
        with torch.no_grad():
            means = torch.cat([policy.forward({n: v[i:i + 4] for n, v in self._cloth_obs.items()},
                                              is_training=False) for i in range(0, 16, 4)], dim=0)
        difference = means - self._cloth_targets
        row = dict(optimizer_step=self._optimizer_step_ordinal,
                   teacher_action_rmse=float(difference.square().mean().sqrt()),
                   teacher_action_max_abs=float(difference.abs().max()),
                   retention_loss_before_step=self._cloth_last_retention_loss,
                   frozen_parameter_digest=parameter_digest(self._cloth_frozen),
                   head_parameter_digest=parameter_digest(policy.e55_head.named_parameters()))
        self._cloth_records.append(row)
        self._save_cloth_receipt()


class E55ResidualTrainer(E55RecordOnlyDriftMixin, ClothCurveTrainer):
    """v1.2 native-640: 035 loads EXACT (no interleave, no weight surgery); the
    residual head is the only trainable part; channels reach ONLY the head via
    the obs-hook cell (e55_obs_terms.CHANNEL_CELL)."""
    _e55_ready = False  # class attr: create_optimizer runs inside super().__init__

    def _e55_policy(self):
        return self.accelerator.unwrap_model(self.model).policy

    def _e55_ensure_head(self):
        if self._e55_ready:
            return
        policy = self._e55_policy()
        specs = {}
        for name, start, end, _dims in policy.actor_module.tokenizer_obs_specs:
            specs[name] = (start, end - start)
        cmd_start, cmd_width = specs["command_multi_future_nonflat"]
        if cmd_width != 580:
            raise ValueError("E55: command group width %d != 10x58" % cmd_width)
        if "motion_anchor_ori_b_mf_nonflat" in specs and specs["motion_anchor_ori_b_mf_nonflat"][1] != 60:
            raise ValueError("E55: anchor term width %d != vendor 10x6 (native contract)" % specs["motion_anchor_ori_b_mf_nonflat"][1])
        head = e55_residual.install_injected(policy, cmd_start, slots=10, slot_anchor=58,
                                             ori_width=6, channels=6,
                                             cmd_start=cmd_start, mf_start=None)
        for name, param in policy.named_parameters():
            param.requires_grad = name.startswith("e55_head.")
        self._e55_ready = True

    def load_checkpoint(self, *args, **kwargs):
        """035 base keys load untouched (no interleave, v1.2); the zero-init head is
        already installed (create_optimizer precedes load), so torch.load is wrapped
        once to add the head's zero state to the dict - live==source holds and the
        FinetuneTrainer exact guard passes without touching 035 weights."""
        import torch as _torch
        orig_load = _torch.load
        head_state = {k: v.detach().clone() for k, v in self._e55_policy().e55_head.state_dict().items()}
        def _patched_load(*a, **kw):
            loaded = orig_load(*a, **kw)
            if isinstance(loaded, dict) and "policy_state_dict" in loaded:
                for k, v in head_state.items():
                    loaded["policy_state_dict"].setdefault("e55_head." + k, v)
            return loaded
        _torch.load = _patched_load
        try:
            return super().load_checkpoint(*args, **kwargs)
        finally:
            _torch.load = orig_load

    def create_optimizer(self):
        self._e55_ensure_head()
        opt = super().create_optimizer()
        head_params = [p for n, p in self._e55_policy().named_parameters() if n.startswith("e55_head.")]
        in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
        missing = [p for p in head_params if id(p) not in in_opt]
        if missing:
            base_g = opt.param_groups[0]
            kw = {k: v for k, v in base_g.items() if k != "params"}
            kw["params"] = missing
            for _p in missing:
                _p.requires_grad_(True)
            opt.add_param_group(kw)
            print("[e55-fix] head param_group added n=%d lr=%s unfrozen" % (len(missing), kw.get("lr")), flush=True)
        if os.environ.get("YUANQI_E55_PROBE"):
            in_opt = set()
            for group in opt.param_groups:
                for p in group["params"]:
                    in_opt.add(id(p))
            head_ids, total_ids = set(), set()
            for name, p in self._e55_policy().named_parameters():
                total_ids.add(id(p))
                if name.startswith("e55_head."):
                    head_ids.add(id(p))
                    print("[e55-probe] opt-contains %s: %s" % (name, id(p) in in_opt), flush=True)
            print("[e55-probe] policy params=%d head=%d opt_members=%d" % (
                len(total_ids), len(head_ids), len(total_ids & in_opt)), flush=True)
            act = self._e55_policy().actor_module
            print("[e55-probe] actor_id=%d fwet_wrapped=%s" % (
                id(act), getattr(getattr(act, "forward_with_external_tokens", None), "_e55_wrapped", False)), flush=True)
        return opt
