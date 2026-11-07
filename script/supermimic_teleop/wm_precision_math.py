"""Independent precision comparison with the original physical-only probe gates."""

import copy
from dataclasses import asdict, replace

import torch


def fp32_shadow(native):
    if native.cfg.bf16 is not True:
        raise ValueError("The registered native model must retain BF16 autocast")
    shadow = copy.deepcopy(native)
    shadow.cfg = replace(native.cfg, bf16=False)
    before, after = asdict(native.cfg), asdict(shadow.cfg)
    if {k for k in before if before[k] != after[k]} != {"bf16"}:
        raise ValueError("The precision comparison may only change the local autocast flag")
    for name, value in native.state_dict().items():
        if not torch.equal(value, shadow.state_dict()[name]):
            raise ValueError("Shadow tensors differ from the native checkpoint")
    if any(p.requires_grad for p in native.parameters()) or any(p.requires_grad for p in shadow.parameters()):
        raise ValueError("Both models must be frozen before the precision comparison")
    return shadow.eval()


def anchor_vjp(step, state, action, context, present, feature_fn):
    action = action.detach().clone().requires_grad_(True)
    features = feature_fn(step(state, action, context)[..., :240])
    weights = torch.cos(torch.arange(features.shape[-1], device=features.device, dtype=features.dtype)+.5)
    gradient = torch.autograd.grad((features*weights).sum(), action)[0]
    if not torch.isfinite(gradient).all():
        raise ValueError("Nonfinite physical VJP")
    return dict(present_joints=[dict(joint=j, autograd=float(gradient[0, j]))
        for j in present.nonzero().flatten().tolist()],
        absent_joint_gradient_exact_zero=bool(torch.equal(gradient[:, ~present],
            torch.zeros_like(gradient[:, ~present]))), command_memory_excluded=True)


def native_reproduction(actual, saved):
    original = {r["joint"]:r["autograd"] for r in saved["present_joints"]}
    found = {r["joint"]:r["autograd"] for r in actual["present_joints"]}
    if (len(original) != len(saved["present_joints"]) or len(found) != len(actual["present_joints"])
            or set(original) != set(found)):
        raise ValueError("Native anchor joint ownership differs")
    errors = {j:abs(found[j]-original[j]) for j in original}
    return dict(passed=actual["absent_joint_gradient_exact_zero"] and all(v == 0. for v in errors.values()),
        exact_reproduction=True, maximum_absolute_error=max(errors.values()), per_joint_absolute_error=errors)


def prediction_preservation(native, shadow):
    metrics = ("joint_position_rad", "body_position_m")
    output = []
    for field, keys, ratio in (("groups", ("origin", "split"), 1.10),
                              ("per_source", ("origin", "split", "clip_id"), 1.25)):
        def indexed(summary):
            rows = [r for r in summary[field] if r["split"] == "validation"]
            values = {tuple(r[k] for k in keys):r for r in rows}
            if len(values) != len(rows):
                raise ValueError("Duplicate validation source in precision comparison")
            return values
        original, candidate = indexed(native), indexed(shadow)
        if not original or set(original) != set(candidate):
            raise ValueError("Precision comparison must preserve all validation sources")
        for key, row in original.items():
            for metric in metrics:
                baseline, value = row["wm"][metric], candidate[key]["wm"][metric]
                if not all(torch.isfinite(torch.tensor(v)) and v >= 0 for v in (baseline, value)):
                    raise ValueError("Nonfinite or negative precision-comparison error")
                bound = ratio*baseline+1e-6
                output.append(dict(scope=field, **dict(zip(keys, key)), metric=metric,
                    native=baseline, shadow=value, maximum=bound, passed=value <= bound))
    return dict(passed=all(r["passed"] for r in output), checks=output,
        validation_origin_mean_ratio=1.10, validation_per_source_ratio=1.25, additive_tolerance=1e-6)
