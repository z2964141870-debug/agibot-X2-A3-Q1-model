"""Frozen identities and gates for the Primebot-first Q1 experiment."""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path("/opt/data/private/ZHY")
HERE = Path(__file__).resolve().parent
UPSTREAM = "118338b19dcd14f768d1e7de5a48d04b78987688"
MORPH_UPSTREAM = "41c8382f127a6f5109bfe047445a9180f71f1fc0"
FIXED = ("walk_forward_loop_001__A022", "walk_forward_start_002__A023",
         "walk_forward_stop_002__A024")
CAPACITY = ((4096, 2048, 2048), (2048, 1024, 1024), (1024, 512, 512))
TARGET_TRANSITIONS = 410009600


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n")
    temporary.replace(path)


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def capacity_failure(text):
    # Peer transport errors and device faults do not justify a capacity fallback.
    faults = ("unspecified launch failure", "illegal memory access", "device-side assert", "GPU requires reset")
    return not any(f in text for f in faults) and ("CUDA out of memory" in text or "torch.OutOfMemoryError" in text)


def evaluation_gate(pairs, complete_library, complete_media, audit_passed):
    expected = {(name, seed) for name in FIXED for seed in (0, 1, 2)}
    keys = [(p["clip"], p["seed"]) for p in pairs]
    reasons = []
    if len(keys) != 9 or set(keys) != expected:
        reasons.append("missing_or_duplicate_fixed_cases")
    if any(not p.get("final_complete") for p in pairs):
        reasons.append("fixed_final_completion_below_9_of_9")
    valid = bool(pairs) and all(p.get("common_steps", 0) > 0 for p in pairs)
    scores = {}
    for metric in ("joint_rmse_rad", "body_relative_rmse_m"):
        values = [[p.get(stage + "_" + metric) for p in pairs]
                  for stage in ("initial", "final")]
        if not valid or any(v is None or not math.isfinite(v) or v < 0
                            for column in values for v in column):
            reasons.append("missing_comparable_" + metric)
            continue
        initial, final = (sum(column) / len(column) for column in values)
        scores[metric] = dict(initial=initial, final=final)
        if not final < initial:
            reasons.append("no_improvement_" + metric)
    for name, passed in (("library_evaluation", complete_library),
                         ("media_delivery", complete_media), ("checkpoint_audit", audit_passed)):
        if not passed:
            reasons.append("incomplete_" + name)
    return dict(passed=not reasons, gate_reason=reasons or ["run_through_and_tracking_improved"],
                paired_metrics=scores, natural_gait_thresholds_are_diagnostic=True)


def morph_settings(environments):
    require(environments in (1024, 512), "Only the two approved shared capacity tiers")
    factor = 1024 // environments
    return dict(num_envs=environments, wm_batch=environments // 2,
                imag_batch=environments, kp_imag_batch=environments,
                iterations=160000 * factor, wm_only_iters=2000 * factor,
                warmup_env_steps=400 * factor, ckpt_every=16000 * factor)


def matched_checkpoint(rows):
    require(bool(rows), "No audited checkpoint is available")
    row = min(rows, key=lambda r: abs(r["global_env_steps"] - TARGET_TRANSITIONS))
    difference = abs(row["global_env_steps"] - TARGET_TRANSITIONS) / TARGET_TRANSITIONS
    require(difference <= .01, "No checkpoint within one percent of the Q1 exposure target")
    return dict(**row, exposure_relative_difference=difference)
