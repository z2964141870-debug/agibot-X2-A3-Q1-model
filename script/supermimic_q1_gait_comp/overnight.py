"""Explicit one-night authority, deadlines and conservative PPO rate bounds."""

from __future__ import annotations

from datetime import datetime, timezone
import math

OPTIMIZATION_DEADLINE = datetime(2026, 10, 28, 23, 0, tzinfo=timezone.utc).timestamp()
DELIVERY_DEADLINE = datetime(2026, 10, 29, 1, 0, tzinfo=timezone.utc).timestamp()
MAXIMUM_TRAINING_SECONDS = 14400.
CHECKS = (1200., 2400., 3500., 7100., 10700., 14300.)


def bounded_learning_rate(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Learning rate must be finite and positive")
    return min(5e-5, max(1e-5, float(value)))


def next_check(total):
    return next((value for value in CHECKS if value > total), None)


def continuing_evidence(repeats):
    """At each hour retain both walks and improve at least one foot objective."""
    passed = []
    for repeat in repeats:
        walks = repeat.get("walking_ratios", [])
        required = ("walking_slip", "swing_error", "walking_tilt", "target_change")
        valid = len(walks) == 2 and all(
            isinstance(w.get(k), (int, float)) and math.isfinite(w[k])
            for w in walks for k in required)
        ok = valid and all(
            min(w["walking_slip"], w["swing_error"]) <= .90
            and max(w["walking_slip"], w["swing_error"], w["walking_tilt"], w["target_change"]) <= 1.10
            for w in walks)
        ratios = repeat.get("ratios", {})
        ok = ok and all(isinstance(ratios.get(k), (int, float))
                        and math.isfinite(ratios[k]) and ratios[k] <= 1.15
                        for k in ("upper_rmse", "wrist_rmse", "raised_peak_error"))
        drift, previous = repeat.get("stationary_drift_m"), repeat.get("baseline_stationary_drift_m")
        ok = ok and isinstance(drift, (int, float)) and isinstance(previous, (int, float))
        ok = ok and math.isfinite(drift) and math.isfinite(previous) and drift <= max(.10, previous*1.10)
        ok = ok and repeat.get("validation_failures", 999) <= 1 and repeat.get("validation_completion", 0.) >= .98
        ok = ok and repeat.get("fixed_demos_complete", False) and repeat.get("walking_progress_preserved", False)
        passed.append(bool(ok))
    return {"passed": passed, "continue_authorized": len(repeats) == 3 and sum(passed) >= 2
            and all(r.get("fixed_demos_complete", False) for r in repeats),
            "protocol": "Both fixed walks each improve slip or swing >=10%, preserve other gait metrics <=10%, upper/wrist/peak <=15%; all fixed demos complete in3 repeats; this overnight authority ends07:00 Beijing"}
