"""Predeclared RF15 extension decisions from actual saved physical traces."""

from __future__ import annotations

import numpy as np


def relative(current, baseline):
    if current is None or baseline is None or not np.isfinite(current) or not np.isfinite(baseline):
        return None
    if abs(baseline) < 1e-8:
        return 1. if abs(current) < 1e-8 else None
    return float(current / baseline)


def gate(repeats):
    passed, severe = [], []
    for repeat in repeats:
        ratios = repeat["ratios"]
        finite = all(v is not None and np.isfinite(v) for v in ratios.values())
        ok = finite and ratios["walking_slip"] <= .90 and ratios["swing_error"] <= .90
        ok = ok and ratios["walking_tilt"] <= 1.10 and ratios["target_change"] <= 1.10
        ok = ok and ratios["upper_rmse"] <= 1.15 and ratios["wrist_rmse"] <= 1.15
        ok = ok and ratios["raised_peak_error"] <= 1.15 and repeat["stationary_drift_m"] <= max(.10, repeat["baseline_stationary_drift_m"] * 1.10)
        ok = ok and repeat["validation_failures"] <= 1 and repeat["validation_completion"] >= .98
        ok = ok and repeat["fixed_demos_complete"] and repeat["walking_progress_preserved"]
        bad = not finite or repeat["validation_failures"] >= 6
        bad = bad or (ratios["walking_slip"] is not None and ratios["walking_slip"] >= 1.50)
        bad = bad or (ratios["walking_tilt"] is not None and ratios["walking_tilt"] >= 1.25)
        passed.append(bool(ok))
        severe.append(bool(bad))
    full = all(r["fixed_demos_complete"] for r in repeats)
    return {"repeat_passed": passed, "repeat_severe": severe,
            "improvement_confirmed": sum(passed) >= 2 and full,
            "physical_regression": sum(severe) >= 2,
            "root_xy_exact_threshold": None, "maximum_training_seconds": 14400,
            "protocol": "At least2/3 satisfy all physical gains; all fixed demos complete3/3. Missing/nonfinite fails closed."}
