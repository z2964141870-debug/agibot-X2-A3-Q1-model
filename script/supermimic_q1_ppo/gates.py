"""Pure, explicit rules for the user-authorized RF12 continuation budget."""

from __future__ import annotations

import math

MAX_TOTAL_SECONDS = 28800.
CHECKPOINTS = (3500., 7100., 10700.)
EXTRA_SECONDS = 18000.


def ratio(value, baseline):
    if not all(isinstance(x, (float, int)) and math.isfinite(x) and x >= 0 for x in (value, baseline)):
        return None
    return value / max(baseline, 1e-12)


def classify_repeats(comparisons):
    if len(comparisons) != 3:
        raise ValueError("Exactly three physical repeats are required")
    decisions = []
    for comparison in comparisons:
        r = comparison["ratios"]
        limits = {"upper_rmse": .85, "wrist_relative_rmse": .85, "raised_peak_error": .75,
                  "root_xy_rmse": 1.10, "leg_rmse": 1.10,
                  "target_oscillation": 1.20, "loaded_slip": 1.20}
        checks = {key: r.get(key) is not None and math.isfinite(r[key]) and r[key] <= limit
                  for key, limit in limits.items()}
        checks["validation_completion"] = comparison["completion"] >= .98
        checks["validation_failures"] = comparison["failures"] <= 1
        checks["fixed_demos_complete"] = comparison["fixed_demos_complete"]
        severe = (comparison["failures"] >= 6
                  or any(r.get(key) is not None and r[key] >= 1.25 for key in ("root_xy_rmse", "upper_rmse")))
        decisions.append({"pass": all(checks.values()), "checks": checks, "severe": severe})
    passed = sum(row["pass"] for row in decisions)
    all_demos = all(row["fixed_demos_complete"] for row in comparisons)
    return {"extension_qualified": passed >= 2 and all_demos,
            "severe_regression": sum(row["severe"] for row in decisions) >= 2,
            "passing_repeats": passed, "all_three_fixed_demos_complete": all_demos,
            "per_repeat": decisions}


def next_phase(phase, elapsed, gate, extension_deadline=None, severe_streak=0):
    if elapsed >= MAX_TOTAL_SECONDS:
        return {"action": "stop", "reason": "absolute_training_budget"}
    if phase == "C":
        streak = severe_streak + 1 if gate["severe_regression"] else 0
        if streak >= 2:
            return {"action": "stop", "reason": "two_extension_regressions", "severe_streak": streak}
        if elapsed >= extension_deadline - 60:
            return {"action": "stop", "reason": "extension_budget_completed", "severe_streak": streak}
        return {"action": "continue", "phase": "C", "deadline": extension_deadline, "severe_streak": streak}
    if gate["severe_regression"]:
        return {"action": "stop", "reason": "physical_regression"}
    if gate["extension_qualified"]:
        return {"action": "continue", "phase": "C", "deadline": min(MAX_TOTAL_SECONDS, elapsed + EXTRA_SECONDS),
                "severe_streak": 0}
    if elapsed >= CHECKPOINTS[-1]:
        return {"action": "stop", "reason": "three_hours_without_qualification"}
    return {"action": "continue", "phase": "B", "deadline": CHECKPOINTS[-1], "severe_streak": 0}
