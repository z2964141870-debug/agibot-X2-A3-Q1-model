#!/usr/bin/env python3
"""Independently recompute J01 evidence from saved arrays, without new physics."""

import argparse
import json
from pathlib import Path

import numpy as np

from supermimic.a3_assets import identity


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def compare_vectors(a, b, floor):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return {"reference_norm": na, "candidate_norm": nb,
            "degenerate_reference": na <= floor, "degenerate_candidate": nb <= floor,
            "cosine": float(np.clip((a * b).sum() / (na * nb), -1, 1)) if na > floor and nb > floor else None,
            "norm_ratio": nb / na if na > floor else None,
            "relative_l2_error": float(np.linalg.norm(b - a)) / max(na, floor),
            "absolute_rmse": float(np.sqrt(np.mean((b - a) ** 2)))}


def same(actual, expected):
    if isinstance(expected, dict):
        check(actual.keys() == expected.keys(), "Metric keys changed")
        for key in expected:
            same(actual[key], expected[key])
    elif isinstance(expected, list):
        check(len(actual) == len(expected), "Metric list length")
        for a, e in zip(actual, expected):
            same(a, e)
    elif isinstance(expected, float):
        check(np.isclose(actual, expected, rtol=1e-10, atol=1e-12), "Numeric evidence mismatch")
    else:
        check(actual == expected, "Evidence value mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads((args.diagnostic / "plan.json").read_text())
    receipt_path = args.diagnostic / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    check(receipt["status"] == "ACTION_RESPONSE_DIAGNOSTIC_COMPLETED", "Diagnostic incomplete")
    check(receipt["plan"] == identity(args.diagnostic / "plan.json"), "Plan changed")
    for item in [plan["entry"], plan["checkpoint"], plan["evaluation"], plan["backend"], *plan["backend_sources"]]:
        check(identity(item["path"]) == item, "Frozen source/input changed")
    check(len(plan["samples"]) == len(receipt["samples"]) == len(receipt["baselines"]) == 8, "Sample count changed")
    check(plan["max_control_steps"] == 4352 and plan["max_physics_substeps"] == 87040, "Budget changed")
    check(receipt["optimizer_updates"] == receipt["new_policy_inference"] == 0 and receipt["wm_float32_unchanged"], "Frozen diagnostic contract")
    rng = np.random.default_rng(20261023)
    steps, metrics, controls_passed = 0, [], []
    groups = {"root": (np.arange(3), 1e-6), "active_joints": (7 + np.asarray(plan["active_indices"]), 1e-5)}
    control_flags = []
    for sample, result, baseline, native_run in zip(plan["samples"], receipt["samples"], receipt["baselines"], receipt["native_runs"]):
        directions = rng.normal(size=(4, 29))
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        check(np.array_equal(directions, np.asarray(sample["directions"])), "Seed/directions changed")
        check(sample["start"] == int(np.floor(sample["fraction"] * (sample["executed_steps"] - 32))), "Sample changed")
        for item in (sample["trace"], sample["reference"], result["raw"], baseline["raw"], native_run["raw"]):
            check(identity(item["path"]) == item, "Raw/input identity changed")
        with np.load(sample["trace"]["path"], allow_pickle=False) as saved, np.load(baseline["raw"]["path"], allow_pickle=False) as base, np.load(result["raw"]["path"], allow_pickle=False) as raw, np.load(native_run["raw"]["path"], allow_pickle=False) as native:
            t = sample["start"]
            check(np.array_equal(base["truth"], saved["state"][t:t + 32]), "Replay target changed")
            check(np.all(np.abs(base["state"].astype(np.float64) - base["truth"]) <= np.array([2e-6] * 42 + [2e-5] * 41)), "Replay precision failed")
            check(bool(baseline["passed"]) and int(base["length"]) == 32 and not base["terminated"].any() and not base["timeouts"].any(), "Replay validity failed")
            check(np.array_equal(raw["saved_actions"], saved["actions"][t:t + 32]), "Fixed actions changed")
            check(t > 0 and np.array_equal(raw["initial_state"], saved["state"][t - 1:t]), "Saved initial state changed")
            check(np.array_equal(raw["previous_action"], saved["actions"][t - 1:t].clip(-20, 20)), "Previous action changed")
            check(raw["native"].shape == raw["model"].shape == (17, 32, 83), "Raw trajectory shape")
            for key in native.files:
                check(np.array_equal(raw[key], native[key], equal_nan=True), "Native/model evidence changed")
            check(np.array_equal(raw["native"][0], base["state"]), "Baseline reuse changed")
            check(np.isfinite(raw["model"]).all(), "Nonfinite model rollout")
            check(np.array_equal(raw["directions64"], directions), "Raw directions changed")
            for i in range(17):
                length = int(raw["lengths"][i])
                check(0 < length <= 32 and np.isfinite(raw["native"][i, :length]).all(), "Native length/finite contract")
                check(np.isnan(raw["native"][i, length:]).all(), "Invalid tail not retained")
                check(not raw["terminated"][i, length:].any() and not raw["timeouts"][i, length:].any(), "Termination tail changed")
                if length < 32:
                    check(raw["terminated"][i, length - 1] or raw["timeouts"][i, length - 1], "Unexplained short variant")
                steps += length
            for item in result["analysis"]["responses"]:
                direction, magnitude, horizon = item["direction"], item["magnitude"], item["horizon"]
                offset = 1 + 4 * direction + (0 if magnitude == .05 else 2)
                minus, plus = offset, offset + 1
                delta = directions[direction].astype(np.float32) * np.float32(magnitude)
                check(np.array_equal(raw["deltas"][minus], -delta) and np.array_equal(raw["deltas"][plus], delta), "Perturbation changed")
                valid = all(int(raw["lengths"][i]) >= horizon and not raw["terminated"][i, :horizon].any() and not raw["timeouts"][i, :horizon].any() for i in (minus, plus))
                check(item["valid_physics_response"] == valid, "Termination exclusion mismatch")
                if not valid:
                    check(item["native_vs_model_fd"] is None and item["native_vs_model_jvp"] is None, "Invalid response scored")
                    metrics.append(item)
                    continue
                selected, floor = groups[item["group"]]
                a = (raw["native"][plus, horizon - 1, selected].astype(np.float64) - raw["native"][minus, horizon - 1, selected].astype(np.float64)) / (2 * magnitude)
                b = (raw["model"][plus, horizon - 1, selected].astype(np.float64) - raw["model"][minus, horizon - 1, selected].astype(np.float64)) / (2 * magnitude)
                hindex = (1, 8, 32).index(horizon)
                same(item["native_vs_model_fd"], compare_vectors(a, b, floor))
                same(item["native_vs_model_jvp"], compare_vectors(a, raw["jvp32"][direction, hindex, selected], floor))
                metrics.append(item)
            for item in result["analysis"]["numerical_controls"]:
                direction, hindex = item["direction"], (1, 8, 32).index(item["horizon"])
                selected, _ = groups[item["group"]]
                ad = raw["jvp64"][direction, hindex, selected]
                ad32 = raw["jvp32"][direction, hindex, selected].astype(np.float64)
                flags = []
                for eindex, reported in enumerate(item["float64_jvp_vs_fd"]):
                    fd = raw["control_fd64"][eindex, direction, hindex, selected]
                    flags.append(bool(np.all(np.abs(fd - ad) <= 1e-7 + 1e-3 * np.abs(ad))))
                    check(flags[-1] == reported["passed"], "Float64 control mismatch")
                flag32 = bool(np.all(np.abs(ad32 - ad) <= 1e-5 + .02 * np.abs(ad)))
                check(flag32 == item["float32_vs_float64_jvp_passed"], "Float32 control mismatch")
                control_flags.append(all(flags) and flag32)
        controls_passed.append({"sample": sample["sample"], "raw_identity_verified": True})
    for horizon, groups_summary in receipt["summary"].items():
        for group, reported in groups_summary.items():
            rows = [x for x in metrics if x["horizon"] == int(horizon) and x["group"] == group]
            check(reported["total"] == len(rows) and reported["valid"] == sum(x["valid_physics_response"] for x in rows), "Summary denominator changed")
            for name in ("native_vs_model_fd", "native_vs_model_jvp"):
                values = [x[name] for x in rows if x[name] is not None]
                cs = [x["cosine"] for x in values if x["cosine"] is not None]
                ratios = [x["norm_ratio"] for x in values if x["norm_ratio"] is not None]
                same(reported[name], {"nondegenerate": len(cs), "median_cosine": float(np.median(cs)) if cs else None,
                                      "positive_cosine_fraction": float(np.mean(np.asarray(cs) > 0)) if cs else None,
                                      "median_norm_ratio": float(np.median(ratios)) if ratios else None,
                                      "median_relative_l2_error": float(np.median([x["relative_l2_error"] for x in values])) if values else None})
    check(receipt["counter"]["control_steps_attempted"] == receipt["counter"]["control_steps_completed"] == steps <= 4352, "Physical budget mismatch")
    check(receipt["counter"]["physics_substeps_completed"] == steps * 20, "Substep budget mismatch")
    check(receipt["all_numerical_controls_passed"] == all(control_flags), "Numerical summary mismatch")
    result = {"status": "ACTION_RESPONSE_EVIDENCE_INDEPENDENTLY_VERIFIED", "diagnostic": identity(receipt_path),
              "plan": identity(args.diagnostic / "plan.json"), "entry": identity(__file__),
              "control_steps": steps, "physics_substeps": steps * 20, "response_rows": len(metrics),
              "numerical_control_rows": len(control_flags), "all_numerical_controls_passed": all(control_flags),
              "samples": controls_passed, "new_physics_steps": 0, "optimizer_updates": 0}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
