#!/usr/bin/env python3
"""Independently recompute J01 response statistics from saved arrays."""

import argparse
import json
from pathlib import Path

import numpy as np

from supermimic.q1_assets import identity


def check_identity(record):
    if identity(record["path"]) != record:
        raise ValueError("Saved input identity changed")


def response_metrics(actual, predicted, threshold):
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(predicted, dtype=np.float64)
    na, nb = np.sqrt(a @ a), np.sqrt(b @ b)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite saved response")
    return {"physics_norm": float(na), "model_norm": float(nb),
            "cosine": float(np.clip(a @ b / na / nb, -1, 1))
            if min(na, nb) > threshold else None,
            "norm_ratio": float(nb / na) if na > threshold else None,
            "relative_l2_error": float(np.linalg.norm(b - a) / max(na, threshold))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = json.loads(args.input.read_text())
    check_identity(receipt["plan"])
    plan = json.loads(Path(receipt["plan"]["path"]).read_text())
    if receipt["status"] != "Q1_ACTION_RESPONSE_DIAGNOSTIC_COMPLETED":
        raise ValueError("J01 did not complete")
    results, all_responses = [], []
    max_error = 0.
    controls = 0
    for sample, entry in zip(plan["samples"], receipt["samples"], strict=True):
        check_identity(entry["raw"])
        check_identity(sample["trace"])
        with np.load(entry["raw"]["path"], allow_pickle=False) as data:
            raw = {k: data[k].copy() for k in data.files}
        with np.load(sample["trace"]["path"], allow_pickle=False) as data:
            baseline = data["state"][sample["start"]:sample["start"] + 32]
        if not np.array_equal(raw["native"][0], baseline):
            raise AssertionError("Saved unperturbed baseline differs")
        if not np.allclose(raw["directions"], sample["directions"], atol=0, rtol=0):
            raise AssertionError("Directions differ from preregistration")
        if not np.array_equal(raw["failures"], raw["termination_masks"].any(axis=1)):
            raise AssertionError("Termination masks disagree")
        controls += int(raw["lengths"].sum())
        for row in entry["responses"]:
            d, m, h = row["direction"], row["magnitude"], row["horizon"]
            hi = (1, 8, 32).index(h)
            minus = 1 + d * 4 + (0 if m == .05 else 2)
            plus = minus + 1
            expected_delta = raw["directions"][d] * m
            if not (np.allclose(raw["delta"][plus], expected_delta, atol=1e-8)
                    and np.allclose(raw["delta"][minus], -expected_delta, atol=1e-8)):
                raise AssertionError("Perturbation differs from preregistration")
            valid = (min(raw["lengths"][[minus, plus]]) >= h
                     and not raw["termination_masks"][[minus, plus], :h].any())
            if bool(valid) != row["valid"]:
                raise AssertionError("Response validity differs")
            if not valid:
                if row["model_jvp"] is not None or row["model_fd"] is not None:
                    raise AssertionError("Invalid response was scored")
                continue
            ids, floor = (slice(0, 3), 1e-6) if row["group"] == "root" else (slice(7, 29), 1e-5)
            physical = (raw["native"][plus, h - 1].astype(float)
                        - raw["native"][minus, h - 1].astype(float)) / (2 * m)
            fd = (raw["model"][plus, hi].astype(float)
                  - raw["model"][minus, hi].astype(float)) / (2 * m)
            recomputed = {}
            for key, predicted in (("model_fd", fd), ("model_jvp", raw["jvp"][d, hi])):
                calculated = response_metrics(physical[ids], predicted[ids], floor)
                for field, value in calculated.items():
                    expected = row[key][field]
                    if value is None or expected is None:
                        if value is not expected:
                            raise AssertionError("Weak-response suppression differs")
                    else:
                        error = abs(value - expected)
                        max_error = max(max_error, error)
                        if error > 1e-10 * max(1., abs(expected)):
                            raise AssertionError("Saved response arithmetic differs")
                recomputed[key] = calculated
            all_responses.append({**row, **recomputed})
        results.append({"sample": entry["sample"], "baseline_exact": True,
                        "raw": entry["raw"], "control_steps": int(raw["lengths"].sum())})
    summary = {}
    for key, expected in receipt["summary"].items():
        horizon, group = key.split("_")
        selected = [r["model_jvp"] for r in all_responses
                    if r["horizon"] == int(horizon[1:]) and r["group"] == group]
        cos = [r["cosine"] for r in selected if r["cosine"] is not None]
        ratios = [r["norm_ratio"] for r in selected if r["norm_ratio"] is not None]
        actual = {"total": sum(r["horizon"] == int(horizon[1:]) and r["group"] == group
                              for entry in receipt["samples"] for r in entry["responses"]),
                  "valid": len(selected), "nondegenerate": len(cos),
                  "median_cosine": float(np.median(cos)),
                  "positive_cosine_fraction": float(np.mean(np.asarray(cos) > 0)),
                  "median_norm_ratio": float(np.median(ratios)),
                  "median_relative_l2_error": float(np.median([r["relative_l2_error"] for r in selected]))}
        if any(not np.isclose(actual[k], v, atol=1e-10, rtol=1e-10) for k, v in expected.items()):
            raise AssertionError("Summary differs from raw response arithmetic")
        summary[key] = actual
    if controls != receipt["control_steps"] or controls != plan["max_control_steps"]:
        raise AssertionError("J01 control count differs")
    result = {"status": "SAVED_ACTION_RESPONSE_AUDIT_PASSED", "receipt": identity(args.input),
              "source": identity(__file__), "samples": results, "response_rows": len(all_responses),
              "max_scalar_abs_error": max_error, "summary": summary,
              "new_physics_steps": 0, "new_network_calls": 0, "optimizer_updates": 0,
              "limits": "Arithmetic and identity audit of saved arrays; numerical model controls are not re-executed.",
              "storage_status": "LOCAL_ONLY"}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
