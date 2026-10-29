"""Calibrate height reward scales on saved soles and replay the new physical gate."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "project_tools"))
from analyze_q1_lift_windows import read_series, shared_source
from prepare_q1_gait_review import identity, write_csv
from physical_gate import FIXED, evaluate

WIDTHS_M = (.01, .02, .04)
BASE_WEIGHTS = (.10, .25, .50)


def phase_labels(contact, ticks):
    contact, ticks = np.asarray(contact), np.asarray(ticks)
    if contact.dtype != np.bool_ or contact.ndim != 2 or contact.shape[1] != 2 or len(contact) < 2:
        raise ValueError("Expected original boolean two-foot contact intentions")
    if ticks.dtype.kind not in "iu" or ticks.ndim != 1 or (ticks < 0).any():
        raise ValueError("Expected nonnegative integer 50Hz ticks")
    lo = np.minimum(ticks * 3 // 5, len(contact) - 1)
    hi = np.minimum((ticks * 3 + 4) // 5, len(contact) - 1)
    stance = contact[lo] & contact[hi]
    swing = ~(contact[lo] | contact[hi])
    return stance, swing


def scale_statistics(error, mask, width):
    error, mask = np.asarray(error), np.asarray(mask)
    if (error.shape != mask.shape or error.ndim != 2 or error.shape[1] != 2
            or mask.dtype != np.bool_ or not np.isfinite(error).all()
            or not np.isfinite(width) or width <= 0):
        raise ValueError("Expected finite two-foot errors and exact phase masks")
    active = mask.any(-1)
    if not active.any():
        return dict(foot_frames=0, rmse_m=None, score_on_active_frames=None,
                    perfect_gain_on_active_frames=None, absolute_gradient_per_m=None,
                    gradient_below_1_per_m_fraction=None)
    score = np.exp(-np.square(error / width))
    derivative = 2 * np.abs(error) / width**2 * score
    count = np.maximum(mask.sum(-1), 1)
    return dict(foot_frames=int(mask.sum()), rmse_m=float(np.sqrt(np.square(error[mask]).mean())),
                score_on_active_frames=float(((score * mask).sum(-1) / count)[active].mean()),
                perfect_gain_on_active_frames=float((((1 - score) * mask).sum(-1) / count)[active].mean()),
                absolute_gradient_per_m=float(derivative[mask].mean()),
                gradient_below_1_per_m_fraction=float((derivative[mask] < 1.).mean()))


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    actual_path = args.trial / "review/sole_v01/fixed_actual_sole_timeseries.csv"
    source_paths = [actual_path, args.reference, args.library, Path(__file__), HERE / "physical_gate.py",
                    actual_path.parent / "summary.json", args.reference.parent / "summary.json"]
    receipts = [json.loads(path.read_text()) for path in source_paths[-2:]]
    if receipts[0]["status"] != "RF16_SAVED_FIXED_SOLE_REVIEW_READY" or receipts[1]["status"] != "SAVED_POSE_SOLE_GEOMETRY_AUDITED":
        raise ValueError("Unexpected completed saved-geometry receipts")
    library_identity = shared_source(receipts, "evaluation_library.json")
    scene_identity = shared_source(receipts, "scene.xml")
    if {key: identity(args.library)[key] for key in ("bytes", "sha256")} != {
            key: library_identity[key] for key in ("bytes", "sha256")}:
        raise ValueError("Saved sole evidence uses a different original library")
    library = json.loads(args.library.read_text())
    source_rows = [row for rows in library["splits"].values() for row in rows]
    by_name = {row["name"]: row for row in source_rows}
    if len(by_name) != 330 or len(source_rows) != 330:
        raise ValueError("Expected original unique330 library")
    metadata = {}
    for name in FIXED:
        row = by_name[name]
        path = Path(row["path"])
        meta = path.with_suffix(".json")
        value = json.loads(meta.read_text())
        if identity(path)["sha256"] != row["sha256"] or value["output_sha256"] != row["sha256"]:
            raise ValueError("Original CSV or metadata identity differs")
        if value["fps"] != 30. or len(value["contact"]) != value["frames"]:
            raise ValueError("Expected original30Hz contact intentions")
        metadata[name] = np.asarray(value["contact"])
        phase_labels(metadata[name], np.array([0]))
        source_paths.extend((path, meta))
    review_paths = [args.trial / "review" / stage / "fixed_demos.csv"
                    for stage in ("initial", "stage_01", "stage_02", "training_terminal")]
    source_paths.extend(review_paths)
    before = [identity(path) for path in source_paths]
    actual = read_series(actual_path, ("label", "repeat", "clip", "side"),
                         ("sole_height_m", "native_force_N"), first_tick=1)
    reference = read_series(args.reference, ("clip", "side"), ("sole_height_m",),
                            first_tick=1, reference_only=True)
    demos = {}
    for path in review_paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                key = (row["label"], row["repeat"], row["clip"])
                if key in demos:
                    raise ValueError("Duplicate saved fixed-review row")
                demos[key] = row
    metrics, scales = [], []
    scenarios = sorted({key[:3] for key in actual})
    for key in scenarios:
        label, repeat, clip = key
        if clip not in FIXED or any((*key, side) not in actual for side in ("left", "right")):
            raise ValueError("Expected both feet for each original fixed clip")
        count = len(actual[(*key, "left")])
        if len(actual[(*key, "right")]) != count:
            raise ValueError("Saved left/right physical durations differ")
        demo = demos[key]
        if int(demo["steps"]) != count:
            raise ValueError("Saved sole and original reviewer duration differ")
        ticks = np.arange(1, count + 1)
        heights = np.array([[actual[(*key, side)][tick]["sole_height_m"]
                             for side in ("left", "right")] for tick in ticks])
        target = np.array([[reference[clip, side][tick]["sole_height_m"]
                            for side in ("left", "right")] for tick in ticks])
        force = np.array([[actual[(*key, side)][tick]["native_force_N"]
                           for side in ("left", "right")] for tick in ticks])
        stance, swing = phase_labels(metadata[clip], ticks)
        error = heights - target
        frame_count = int(swing.sum())
        metric = dict(demo, swing_sole_foot_frames=frame_count,
                      swing_sole_phase_sha256=hashlib.sha256(swing.tobytes()).hexdigest(),
                      swing_sole_z_rmse_m=float(np.sqrt(np.square(error[swing]).mean())) if frame_count else None,
                      declared_swing_loaded_fraction=float((force[swing] > 5.).mean()) if frame_count else None,
                      declared_swing_below5mm_fraction=float((heights[swing] <= .005).mean()) if frame_count else None)
        metrics.append(metric)
        for phase_name, mask in (("declared_swing", swing), ("declared_stance", stance)):
            for width in WIDTHS_M:
                values = scale_statistics(error, mask, width)
                for weight in BASE_WEIGHTS:
                    gain = values["perfect_gain_on_active_frames"]
                    scales.append(dict(label=label, repeat=repeat, clip=clip, phase=phase_name,
                        width_m=width, base_weight=weight, **values,
                        base_perfect_gain_per20ms=None if gain is None else .02 * weight * gain,
                        ppo_z_perfect_gain_per20ms=None if gain is None else .01 * gain,
                        analytic_only=True))
    labels = sorted({row["label"] for row in metrics})
    if labels != ["initial_static075", "stage_01", "stage_02", "stage_03"] or len(metrics) != 36:
        raise ValueError("Expected four RF16 phases and36 complete fixed-review groups")
    baseline = [row for row in metrics if row["label"] == "initial_static075"]
    replay = {label: evaluate(baseline, [row for row in metrics if row["label"] == label])
              for label in labels if label != "initial_static075"}
    if before != [identity(path) for path in source_paths]:
        raise ValueError("Saved calibration source changed during analysis")
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "fixed_goal_metrics.csv", metrics)
    write_csv(args.output / "scale_grid.csv", scales)
    (args.output / "gate_replay.json").write_text(json.dumps(replay, indent=2, allow_nan=False) + "\n")
    result = dict(status="GAIT_GOAL_SAVED_SCALE_CALIBRATION_READY", captured_utc=datetime.now(timezone.utc).isoformat(),
                  sources=before, common_scene=scene_identity, common_library=library_identity,
                  fixed_groups=len(metrics), saved_states=sum(int(row["steps"]) for row in metrics),
                  scale_grid_rows=len(scales), widths_m=WIDTHS_M, base_weights=BASE_WEIGHTS,
                  phase_mapping="Exact rational3/5, both on=stance, both off=swing, mixed=neutral; post-step f+1",
                  new_integration_steps=0, policy_inference_calls=0, optimizer_updates=0,
                  user_accepted_checkpoint=None, physical_improvement_claimed=False,
                  limitations=["A new score for identical saved poses is not new walking improvement.",
                      "Reward gain is an analytic perfect-height counterfactual, not a realizable action or marginal control cost.",
                      "Declared retarget phases are intentions; native force is the final5ms substep, not a20ms impulse.",
                      "The historical initial trace lacks a historical SHA; saved geometry receipts and current CSV before/after identity are retained.",
                      "Scale grid includes stance-height diagnostics; final candidate coefficients must be explicitly declared before training.",
                      "Numeric fixed gate does not certify full330 generalization, actual clearance, media review or user adoption."])
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "fixed_groups", "saved_states", "scale_grid_rows")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("trial", "reference", "library", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
