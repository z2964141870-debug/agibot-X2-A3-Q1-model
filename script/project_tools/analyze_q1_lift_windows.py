"""Pair saved collision-sole height and native force with reference lift windows."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from prepare_q1_gait_review import identity, write_csv

DT = .02
LIFT_M = .02
FORCE_N = 5.
SIDES = ("left", "right")


def read_series(path, keys, values, *, first_tick, reference_only=False):
    result = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {*keys, *values, "seconds"}
        if reference_only:
            required.add("case")
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing saved time-series fields: " + str(path))
        for row in reader:
            if reference_only and row["case"] != "reference":
                continue
            key = tuple(row[name] for name in keys)
            if row["side"] not in SIDES or any(not part for part in key):
                raise ValueError("Invalid clip, label, repeat or side")
            seconds = float(row["seconds"])
            if not math.isfinite(seconds):
                raise ValueError("Nonfinite saved timestamp")
            tick = round(seconds / DT)
            if tick < first_tick or abs(seconds - tick * DT) > 1e-8:
                raise ValueError("Expected an exact saved 20ms time axis")
            sample = {name: float(row[name]) for name in values}
            if not all(math.isfinite(value) for value in sample.values()):
                raise ValueError("Nonfinite saved measurement")
            if "declared_stable_contact" in sample and sample["declared_stable_contact"] not in (0., 1.):
                raise ValueError("Declared stable contact must be boolean")
            group = result.setdefault(key, {})
            if tick in group:
                raise ValueError("Duplicate clip/side/time sample")
            group[tick] = sample
    if not result:
        raise ValueError("No selected saved time series")
    for group in result.values():
        if sorted(group) != list(range(first_tick, max(group) + 1)):
            raise ValueError("Gapped saved time axis")
    return result


def windows(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def measure(height, reference, force, mask):
    count = int(mask.sum())
    if not count:
        return dict(frames=0, actual_height_p95_m=None, reference_height_p95_m=None,
                    height_rmse_m=None, height_bias_m=None, native_loaded_fraction=None,
                    near_floor5mm_fraction=None)
    actual, target, normal = height[mask], reference[mask], force[mask]
    return dict(frames=count, actual_height_p95_m=float(np.quantile(actual, .95)),
                reference_height_p95_m=float(np.quantile(target, .95)),
                height_rmse_m=float(np.sqrt(np.mean((actual-target)**2))),
                height_bias_m=float(np.mean(actual-target)),
                native_loaded_fraction=float(np.mean(normal > FORCE_N)),
                near_floor5mm_fraction=float(np.mean(actual <= .005)))


def analyze(actual, reference, support, baseline_label):
    scenarios = sorted({key[:3] for key in actual})
    clips = {key[2] for key in actual}
    baseline_pairs = {(repeat, clip) for label, repeat, clip in scenarios
                      if label == baseline_label}
    if not baseline_pairs:
        raise ValueError("Missing matching baseline repeat/clip")
    for label, repeat, clip in scenarios:
        if any((label, repeat, clip, side) not in actual for side in SIDES):
            raise ValueError("Each saved scenario requires both feet")
        if len(actual[label, repeat, clip, "left"]) != len(actual[label, repeat, clip, "right"]):
            raise ValueError("Left/right physical durations differ")
        if any((baseline_label, repeat, clip, side) not in actual for side in SIDES):
            raise ValueError("Missing matching baseline repeat/clip")
    for label in {key[0] for key in actual}:
        if {(repeat, clip) for name, repeat, clip in scenarios if name == label} != baseline_pairs:
            raise ValueError("Each label requires the same baseline repeat/clip pairs")
    common = {clip: min(len(series) for key, series in actual.items() if key[2] == clip)
              for clip in clips}
    for clip in clips:
        for side in SIDES:
            if (clip, side) not in reference or (clip, side) not in support:
                raise ValueError("Reference clip/side identity missing")
            ref, phase = reference[clip, side], support[clip, side]
            if len(phase) != len(ref) + 1 or max(phase) != max(ref):
                raise ValueError("Expected reference f+1 and phase f0..end alignment")
            if any(len(series) > len(ref) for key, series in actual.items()
                   if key[2:] == (clip, side)):
                raise ValueError("Physical duration exceeds its reference")
    rows, events, support_rows = [], [], []
    for label, repeat, clip in scenarios:
        count = common[clip]
        ticks = range(1, count+1)
        forces = []
        for side in SIDES:
            series = actual[label, repeat, clip, side]
            height = np.array([series[t]["sole_height_m"] for t in ticks])
            force = np.array([series[t]["native_force_N"] for t in ticks])
            target = np.array([reference[clip, side][t]["sole_height_m"] for t in ticks])
            # Physical poses and reference soles are post-step f+1; phase CSV also
            # contains reset f0. Pair by integer tick, never by CSV row number.
            stable = np.array([support[clip, side][t]["declared_stable_contact"] for t in ticks], dtype=bool)
            lift = target > LIFT_M
            base = dict(label=label, repeat=repeat, clip=clip, side=side,
                        common_prefix_frames=count, available_physical_frames=len(series))
            for name, mask in (("all", np.ones(count, dtype=bool)),
                               ("declared_stable", stable),
                               ("not_declared_stable", ~stable),
                               ("reference_sole_above20mm", lift)):
                rows.append(dict(**base, phase=name, **measure(height, target, force, mask)))
            for number, (start, end) in enumerate(windows(lift)):
                peak = float(height[start:end].max())
                target_peak = float(target[start:end].max())
                events.append(dict(**base, window=number, start_seconds=(int(start)+1)*DT,
                    end_exclusive_seconds=(int(end)+1)*DT,
                    duration_seconds=(int(end)-int(start))*DT,
                    touches_prefix_start=bool(start == 0),
                    touches_prefix_end=bool(end == count),
                    actual_peak_m=peak, reference_peak_m=target_peak,
                    peak_shortfall_m=target_peak-peak,
                    within_window_peak_lag_seconds=float(
                        (height[start:end].argmax()-target[start:end].argmax())*DT),
                    **measure(height, target, force,
                              (np.arange(count) >= start) & (np.arange(count) < end))))
            forces.append(force > FORCE_N)
        left, right = forces
        code = left.astype(int) + 2*right.astype(int)
        single = code[(code == 1) | (code == 2)]
        support_rows.append(dict(label=label, repeat=repeat, clip=clip, common_prefix_frames=count,
            final_substep_both_loaded_fraction=float(np.mean(left & right)),
            final_substep_left_only_fraction=float(np.mean(left & ~right)),
            final_substep_right_only_fraction=float(np.mean(~left & right)),
            final_substep_neither_loaded_fraction=float(np.mean(~left & ~right)),
            raw_single_loaded_side_changes=int(np.count_nonzero(np.diff(single)))))
    lookup = {(r["label"], r["repeat"], r["clip"], r["side"], r["phase"]): r for r in rows}
    comparisons = []
    for row in rows:
        if row["label"] == baseline_label:
            continue
        baseline = lookup[baseline_label, row["repeat"], row["clip"], row["side"], row["phase"]]
        changes = {name+"_delta": None if row[name] is None else row[name]-baseline[name]
                   for name in ("actual_height_p95_m", "height_rmse_m", "native_loaded_fraction")}
        comparisons.append(dict(label=row["label"], baseline_label=baseline_label,
            repeat=row["repeat"], clip=row["clip"], side=row["side"], phase=row["phase"],
            common_prefix_frames=row["common_prefix_frames"], **changes))
    return rows, events, support_rows, comparisons


def shared_source(summaries, basename):
    identities = []
    for summary in summaries:
        matches = [row for row in summary["sources"] if Path(row["path"]).name == basename]
        if len(matches) != 1:
            raise ValueError("Missing unique source identity: " + basename)
        identities.append(matches[0])
    content = {name: identities[0][name] for name in ("bytes", "sha256")}
    if any({name: row[name] for name in content} != content for row in identities[1:]):
        raise ValueError("Saved geometry/phase source identity differs: " + basename)
    return dict(**content, recorded_paths=sorted({row["path"] for row in identities}))


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    inputs = [args.actual, args.reference, args.support]
    receipts = [path.parent / "summary.json" for path in inputs]
    before = [identity(path) for path in inputs + receipts]
    summaries = [json.loads(path.read_text()) for path in receipts]
    if [row["status"] for row in summaries] != ["RF16_SAVED_FIXED_SOLE_REVIEW_READY",
            "SAVED_POSE_SOLE_GEOMETRY_AUDITED", "SAVED_REFERENCE_SUPPORT_AUDITED"]:
        raise ValueError("Unexpected saved evidence receipts")
    scene = shared_source(summaries, "scene.xml")
    library = shared_source(summaries, "evaluation_library.json")
    actual = read_series(args.actual, ("label", "repeat", "clip", "side"),
                         ("sole_height_m", "native_force_N"), first_tick=1)
    reference = read_series(args.reference, ("clip", "side"), ("sole_height_m",),
                            first_tick=1, reference_only=True)
    support = read_series(args.support, ("clip", "side"), ("declared_stable_contact",), first_tick=0)
    tables = analyze(actual, reference, support, args.baseline_label)
    if before != [identity(path) for path in inputs + receipts]:
        raise ValueError("Input evidence changed while reading")
    args.output.mkdir(parents=True, exist_ok=False)
    names = ("phase_feet.csv", "reference_lift_windows.csv", "paired_support.csv", "paired_changes.csv")
    for name, rows in zip(names, tables):
        write_csv(args.output / name, rows)
    summary = dict(status="SAVED_PAIRED_LIFT_WINDOWS_ANALYZED",
        captured_utc=datetime.now(timezone.utc).isoformat(), sources=before,
        common_scene=scene, common_library=library, tool=identity(Path(__file__)),
        baseline_label=args.baseline_label, rows={name: len(rows) for name, rows in zip(names, tables)},
        diagnostic_lift_threshold_m=LIFT_M, diagnostic_force_threshold_N=FORCE_N,
        new_integration_steps=0, policy_inference_calls=0, optimizer_updates=0,
        user_accepted_checkpoint=None, physical_success_claimed=False,
        limits=["All stages/repeats share the shortest saved prefix per clip; failure suffixes are excluded.",
                "Reference sole above20mm defines a geometric window, not a certified swing phase or step count.",
                "Not-declared-stable includes neutral phase transitions; it is not an RF16 swing label.",
                "Native force is the final 5ms substep; unloaded snapshots do not establish full-step load transfer.",
                "Window boundary flags identify prefix edges; peaks/lag are restricted to that saved window.",
                "Source receipts share library/scene identities; local CSV before/after SHA is recorded, not a new historical trace audit.",
                "These are RF16 composition/head trajectories; they do not isolate the pure SuperMimic teacher's cause.",
                "No new data acceptance gate, reward change or training effect is claimed."])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: summary[key] for key in ("status", "rows", "new_integration_steps")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("actual", "reference", "support", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--baseline-label", required=True)
    run(parser.parse_args())
