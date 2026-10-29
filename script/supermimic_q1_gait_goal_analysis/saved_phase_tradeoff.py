"""Compare completed GAIT_GOAL01 saved foot phases without policy or integration."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "project_tools"), str(HERE.parent / "supermimic_q1_sole")]
from audit_q1_saved_sole_geometry import foot_geometry, measure
from prepare_q1_gait_review import FIXED, identity, write_csv
from reward_signal import saved_clip


def phases(contact, count):
    contact = np.asarray(contact)
    if contact.dtype != np.bool_ or contact.ndim != 2 or contact.shape[1] != 2 or len(contact) < 2:
        raise ValueError("Expected the immutable two-foot boolean contact schedule")
    ticks = np.arange(count, dtype=np.int64) * 3
    lo = np.minimum(ticks // 5, len(contact) - 1)
    hi = np.minimum((ticks + 4) // 5, len(contact) - 1)
    return contact[lo] & contact[hi], ~(contact[lo] | contact[hi])


def average(values, mask, *, rms=False):
    values, mask = np.asarray(values), np.asarray(mask)
    if values.shape != mask.shape or mask.dtype != np.bool_ or not np.isfinite(values).all():
        raise ValueError("Expected finite values and an aligned boolean mask")
    if not mask.any():
        return None
    return float(np.sqrt(np.mean(values[mask] ** 2)) if rms else values[mask].mean())


def regions(count):
    index = np.arange(count) * 4 // count
    return [("full", np.ones(count, dtype=bool))] + [
        (f"quarter_{quarter + 1}", index == quarter) for quarter in range(4)]


def frame_metrics(value, mask):
    feet = np.broadcast_to(mask[:, None], value["height"].shape)
    loaded = value["force"] > 5.
    swing = feet & value["swing"]
    return dict(frames=int(mask.sum()),
        loaded_foot_frames=int((feet & loaded).sum()),
        swing_foot_frames=int(swing.sum()),
        loaded_slip_mps=average(value["slip"], feet & loaded),
        swing_sole_z_rmse_m=average(value["height"] - value["reference_height"], swing, rms=True),
        tilt_rms_rad=average(value["tilt"], mask, rms=True))


def paired_regions(initial, candidate):
    for key in ("frames", "swing", "stance"):
        np.testing.assert_array_equal(initial[key], candidate[key])
    np.testing.assert_allclose(initial["reference_height"], candidate["reference_height"], atol=2e-6, rtol=2e-5)
    rows = []
    for name, mask in regions(len(initial["tilt"])):
        before, after = frame_metrics(initial, mask), frame_metrics(candidate, mask)
        row = dict(region=name, frames=before["frames"])
        for metric in ("loaded_slip_mps", "swing_sole_z_rmse_m", "tilt_rms_rad"):
            left, right = before[metric], after[metric]
            row.update({metric + "_initial": left, metric + "_candidate": right,
                        metric + "_delta": None if left is None or right is None else right - left,
                        metric + "_ratio": None if left is None or right is None or left == 0 else right / left})
        row.update(initial_loaded_foot_frames=before["loaded_foot_frames"],
                   candidate_loaded_foot_frames=after["loaded_foot_frames"],
                   slip_down_and_tilt_up=bool(before["loaded_slip_mps"] is not None
                       and after["loaded_slip_mps"] is not None
                       and after["loaded_slip_mps"] < before["loaded_slip_mps"]
                       and after["tilt_rms_rad"] > before["tilt_rms_rad"]))
        feet = np.broadcast_to(mask[:, None], initial["force"].shape)
        initial_load, candidate_load = initial["force"] > 5., candidate["force"] > 5.
        common = feet & initial_load & candidate_load
        row.update(common_loaded_foot_frames=int(common.sum()),
                   common_loaded_slip_initial=average(initial["slip"], common),
                   common_loaded_slip_candidate=average(candidate["slip"], common),
                   initial_load_lost_fraction=average(~candidate_load, feet & initial_load),
                   candidate_load_gained_fraction=average(~initial_load, feet & candidate_load))
        rows.append(row)
    return rows


def foot_conditions(value):
    rows = []
    for foot, side in enumerate(("left", "right")):
        height, reference = value["height"][:, foot], value["reference_height"][:, foot]
        loaded = value["force"][:, foot] > 5.
        swing, stance = value["swing"][:, foot], value["stance"][:, foot]
        masks = dict(intent_swing=swing, raised_reference_10mm=swing & (reference >= .01),
                     intent_stance=stance, native_loaded=loaded)
        for condition, mask in masks.items():
            rows.append(dict(side=side, condition=condition, foot_frames=int(mask.sum()),
                sole_z_mean_m=average(height, mask), reference_z_mean_m=average(reference, mask),
                sole_z_p95_m=float(np.quantile(height[mask], .95)) if mask.any() else None,
                height_rmse_m=average(height - reference, mask, rms=True),
                native_loaded_fraction=average(loaded, mask),
                near_floor5mm_fraction=average(height <= .005, mask),
                above10mm_fraction=average(height > .01, mask),
                loaded_slip_mps=average(value["slip"][:, foot], mask & loaded)))
    return rows


def plot(output, saved):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for line in ("base", "ppo"):
        labels = list(dict.fromkeys(key[1] for key in saved if key[0] == line))
        final = labels[-1]
        figure, axes = plt.subplots(4, 2, figsize=(13, 10.5), constrained_layout=True)
        for column, clip in enumerate(FIXED[:2]):
            initial, candidate = (saved[(line, label, 0, clip)] for label in ("initial", final))
            seconds = (np.arange(len(initial["tilt"])) + 1) * .02
            for foot in range(2):
                ax = axes[foot, column]
                ax.plot(seconds, initial["reference_height"][:, foot] * 1000, color="#252525", label="Reference", lw=1.3)
                ax.plot(seconds, initial["height"][:, foot] * 1000, color="#888888", label="Initial", lw=1)
                ax.plot(seconds, candidate["height"][:, foot] * 1000, color="#087f8c", label="After training", lw=1)
                ax.axhline(5, color="#bf553c", ls=":", lw=.8)
                ax.set_ylabel(("Left" if foot == 0 else "Right") + " sole Z (mm)")
            for value, color, label in ((initial, "#888888", "Initial"), (candidate, "#087f8c", "After training")):
                loaded = value["force"] > 5.
                denominator = loaded.sum(-1)
                slip = np.divide((value["slip"] * loaded).sum(-1), denominator,
                                 out=np.full(len(denominator), np.nan), where=denominator > 0)
                axes[2, column].plot(seconds, slip, color=color, lw=1, label=label)
                axes[3, column].plot(seconds, np.degrees(value["tilt"]), color=color, lw=1, label=label)
            axes[0, column].set_title("Stop walk" if column == 0 else "Start walk")
            axes[2, column].set_ylabel("Native loaded slip (m/s)")
            axes[3, column].set_ylabel("Base tilt (degrees)")
            axes[3, column].set_xlabel("Saved control time (s)")
        for ax in axes.flat:
            ax.grid(alpha=.18)
        axes[0, 0].legend(fontsize=8, ncol=3, loc="upper right")
        figure.suptitle(f"Q1 {line.upper()}: saved fixed walks, repeat 0, no new rollout", fontsize=13)
        figure.supxlabel("Whole collision sole, world floor Z=0. Dotted line: 5mm diagnostic. Native force sampled at final 5ms substep.", fontsize=9)
        figure.savefig(output / f"{line}_saved_phase_tradeoff.png", dpi=150)
        figure.savefig(output / f"{line}_saved_phase_tradeoff.pdf")
        plt.close(figure)


def run(args):
    import mujoco

    if args.output.exists():
        raise FileExistsError(args.output)
    dependencies = [Path(__file__), HERE.parent / "supermimic_q1_sole/reward_signal.py",
                    HERE.parent / "project_tools/audit_q1_saved_sole_geometry.py",
                    HERE.parent / "project_tools/prepare_q1_gait_review.py", args.scene]
    saved, tracked, trace_sources, conditions, bins, curves = {}, {}, [], [], [], []
    def remember(paths):
        for path in paths:
            value = identity(path)
            if str(path) in tracked and tracked[str(path)] != value:
                raise ValueError("Analysis dependency changed: " + str(path))
            tracked[str(path)] = value
    remember(dependencies)
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    geometry = [foot_geometry(model, side) for side in ("left", "right")]
    floor = model.geom("floor").id
    if (int(model.geom_type[floor]) != int(mujoco.mjtGeom.mjGEOM_PLANE)
            or not np.allclose(model.geom_pos[floor], 0.)
            or not np.allclose(model.geom_quat[floor], [1., 0., 0., 0.])):
        raise ValueError("Expected the original horizontal world floor")
    for line, trial in (("base", args.base), ("ppo", args.ppo)):
        status_path = trial / "status.json"
        post_path = trial / "postdelivery_status.json"
        library_path, selection_path, plan_path = (trial / name for name in
            ("evaluation/evaluation_library.json", "run/checkpoint_selection.json", "plan.json"))
        remember((status_path, post_path, library_path, selection_path, plan_path))
        status, post = (json.loads(path.read_text()) for path in (status_path, post_path))
        if post["status"] != "COMPLETED" or status["status"] != ("COMPLETED" if line == "base" else "FAILED"):
            raise ValueError("Expected completed saved delivery; PPO original failure stays preserved")
        library = json.loads(library_path.read_text())["splits"]["train"]
        selection = json.loads(selection_path.read_text())
        if len(library) != 330 or len({row["name"] for row in library}) != 330 or selection["user_accepted_checkpoint"] is not None:
            raise ValueError("Original unique330 library and null adoption required")
        frozen = {row["path"]: row for row in json.loads(plan_path.read_text())["source"]}
        asset_paths = [Path(path) for path in frozen if path.startswith(str(args.scene.parent) + "/")]
        for path in asset_paths:
            if identity(path) != frozen[str(path)]:
                raise ValueError("Frozen geometry source changed: " + str(path))
        remember(asset_paths)
        indices = [next(index for index, row in enumerate(library) if row["name"] == name) for name in FIXED]
        contacts = {}
        for index in indices:
            csv_path = Path(library[index]["path"])
            metadata_path = csv_path.with_suffix(".json")
            remember((csv_path, metadata_path))
            metadata = json.loads(metadata_path.read_text())
            if (identity(csv_path)["sha256"] != library[index]["sha256"]
                    or metadata["output_sha256"] != library[index]["sha256"] or metadata["fps"] != 30.):
                raise ValueError("Original fixed reference identity or rate changed")
            for path in (csv_path, metadata_path):
                if identity(path) != frozen[str(path)]:
                    raise ValueError("Frozen fixed reference changed")
            contacts[index] = metadata["contact"]
        for candidate in selection["candidates"]:
            label = candidate["label"]
            for repeat, historical in enumerate(candidate["traces"]):
                trace_path = Path(historical["path"])
                before = identity(trace_path)
                if trace_path != trial / f"evaluation/{label}_repeat{repeat}.npz" or before != historical:
                    raise ValueError("Trace path or historical SHA differs")
                result_path = trace_path.with_suffix(".json")
                result_identity = identity(result_path)
                result = json.loads(result_path.read_text())
                with np.load(trace_path, allow_pickle=False) as archive:
                    reference_steps = archive["reference_steps"]
                    starts = np.r_[0, np.cumsum(reference_steps + 1)[:-1]]
                    arrays = {key: archive[key][:, indices].copy() for key in (
                        "states", "targets", "frames", "alive", "force", "slip", "reference_p", "reference_quat", "reference_q")}
                    arrays.update(steps=archive["steps"][indices], reference_steps=reference_steps[indices])
                if result["label"] != label or result["repeat"] != repeat or len(result["per_clip"]) != 330:
                    raise ValueError("Saved JSON identity or library differs")
                for slot, index in enumerate(indices):
                    clip, entry = FIXED[slot], result["per_clip"][index]
                    fixed = next(row for row in result["goal_fixed"] if row["clip"] == clip)
                    if entry["name"] != clip:
                        raise ValueError("Clip order differs")
                    value = saved_clip(arrays, slot, int(entry["steps"]))
                    count = len(value["frames"])
                    np.testing.assert_array_equal(value["frames"], starts[index] + np.arange(1, count + 1))
                    stance, swing = phases(contacts[index], int(reference_steps[index]) + 1)
                    if hashlib.sha256(swing[1:].tobytes()).hexdigest() != fixed["swing_sole_phase_sha256"]:
                        raise ValueError("Reconstructed intent phase differs from the frozen physical gate")
                    positions = np.concatenate((value["states"][:, :7], value["states"][:, 13:35]), -1)
                    reference_positions = np.concatenate([arrays[key][:count, slot] for key in
                        ("reference_p", "reference_quat", "reference_q")], -1)
                    height, _ = measure(model, data, positions, geometry)
                    reference_height, _ = measure(model, data, reference_positions, geometry)
                    x, y = value["states"][:, 4:6].T
                    value.update(height=height, reference_height=reference_height,
                        tilt=np.arccos(np.clip(1 - 2 * (x*x + y*y), -1, 1)),
                        swing=swing[1:count+1], stance=stance[1:count+1])
                    metrics = frame_metrics(value, np.ones(count, dtype=bool))
                    for measured, original in (("swing_sole_z_rmse_m", "swing_sole_z_rmse_m"),
                                               ("loaded_slip_mps", "loaded_slip_mps"),
                                               ("tilt_rms_rad", "torso_tilt_rms_rad")):
                        if metrics[measured] is not None and fixed[original] is not None:
                            np.testing.assert_allclose(metrics[measured], fixed[original], atol=2e-6, rtol=2e-5)
                    key = line, label, repeat, clip
                    if key in saved:
                        raise ValueError("Duplicate saved trial identity")
                    saved[key] = value
                    header = dict(line=line, label=label, repeat=repeat, clip=clip, training_seconds=result["training_seconds"])
                    conditions.extend({**header, **row} for row in foot_conditions(value))
                    bins.extend(dict(**header, region=region, **frame_metrics(value, mask))
                                for region, mask in regions(count))
                    for foot, side in enumerate(("left", "right")):
                        curves.extend(dict(**header, side=side, seconds=(frame+1)*.02,
                            global_reference_frame=int(value["frames"][frame]),
                            sole_z_m=float(height[frame, foot]), reference_sole_z_m=float(reference_height[frame, foot]),
                            native_force_N=float(value["force"][frame, foot]),
                            native_slip_mps=float(value["slip"][frame, foot]),
                            intent_swing=bool(value["swing"][frame, foot]),
                            intent_stance=bool(value["stance"][frame, foot]), tilt_rad=float(value["tilt"][frame]))
                            for frame in range(count))
                if identity(trace_path) != before or identity(result_path) != result_identity:
                    raise ValueError("Saved evidence changed during offline analysis")
                trace_sources.extend((before, result_identity))
                print(json.dumps(dict(line=line, label=label, repeat=repeat, saved_only=True)), flush=True)
    sources = list(tracked.values())
    paired = []
    for (line, label, repeat, clip), value in saved.items():
        if label != "initial":
            paired.extend(dict(line=line, label=label, repeat=repeat, clip=clip, **row)
                          for row in paired_regions(saved[(line, "initial", repeat, clip)], value))
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("foot_conditions", conditions), ("time_bins", bins),
                       ("paired_regions", paired), ("foot_timeseries", curves)):
        write_csv(args.output / (name + ".csv"), rows)
    plot(args.output, saved)
    if sources != [identity(Path(row["path"])) for row in sources]:
        raise ValueError("Analysis dependencies changed")
    summary = dict(status="SAVED_PHASE_TRADEOFF_ANALYSIS_READY", captured_utc=datetime.now(timezone.utc).isoformat(),
        source_base_commit=args.source_base_commit, sources=sources, trace_sources=trace_sources,
        saved_clips=len(saved), foot_condition_rows=len(conditions), time_bin_rows=len(bins),
        paired_region_rows=len(paired), foot_timeseries_rows=len(curves),
        historical_initial_and_stage_trace_sha_verified=True, published_metrics_reconstructed=True,
        new_integration_steps=0, policy_inference_calls=0, optimizer_updates=0,
        goal_complete=False, user_accepted_checkpoint=None,
        limits=["Saved outcomes across training stages are not matched interventions or proof of a single cause.",
            "Native force/slip are the final 5ms substep samples, not integrated 20ms impulses.",
            "10mm reference windows and 5mm near-floor diagnostics are descriptive, not new acceptance gates.",
            "Conditional slip can change because loaded frame membership changes; counts are retained.",
            "Intent phase is immutable retarget metadata, not measured support truth.",
            "Base quaternion tilt follows the original published gate; it is not an independent torso sensor.",
            "Plots show repeat 0 only; CSVs include all three fixed clips and all repeats."])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "saved_clips", "foot_timeseries_rows")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("base", "ppo", "scene", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-base-commit", required=True)
    run(parser.parse_args())
