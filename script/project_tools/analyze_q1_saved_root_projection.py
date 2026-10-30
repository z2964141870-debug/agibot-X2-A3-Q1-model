"""Decompose saved root-orientation sole geometry, without policy or dynamics."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_gait_goal_analysis"),
               str(HERE.parent / "supermimic_q1_sole")]
from prepare_q1_gait_review import FIXED, identity, write_csv
from saved_control_usage import phase_rows
from saved_pd_demand import preceding_states
from analyze_q1_saved_target_pose import describe


def up_in_root(rotations):
    rotations = np.asarray(rotations, dtype=np.float64)
    if (rotations.ndim != 3 or rotations.shape[1:] != (3, 3) or not len(rotations)
            or not np.isfinite(rotations).all()
            or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=2e-6)
            or not np.allclose(np.linalg.det(rotations), 1., atol=2e-6)):
        raise ValueError("Expected nonempty proper root rotation matrices")
    return rotations[:, 2, :].copy()


def foot_support(positions, rotations, points, radii, normals):
    positions, rotations, points, radii, normals = [np.asarray(v, dtype=np.float64)
        for v in (positions, rotations, points, radii, normals)]
    count = len(positions)
    if (not count or positions.shape != (count, 3) or rotations.shape != (count, 3, 3)
            or points.ndim != 2 or points.shape[1:] != (3,) or not len(points)
            or radii.shape != (len(points),) or normals.shape != (count, 3)
            or not all(np.isfinite(v).all() for v in (positions, rotations, points, radii, normals))
            or (radii < 0).any() or not np.allclose(np.linalg.norm(normals, axis=-1), 1., atol=2e-6)):
        raise ValueError("Expected finite aligned poses, unit normals and collision points/radii")
    indices = []
    for begin in range(0, count, 128):
        directions = np.einsum("nji,nj->ni", rotations[begin:begin + 128], normals[begin:begin + 128])
        indices.append((np.einsum("ni,vi->nv", directions, points) - radii).argmin(-1))
    indices = np.concatenate(indices)
    selected = positions + np.einsum("nij,nj->ni", rotations, points[indices])
    selected_radii = radii[indices]
    heights = np.einsum("ni,ni->n", normals, selected) - selected_radii
    return heights, selected, selected_radii, indices


def projection_parts(before, after, old_points, old_radii, new_points, new_radii):
    before, after, old_points, old_radii, new_points, new_radii = [np.asarray(v, dtype=np.float64)
        for v in (before, after, old_points, old_radii, new_points, new_radii)]
    count = len(before)
    if (not count or before.shape != (count, 3) or after.shape != before.shape
            or old_points.shape != (count, 2, 3) or new_points.shape != old_points.shape
            or old_radii.shape != (count, 2) or new_radii.shape != old_radii.shape
            or not all(np.isfinite(v).all() for v in
                       (before, after, old_points, old_radii, new_points, new_radii))
            or (old_radii < 0).any() or (new_radii < 0).any()):
        raise ValueError("Expected finite aligned normals and two selected material points/radii")
    reference = np.einsum("ni,nfi->nf", before, old_points) - old_radii
    fixed = np.einsum("ni,nfi->nf", after, old_points) - old_radii
    changed = np.einsum("ni,nfi->nf", after, new_points) - new_radii
    relative = lambda z: z[:, 1] - z[:, 0]
    xyz = (after - before) * (old_points[:, 1] - old_points[:, 0])
    switch = relative(changed) - relative(fixed)
    total = relative(changed) - relative(reference)
    np.testing.assert_allclose(xyz.sum(-1) + switch, total, atol=2e-12, rtol=2e-12)
    return dict(reference_relative_m=relative(reference), same_root_relative_m=relative(changed),
        root_effect_m=total, projection_x_m=xyz[:, 0], projection_y_m=xyz[:, 1],
        projection_z_m=xyz[:, 2], support_switch_m=switch)


def tilt_coordinates(normals):
    normals = np.asarray(normals, dtype=np.float64)
    if (normals.ndim != 2 or normals.shape[1:] != (3,) or not len(normals)
            or not np.isfinite(normals).all()
            or not np.allclose(np.linalg.norm(normals, axis=-1), 1., atol=2e-6)):
        raise ValueError("Expected finite unit root-up projections")
    return np.stack((np.arctan2(normals[:, 1], normals[:, 2]),
                     -np.arcsin(np.clip(normals[:, 0], -1., 1.))), axis=-1)


def draw(output, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#168568", "#bd4b5a", "#b38b22", "#727272")
    components = ("projection_x_m", "projection_y_m", "projection_z_m", "support_switch_m")
    names = ("Local X projection", "Local Y projection", "Local Z projection", "Sole point switch")
    for line in ("base", "ppo"):
        labels = list(dict.fromkeys(key[1] for key in curves if key[0] == line))
        figure = plt.figure(figsize=(13, 8.5), layout="constrained")
        grid = figure.add_gridspec(4, 2, height_ratios=(1, 1, .18, .18))
        for col, clip in enumerate(FIXED[:2]):
            for row, label in enumerate((labels[0], labels[-1])):
                ax = figure.add_subplot(grid[row, col])
                values = [curves[line, label, repeat, clip] for repeat in range(3)]
                time = np.arange(len(values[0]["root_effect_m"])) * .02
                for name, color, title in zip(components, colors, names):
                    ax.plot(time, np.median(np.stack([v[name] for v in values]), axis=0) * 1000,
                            label=title, color=color, lw=1)
                ax.plot(time, np.median(np.stack([v["root_effect_m"] for v in values]), axis=0) * 1000,
                        label="Total root effect", color="#222222", ls="--", lw=1.1)
                ax.axhline(0, color="#555555", lw=.5)
                ax.grid(alpha=.18)
                ax.set_title(f"{'Stop' if col == 0 else 'Start'} walk / {label}")
                ax.set_ylabel("Right minus left change (mm)")
                ax.set_xlabel("Saved control-start time (s)")
        legend = figure.add_subplot(grid[2, :])
        legend.axis("off")
        legend.legend(*ax.get_legend_handles_labels(), ncol=3, loc="center", fontsize=9)
        note = figure.add_subplot(grid[3, :])
        note.axis("off")
        note.text(.5, .5, "Static reference joints with observed pre-root orientation; no balance or contact solve.\nCurves are separate framewise medians: their plotted sum need not equal the median total.",
                  ha="center", va="center", transform=note.transAxes, fontsize=9)
        figure.suptitle(f"Q1 {line.upper()}: exact per-sample root projection decomposition", fontsize=13)
        for extension in ("png", "pdf"):
            figure.savefig(output / f"{line}_root_projection.{extension}", dpi=150)
        plt.close(figure)


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path[:0] = [str(args.runtime / "src"), str(args.runtime)]
    import torch
    from sole_geometry import SoleGeometry
    from unisupermimic.rl.algos.supermimic.geometry import quat_to_mat
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics

    torch.set_num_threads(4)
    tracked = {}
    def remember(path, expected=None):
        actual = identity(path)
        if expected is not None and any(actual[k] != expected[k] for k in ("bytes", "sha256")):
            raise ValueError("Published or frozen input changed: " + str(path))
        if str(path) in tracked and tracked[str(path)] != actual:
            raise ValueError("Input changed while being read: " + str(path))
        tracked[str(path)] = actual
        return actual

    remember(args.target_manifest)
    manifest = json.loads(args.target_manifest.read_text())
    files = manifest["artifacts"]["files"]
    for name in ("summary.json", "target_pose_timeseries.csv"):
        remember(args.target / name, next(row for row in files if row["relative_path"].endswith("/" + name)))
    prior = json.loads((args.target / "summary.json").read_text())
    if (prior["status"] != "SAVED_TARGET_POSE_GEOMETRY_READY" or prior["saved_clips"] != 45
            or prior["new_integration_steps"] or prior["policy_inference_calls"]):
        raise ValueError("Expected completed saved-only target geometry")
    history = {r["path"]: r for r in prior["trace_sources"]}
    published = phase_rows(args.target / "target_pose_timeseries.csv")
    native = args.runtime / "src/unisupermimic/tasks/motion_tracking/q1/supermimic"
    frozen_paths = [native / "kinematics.py", native / "robot.py",
        args.runtime / "src/unisupermimic/rl/algos/supermimic/geometry.py",
        HERE.parent / "supermimic_for_q1/supermimic/kinematics/g1_fk.py",
        HERE.parent / "supermimic_q1_sole/sole_geometry.py"]
    for path in [Path(__file__), HERE / "prepare_q1_gait_review.py",
                 HERE / "analyze_q1_saved_target_pose.py", HERE / "audit_q1_saved_sole_geometry.py",
                 HERE.parent / "supermimic_q1_gait_goal_analysis/saved_control_usage.py",
                 HERE.parent / "supermimic_q1_gait_goal_analysis/saved_pd_demand.py", *frozen_paths]:
        remember(path)
    scene = args.runtime / "q1_assets/scene.xml"
    kin = Q1Kinematics(scene, "cpu")
    geometry = SoleGeometry.from_scene(scene)
    body_ids = [kin.body_names.index(foot.body) for foot in geometry.feet]
    conditions, timeseries, curves = [], [], {}
    maximum_reference_error, maximum_same_root_error, maximum_decomposition_error = 0., 0., 0.
    batches = 0
    for line, trial in (("base", args.base), ("ppo", args.ppo)):
        sources = [trial / name for name in ("status.json", "postdelivery_status.json", "plan.json",
                   "run/checkpoint_selection.json", "evaluation/evaluation_library.json")]
        for path in sources:
            remember(path)
        status, post, plan, selection, library = [json.loads(path.read_text()) for path in sources]
        if (status["status"] != ("COMPLETED" if line == "base" else "FAILED")
                or post["status"] != "COMPLETED" or selection["user_accepted_checkpoint"] is not None
                or len(library["splits"]["train"]) != 330):
            raise ValueError("Expected original outcomes,330 clips and null adoption")
        frozen = {row["path"]: row for row in plan["source"]}
        assets = [Path(path) for path in frozen if path.startswith(str(scene.parent) + "/")]
        for path in [*frozen_paths, *assets]:
            remember(path, frozen[str(path)])
        slots = [next(i for i, row in enumerate(library["splits"]["train"]) if row["name"] == name) for name in FIXED]
        for candidate in selection["candidates"]:
            label = candidate["label"]
            for repeat, receipt in enumerate(candidate["traces"]):
                trace = trial / f"evaluation/{label}_repeat{repeat}.npz"
                if str(trace) != receipt["path"]:
                    raise ValueError("Original trace identity differs")
                remember(trace, history[str(trace)])
                if tracked[str(trace)] != receipt:
                    raise ValueError("Checkpoint trace receipt differs")
                result_path = trace.with_suffix(".json")
                remember(result_path, history[str(result_path)])
                result = json.loads(result_path.read_text())
                if result["label"] != label or result["repeat"] != repeat or len(result["per_clip"]) != 330:
                    raise ValueError("Original evaluation identity differs")
                with np.load(trace, allow_pickle=False) as archive:
                    arrays = {name: archive[name][:, slots].copy() for name in
                        ("states", "frames", "alive", "reference_q", "reference_p", "reference_quat")}
                    arrays.update(initial=archive["initial"][slots], steps=archive["steps"][slots],
                                  reference_steps=archive["reference_steps"][slots])
                for slot, index in enumerate(slots):
                    clip = FIXED[slot]
                    count = int(arrays["steps"][slot])
                    if (count != int(arrays["reference_steps"][slot]) or count != result["per_clip"][index]["steps"]
                            or result["per_clip"][index]["name"] != clip
                            or not arrays["alive"][:count, slot].all()):
                        raise ValueError("Expected unchanged complete fixed prefixes")
                    previous = preceding_states(arrays["initial"][slot], arrays["states"][:count, slot])
                    quats = [arrays["reference_quat"][:count, slot], previous[:, 3:7]]
                    if any(q.shape != (count, 4) or not np.isfinite(q).all()
                           or (np.linalg.norm(q, axis=-1) < 1e-6).any() for q in quats):
                        raise ValueError("Invalid original root quaternions")
                    normals = [up_in_root(quat_to_mat(torch.from_numpy(q.astype(np.float64))).numpy()) for q in quats]
                    positions, rotations = [], []
                    with torch.no_grad():
                        for begin in range(0, count, 128):
                            q = torch.from_numpy(arrays["reference_q"][begin:min(begin + 128, count), slot].astype(np.float32))
                            size = len(q)
                            zeros = torch.zeros((size, 3))
                            eye = torch.eye(3).expand(size, 3, 3)
                            fk = kin.forward(zeros, eye, zeros, zeros, q, torch.zeros_like(q), all_bodies=True)
                            positions.append(fk[0][:, body_ids].numpy().astype(np.float64))
                            rotations.append(fk[1][:, body_ids].numpy().astype(np.float64))
                            batches += 1
                    positions, rotations = np.concatenate(positions), np.concatenate(rotations)
                    supports = [[foot_support(positions[:, foot], rotations[:, foot], surface.points,
                        surface.radii, normal) for foot, surface in enumerate(geometry.feet)] for normal in normals]
                    selected = [np.stack([foot[1] for foot in value], axis=1) for value in supports]
                    radii = [np.stack([foot[2] for foot in value], axis=1) for value in supports]
                    parts = projection_parts(*normals, selected[0], radii[0], selected[1], radii[1])
                    maximum_decomposition_error = max(maximum_decomposition_error,
                        float(np.abs(sum(parts[k] for k in ("projection_x_m", "projection_y_m", "projection_z_m", "support_switch_m")) - parts["root_effect_m"]).max()))
                    tilt = [tilt_coordinates(n) for n in normals]
                    difference = np.arctan2(np.sin(tilt[1] - tilt[0]), np.cos(tilt[1] - tilt[0]))
                    left, right = [published[line, label, repeat, clip, side] for side in ("left", "right")]
                    if len(left) != count or len(right) != count:
                        raise ValueError("Published target duration differs")
                    for foot, data in enumerate((left, right)):
                        np.testing.assert_array_equal([int(r["global_reference_frame"]) for r in data], arrays["frames"][:count, slot])
                        np.testing.assert_array_equal([int(r["sample"]) for r in data], np.arange(count))
                        for variant, key, root_z in ((0, "reference_sole_z_m", arrays["reference_p"][:count, slot, 2]),
                                                    (1, "reference_same_root_sole_z_m", previous[:, 2])):
                            expected = np.array([float(r[key]) for r in data])
                            error = float(np.abs(supports[variant][foot][0] + root_z - expected).max())
                            np.testing.assert_allclose(supports[variant][foot][0] + root_z, expected, atol=2e-6, rtol=2e-5)
                            if variant == 0:
                                maximum_reference_error = max(maximum_reference_error, error)
                            else:
                                maximum_same_root_error = max(maximum_same_root_error, error)
                    header = dict(line=line, label=label, repeat=repeat, clip=clip, training_seconds=result["training_seconds"])
                    for foot, (side, data) in enumerate(zip(("left", "right"), (left, right))):
                        raised = np.array([r["intent_swing"] == "True" and float(r["reference_sole_z_m"]) >= .01 for r in data])
                        masks = dict(full=np.ones(count, dtype=bool), raised_reference_10mm=raised)
                        masks.update({f"raised_reference_10mm_q{quarter + 1}": raised & (np.arange(count) * 4 // count == quarter) for quarter in range(4)})
                        sign = -1 if foot == 0 else 1
                        for condition, mask in masks.items():
                            row = dict(**header, side=side, condition=condition, frames=int(mask.sum()))
                            for key, values in parts.items():
                                row[key.replace("_m", "_mean_m")] = describe(sign * values, mask)["mean"]
                            row["local_y_tilt_delta_mean_rad"] = describe(difference[:, 0], mask)["mean"]
                            row["local_x_tilt_delta_mean_rad"] = describe(difference[:, 1], mask)["mean"]
                            conditions.append(row)
                    for sample in range(count):
                        row = dict(**header, sample=sample, control_start_s=sample * .02,
                            reference_sample_end_s=(sample + 1) * .02, global_reference_frame=int(arrays["frames"][sample, slot]),
                            raised_reference_right=right[sample]["intent_swing"] == "True" and float(right[sample]["reference_sole_z_m"]) >= .01,
                            raised_reference_left=left[sample]["intent_swing"] == "True" and float(left[sample]["reference_sole_z_m"]) >= .01,
                            local_y_tilt_delta_rad=float(difference[sample, 0]), local_x_tilt_delta_rad=float(difference[sample, 1]))
                        row.update({key: float(value[sample]) for key, value in parts.items()})
                        for variant, name in enumerate(("reference", "saved")):
                            for axis, axis_name in enumerate("xyz"):
                                row[name + "_up_local_" + axis_name] = float(normals[variant][sample, axis])
                            for foot, side in enumerate(("left", "right")):
                                row[name + "_" + side + "_point"] = int(supports[variant][foot][3][sample])
                        timeseries.append(row)
                    curves[line, label, repeat, clip] = parts
                print(json.dumps(dict(line=line, label=label, repeat=repeat, saved_geometry_only=True)), flush=True)
    if len(curves) != 45 or len(timeseries) != 12015 or len(conditions) != 540:
        raise ValueError("Expected45 saved fixed clips,12015 states and540 conditions")
    for path, expected in tracked.items():
        if identity(Path(path)) != expected:
            raise ValueError("Input changed during static analysis: " + path)
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "root_projection_conditions.csv", conditions)
    write_csv(args.output / "root_projection_timeseries.csv", timeseries)
    draw(args.output, curves)
    summary = dict(status="SAVED_ROOT_PROJECTION_READY", captured_utc=datetime.now(timezone.utc).isoformat(),
        source_base_commit=args.source_base_commit, sources=list(tracked.values()),
        saved_clips=len(curves), saved_states=len(timeseries), condition_rows=len(conditions),
        static_fk_batches=batches, static_fk_poses=len(timeseries),
        maximum_reference_height_error_m=maximum_reference_error,
        maximum_saved_root_height_error_m=maximum_same_root_error,
        maximum_decomposition_error_m=maximum_decomposition_error,
        geometry=geometry.identity(), new_integration_steps=0, policy_inference_calls=0,
        optimizer_updates=0, new_native_control_environments=0, contact_dynamics_solves=0,
        goal_complete=False, user_accepted_checkpoint=None,
        limitations=["Exact static root-local projection attribution is not a unique training cause or physical intervention.",
            "X/Y/Z are Q1 root-local axes, not independent Euler-angle interventions or torques.",
            "Reference minimizing material points anchor axis terms; reselection and radius changes remain in the switch term.",
            "Reference joint coordinates are next20ms samples, observed root is at control start.",
            "Signed right-minus-left heights remove common translation but are not absolute clearance or support truth.",
            "Separate framewise median curves do not preserve additive identities; per-sample and per-repeat means do.",
            "No original targets, dynamics, reward, model, policy, observation plan or acceptance was changed."])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "saved_clips", "saved_states", "condition_rows",
        "static_fk_batches", "static_fk_poses", "maximum_reference_height_error_m", "maximum_saved_root_height_error_m",
        "maximum_decomposition_error_m")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "base", "ppo", "target", "target-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-base-commit", required=True)
    run(parser.parse_args())
