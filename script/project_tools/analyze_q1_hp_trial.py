"""Compare saved HP-Q1 rollouts without integrating physics or updating a policy."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def rmse(actual, expected):
    return float(np.sqrt(np.mean(np.square(actual - expected))))


def common_states(initial, final, clip):
    count = min(int(initial["steps"][clip]), int(final["steps"][clip]))
    if count < 1:
        raise ValueError("Comparison has no common executed frame")
    frames = initial["frames"][:count, clip]
    if not np.array_equal(frames, final["frames"][:count, clip]):
        raise ValueError("Rollouts do not share the same reference timeline")
    return count, frames, {label: trace["states"][:count, clip]
                           for label, trace in (("initial", initial), ("final", final))}


def validate_trace(trace, clips):
    if trace["initial"].shape != (clips, 57) or trace["states"].shape[1:] != (clips, 57):
        raise ValueError("Expected HP-Q1 state layout")
    if not np.isfinite(trace["initial"]).all():
        raise ValueError("Nonfinite initial physical state")
    for clip in range(clips):
        count = int(trace["steps"][clip])
        alive = trace["alive"][:, clip]
        if count < 1 or count > len(alive) or not alive[:count].all() or alive[count:].any():
            raise ValueError("Executed frames are not a contiguous pre-reset prefix")
        if not np.isfinite(trace["states"][:count, clip]).all():
            raise ValueError("Nonfinite executed physical state")


def support(model, data, geometries, states):
    import mujoco

    labels = []
    for state in states:
        data.qpos[:] = np.r_[state[:7], state[13:35]]
        mujoco.mj_kinematics(model, data)
        labels.append([geometry.minimum(data) < .015 for geometry in geometries])
    near = np.asarray(labels, dtype=bool)
    code = near[:, 0].astype(int) + 2 * near[:, 1].astype(int)
    single = code[(code == 1) | (code == 2)]
    return {"double_near_ground_fraction": float((code == 3).mean()),
            "left_only_fraction": float((code == 1).mean()),
            "right_only_fraction": float((code == 2).mean()),
            "neither_fraction": float((code == 0).mean()),
            "geometric_state_changes": int(np.count_nonzero(np.diff(code))),
            "single_support_side_changes": int(np.count_nonzero(np.diff(single)))}


def summarize(rows):
    result = {}
    metrics = ("root_xy_rmse_m", "leg_rmse_rad", "upper_limb_rmse_rad",
               "wrist_world_rmse_m", "wrist_relative_root_rmse_m")
    for split in ("train", "validation"):
        groups = ["all", *sorted({row["group"] for row in rows if row["split"] == split})]
        result[split] = {}
        for group in groups:
            selected = [row for row in rows if row["split"] == split
                        and (group == "all" or row["group"] == group)]
            value = {"clips": len(selected), "mean_common_seconds": float(np.mean(
                [row["common_steps"] * .02 for row in selected]))}
            for label in ("initial", "final"):
                original = [row[label]["full_rollout"] for row in selected]
                measured = [row[label]["common_prefix"] for row in selected]
                value[label] = {
                    "failed": sum(row["failed"] for row in original),
                    "mean_completion_fraction": float(np.mean([row["completion_fraction"] for row in original])),
                    "mean_full_clip_normalized_return": float(np.mean([row["normalized_full_clip_return"] for row in original])),
                    "mean_net_xy_m": float(np.mean([row["net_xy_m"] for row in original])),
                    "mean_along_target_m": float(np.mean([row["along_target_m"] for row in original])),
                    "common_prefix": {key: float(np.mean([row[key] for row in measured])) for key in metrics},
                }
            result[split][group] = value
    return result


def analyze(args):
    import mujoco
    import torch

    sys.path[:0] = [str(args.runtime / "src"), str(args.runtime)]
    from tools.x2_geometry import CollisionGeometry
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib

    torch.set_num_threads(8)
    library = json.loads(args.library.read_text())
    rows = library["splits"]["train"] + library["splits"]["validation"]
    folder = args.trial / "evaluation"
    traces = {}
    for label in ("initial", "final"):
        with np.load(folder / (label + "_trajectory.npz"), allow_pickle=False) as saved:
            traces[label] = {key: saved[key] for key in saved.files}
    original = {label: json.loads((folder / (label + ".json")).read_text())
                for label in traces}
    for trace in traces.values():
        validate_trace(trace, len(rows))
    initial_error = float(np.max(np.abs(traces["initial"]["initial"] - traces["final"]["initial"])))
    if initial_error > 2e-6:
        raise ValueError(f"Initial physical states differ by {initial_error}")
    scene = args.runtime / "q1_assets/scene.xml"
    kin = Q1Kinematics(scene, "cpu")
    motion = MotionLib([row["path"] for row in rows], kin, "cpu", src_fps=30.)
    wrists = [kin.tracked_idx.index(kin.body_names.index(side + "_wrist_yaw_link"))
              for side in ("left", "right")]
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    feet = [CollisionGeometry(model, (side + "_ankle_roll_link",)) for side in ("left", "right")]
    compared = []
    with torch.inference_mode():
        for clip, row in enumerate(rows):
            count, frames, states = common_states(traces["initial"], traces["final"], clip)
            start = int(motion.clip_start[clip])
            if not np.array_equal(frames, np.arange(start + 1, start + count + 1)):
                raise ValueError("Executed frames are not the clip's frame-zero rollout")
            expected = motion.ref_state(torch.tensor(frames))
            ref_root, ref_joints = expected[0].numpy(), expected[4].numpy()
            ref_wrists = motion.body_p[frames][:, wrists].numpy()
            ref_relative = ref_wrists - ref_root[:, None, :]
            full_frames = np.arange(int(motion.clip_start[clip]), int(motion.clip_end[clip]) + 1)
            full_relative = (motion.body_p[full_frames][:, wrists]
                             - motion.root_p[full_frames, None, :]).numpy()
            value = {"name": row["name"], "split": row["split"], "group": row["group"],
                     "common_steps": count, "reference_steps": len(full_frames) - 1,
                     "full_reference_wrist_peak_frame": full_relative[..., 2].argmax(axis=0).tolist(),
                     "full_reference_max_wrist_height_m": full_relative[..., 2].max(axis=0).tolist()}
            for label, current in states.items():
                full = original[label]["per_clip"][clip]
                if (full["name"], full["split"]) != (row["name"], row["split"]):
                    raise ValueError("Saved outcome and library clip order differ")
                bodies = kin.forward(*unflatten_state(torch.tensor(current)))[0].numpy()
                relative = bodies[:, wrists] - current[:, None, :3]
                prefix = {"root_xy_rmse_m": rmse(current[:, :2], ref_root[:, :2]),
                          "leg_rmse_rad": rmse(current[:, 13:25], ref_joints[:, :12]),
                          "upper_limb_rmse_rad": rmse(current[:, 27:35], ref_joints[:, 14:]),
                          "wrist_world_rmse_m": rmse(bodies[:, wrists], ref_wrists),
                          "wrist_relative_root_rmse_m": rmse(relative, ref_relative),
                          "max_wrist_height_above_root_m": relative[..., 2].max(axis=0).tolist(),
                          "reference_max_wrist_height_m": ref_relative[..., 2].max(axis=0).tolist(),
                          "wrist_peak_control_frame": (relative[..., 2].argmax(axis=0) + 1).tolist(),
                          "reference_peak_control_frame": (ref_relative[..., 2].argmax(axis=0) + 1).tolist(),
                          "support": support(model, data, feet, current)}
                duration = full["steps"] * .02
                value[label] = {"common_prefix": prefix, "full_rollout": full,
                                "mean_along_speed_m_s_including_failures": full["along_target_m"] / duration,
                                "reached_full_reference_wrist_peak_time": [full["steps"] >= peak
                                    for peak in value["full_reference_wrist_peak_frame"]]}
            compared.append(value)
    sources = [args.library, scene, args.trial / "run/receipt.json", folder / "receipt.json"]
    sources += [folder / (label + suffix) for label in traces for suffix in (".json", "_trajectory.npz")]
    result = {"status": "SAVED_TRAJECTORY_COMMON_PREFIX_ANALYSIS_COMPLETED",
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "same_initial_state_max_abs_error": initial_error, "summary": summarize(compared),
              "per_clip": compared, "sources": [identity(path) for path in sources],
              "source": identity(__file__), "physics_integration_steps": 0,
              "policy_inference_calls": 0, "optimizer_updates": 0,
              "support_definition": "Collision geometry lowest point below 15mm; not measured contact force",
              "aggregation": "Equal clip weights; errors use each clip's common executed prefix",
              "limits": "Net displacement and speed include failed rollouts. Geometric support switches alone do not prove stable walking. Full-reference arm peaks outside the executed prefix are not evaluated as achieved."}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "summary": result["summary"]}), flush=True)


def self_test():
    states = np.zeros((4, 1, 57))
    first = {"steps": np.array([4]), "frames": np.arange(4)[:, None], "states": states.copy()}
    second = {"steps": np.array([2]), "frames": np.arange(4)[:, None], "states": states.copy()}
    first["states"][2:] = 100
    count, frames, values = common_states(first, second, 0)
    assert count == 2 and rmse(values["initial"], values["final"]) == 0
    second["frames"][1] = 7
    try:
        common_states(first, second, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched timelines were accepted")
    assert rmse(np.array([0., 2.]), np.array([0., 0.])) == np.sqrt(2.)
    print("PASS: unequal termination prefixes, reference alignment, and RMSE arithmetic; no physics or optimizer")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "library", "trial", "output"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
    else:
        if any(getattr(args, name) is None for name in ("runtime", "library", "trial", "output")):
            p.error("runtime, library, trial, and output are required")
        analyze(args)


if __name__ == "__main__":
    main()
