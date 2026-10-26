"""Summarize full saved Q1 rollouts; no physics integration or policy execution."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from audit_q1_hp_artifacts import identity, read


def aggregate(rows):
    if not rows:
        return None

    def mean(key):
        return float(np.mean([row[key] for row in rows]))

    value = {"clips": len(rows), "completed": sum(not row["failed"] for row in rows),
             "failed": sum(row["failed"] for row in rows)}
    for key in ("completion_fraction", "along_target_m", "net_xy_m", "reference_net_xy_m",
                "path_xy_m", "along_speed_m_s", "shoulder_rmse_rad", "elbow_rmse_rad"):
        value["mean_" + key] = mean(key)
    value["mean_support"] = {key: float(np.mean([row["support"][key] for row in rows]))
                             for key in rows[0]["support"]}
    value["clips_with_both_single_support_sides"] = sum(
        row["support"]["left_only_fraction"] > 0
        and row["support"]["right_only_fraction"] > 0 for row in rows)
    value["mean_abs_wrist_peak_height_error_m"] = np.abs(
        [row["wrist_peak_height_error_m"] for row in rows]).mean(axis=0).tolist()
    value["reached_full_reference_peak_time_count"] = np.asarray(
        [row["reached_full_reference_peak_time"] for row in rows]).sum(axis=0).tolist()
    value["mean_abs_wrist_peak_time_error_seconds_reached_only"] = []
    for side in range(2):
        errors = [abs(row["wrist_peak_time_error_seconds"][side]) for row in rows
                  if row["reached_full_reference_peak_time"][side]]
        value["mean_abs_wrist_peak_time_error_seconds_reached_only"].append(
            float(np.mean(errors)) if errors else None)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "trial", "library", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import mujoco
    import torch

    torch.set_num_threads(8)
    sys.path[:0] = [str(args.runtime / "src"), str(args.runtime)]
    from tools.x2_geometry import CollisionGeometry
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib

    helper_path = Path(__file__).with_name("analyze_q1_hp_trial.py")
    spec = importlib.util.spec_from_file_location("q1_saved_analysis", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    folder = args.evaluation_dir or args.trial / "evaluation"
    library = read(args.library)
    rows = library["splits"]["train"] + library["splits"]["validation"]
    outcome = read(folder / "final.json")
    with np.load(folder / "final_trajectory.npz", allow_pickle=False) as saved:
        trace = {key: saved[key] for key in saved.files}
    helper.validate_trace(trace, len(rows))
    scene = args.runtime / "q1_assets/scene.xml"
    kin = Q1Kinematics(scene, "cpu")
    motion = MotionLib([row["path"] for row in rows], kin, "cpu", src_fps=30.)
    model, data = mujoco.MjModel.from_xml_path(str(scene)), None
    data = mujoco.MjData(model)
    feet = [CollisionGeometry(model, (side + "_ankle_roll_link",))
            for side in ("left", "right")]
    wrists = [kin.tracked_idx.index(kin.body_names.index(side + "_wrist_yaw_link"))
              for side in ("left", "right")]
    per_clip = []
    for index, row in enumerate(rows):
        full = outcome["per_clip"][index]
        if (full["name"], full["split"]) != (row["name"], row["split"]):
            raise ValueError("Saved outcome and reference order differ")
        count = int(trace["steps"][index])
        actual = trace["states"][:count, index]
        frames = trace["frames"][:count, index]
        expected = motion.ref_state(torch.tensor(frames))[4].numpy()
        errors = actual[:, 13:35] - expected
        reference_frames = np.arange(int(motion.clip_start[index]),
                                     int(motion.clip_end[index]) + 1)
        ref_relative = (motion.body_p[reference_frames][:, wrists]
                        - motion.root_p[reference_frames, None, :]).numpy()
        reference_peak = ref_relative[..., 2].argmax(axis=0)
        reference_height = ref_relative[..., 2].max(axis=0)
        actual_height = np.asarray(full["max_wrist_height_above_root_m"])
        actual_peak = np.asarray(full["wrist_peak_frame"]) + 1
        value = {key: full[key] for key in ("name", "split", "group", "failed", "steps",
                 "reference_steps", "completion_fraction", "along_target_m", "net_xy_m",
                 "reference_net_xy_m", "path_xy_m")}
        value.update(
            along_speed_m_s=full["along_target_m"] / (count * .02),
            support=helper.support(model, data, feet, actual),
            shoulder_rmse_rad=float(np.sqrt(np.mean(
                errors[:, [14, 15, 16, 18, 19, 20]] ** 2))),
            elbow_rmse_rad=float(np.sqrt(np.mean(errors[:, [17, 21]] ** 2))),
            max_wrist_height_above_root_m=actual_height.tolist(),
            full_reference_max_wrist_height_m=reference_height.tolist(),
            wrist_peak_height_error_m=(actual_height - reference_height).tolist(),
            wrist_peak_time_error_seconds=((actual_peak - reference_peak) * .02).tolist(),
            reached_full_reference_peak_time=(count >= reference_peak).tolist())
        per_clip.append(value)
    summary = {}
    for split in ("train", "validation"):
        summary[split] = {}
        for group in ("all", *sorted({row["group"] for row in per_clip
                                      if row["split"] == split})):
            chosen = [row for row in per_clip if row["split"] == split
                      and (group == "all" or row["group"] == group)]
            summary[split][group] = {"all_rollouts": aggregate(chosen),
                                    "completed_only": aggregate(
                                        [row for row in chosen if not row["failed"]])}
    result = {"status": "SAVED_FULL_ROLLOUT_MOVEMENT_UPPER_ANALYSIS_COMPLETED",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "summary": summary, "per_clip": per_clip,
              "sources": [identity(path) for path in (args.library, scene,
                  folder / "final.json", folder / "final_trajectory.npz", helper_path)],
              "source": identity(__file__), "physics_integration_steps": 0,
              "policy_inference_calls": 0, "optimizer_updates": 0,
              "cpu_only": not torch.cuda.is_initialized(),
              "limits": "Geometric near-ground is not force contact. Full-prefix errors and "
              "motion include failures; completed_only is a diagnostic subset. Wrist height "
              "error compares observed peak to full-reference peak; timing aggregates include "
              "only sides that reached that reference time. Use common-prefix files for "
              "controlled initial/final or RF08/RF09 error comparisons."}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": identity(args.output),
                      "summary": {split: {group: summary[split][group]
                          for group in ("walking", "gesture")} for split in summary}}))


if __name__ == "__main__":
    main()
