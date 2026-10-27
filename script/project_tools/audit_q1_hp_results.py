"""Check saved Q1 evaluation arithmetic and six-panel CSV without new rollouts."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import numpy as np

from audit_q1_hp_artifacts import identity, read


def close(left, right):
    return bool(np.allclose(left, right, rtol=3e-5, atol=2e-6))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("trial", "runtime", "library", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--benchmark-library", type=Path)
    parser.add_argument("--baseline-evaluation", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch

    torch.set_num_threads(8)
    sys.path[:0] = [str(args.runtime / "src"), str(args.runtime)]
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib

    checks, sources = [], []

    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    trial = args.trial
    status, receipt, plan = (read(trial / name) for name in
                             ("status.json", "run/receipt.json", "plan.json"))
    check("pipeline_completed", status["status"] == "COMPLETED")
    phases = ["train", "evaluation", "dashboard", "analysis"]
    if args.benchmark_library:
        phases.extend(("benchmark", "benchmark_analysis"))
    if args.benchmark_library or args.baseline_evaluation:
        phases.append("baseline_comparison")
    for phase in phases:
        guard_path = trial / (phase + "_guard.json")
        guard = read(guard_path)
        check(phase + "_worker_reaped", guard["status"] == "WORKER_COMPLETED"
              and guard["cleanup"]["worker_reaped"]
              and guard["cleanup"]["worker_return_code"] == 0)
        sources.append(guard_path)
    changed = [item["path"] for item in plan["source"]
               if identity(item["path"]) != item]
    check("frozen_sources_unchanged_through_postprocessing", not changed,
          {"files": len(plan["source"]), "changed": changed})
    kin = Q1Kinematics(args.runtime / "q1_assets/scene.xml", "cpu")
    folders = [(trial/"evaluation", args.library)]
    if args.benchmark_library:
        folders.append((trial/"benchmark", args.benchmark_library))
    for folder, library_path in folders:
        library = read(library_path)
        rows = library["splits"]["train"] + library["splits"]["validation"]
        motion = MotionLib([row["path"] for row in rows], kin, "cpu", src_fps=30.)
        for label in ("initial", "final"):
            result = read(folder / (label + ".json"))
            with np.load(folder / (label + "_trajectory.npz"), allow_pickle=False) as saved:
                trace = {key: saved[key] for key in saved.files}
            prefix = folder.name + "_" + label
            check(prefix + "_checkpoint_identity", result["checkpoint"] == receipt[label])
            valid, mismatches = True, []
            for index, row in enumerate(rows):
                count = int(trace["steps"][index])
                alive = trace["alive"][:, index]
                frames = trace["frames"][:count, index]
                start = int(motion.clip_start[index])
                expected_steps = int(motion.clip_len[index]) - 1
                states = trace["states"][:count, index]
                valid = valid and (0 < count <= expected_steps and alive[:count].all()
                    and not alive[count:].any() and np.isfinite(states).all()
                    and np.array_equal(frames, np.arange(start + 1, start + count + 1)))
                ref = motion.ref_state(torch.tensor(frames))
                first = trace["initial"][index, :2]
                delta = states[-1, :2] - first
                ref_delta = (motion.root_p[motion.clip_end[index], :2]
                             - motion.root_p[start, :2]).numpy()
                direction = ref_delta / max(np.linalg.norm(ref_delta), 1e-12)
                error = states[:, 13:35] - ref[4].numpy()
                expected = {"steps": count, "reference_steps": expected_steps,
                    "failed": bool(trace["failed"][index]),
                    "completion_fraction": count / expected_steps,
                    "normalized_full_clip_return": float(trace["returns"][index])
                    / (expected_steps * .1),
                    "net_xy_m": float(np.linalg.norm(delta)),
                    "reference_net_xy_m": float(np.linalg.norm(ref_delta)),
                    "along_target_m": float(delta @ direction),
                    "path_xy_m": float(np.linalg.norm(np.diff(
                        np.vstack((first, states[:, :2])), axis=0), axis=1).sum()),
                    "root_xy_rmse_m": float(np.sqrt(np.mean(
                        (states[:, :2] - ref[0].numpy()[:, :2]) ** 2))),
                    "leg_rmse_rad": float(np.sqrt(np.mean(error[:, :12] ** 2))),
                    "upper_limb_rmse_rad": float(np.sqrt(np.mean(error[:, 14:] ** 2))),
                    "joint_rmse_rad": float(np.sqrt(np.mean(error ** 2)))}
                measured = result["per_clip"][index]
                if (measured["name"], measured["split"], measured["group"]) != (
                        row["name"], row["split"], row["group"]):
                    mismatches.append((row["name"], "identity"))
                mismatches.extend((row["name"], key) for key, value in expected.items()
                                  if not close(value, measured[key]))
            check(prefix + "_valid_contiguous_frame_zero_traces", valid,
                  {"clips": len(rows)})
            check(prefix + "_per_clip_physical_arithmetic", not mismatches, mismatches)
            for split in ("train", "validation"):
                chosen = [row for row in result["per_clip"] if row["split"] == split]
                expected = {"clips": len(chosen), "failed": sum(row["failed"] for row in chosen)}
                for output_key, key in (("mean_completion_fraction", "completion_fraction"),
                        ("mean_normalized_full_clip_return", "normalized_full_clip_return"),
                        ("mean_root_xy_rmse_m", "root_xy_rmse_m"),
                        ("mean_upper_limb_rmse_rad", "upper_limb_rmse_rad")):
                    expected[output_key] = float(np.mean([row[key] for row in chosen]))
                check(prefix + "_" + split + "_aggregation", all(
                    close(value, result["summary"][split][key]) for key, value in expected.items()))
            sources.extend((folder / (label + ".json"), folder / (label + "_trajectory.npz")))

    evidence = [json.loads(line) for line in
                (trial / "run/training_evidence.jsonl").read_text().splitlines()]
    definitions = {"episode_return": ("normalized_return_sum", "completed_episodes", 1.),
                   "episode_length": ("length_steps_sum", "completed_episodes", 1.),
                   "episode_termination": ("hard_terminations", "completed_episodes", 1.),
                   "step_reward": ("reward_mean", None, .1),
                   "world_model": ("wm_err8", None, 1.),
                   "survival": ("imag_survival_H", None, 1.)}
    expected_points = {}
    for metric, (field, denominator, scale) in definitions.items():
        selected = [row for row in evidence if row[field] is not None]
        steps = np.array([row["step"] for row in selected])
        nums = np.array([row[field] / scale for row in selected])
        dens = np.array([row[denominator] if denominator else 1. for row in selected])
        left = np.searchsorted(steps, steps - 500, side="right")
        sum_num, sum_den = np.r_[0., np.cumsum(nums)], np.r_[0., np.cumsum(dens)]
        for index, row in enumerate(selected):
            total = sum_den[index + 1] - sum_den[left[index]]
            expected_points[(metric, row["step"])] = (
                nums[index], dens[index], nums[index] / dens[index] if dens[index] else None,
                (sum_num[index + 1] - sum_num[left[index]]) / total if total else None,
                row["sim_steps"])
    csv_path = trial / "dashboard/training_dashboard.csv"
    errors, seen = [], set()
    with csv_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["metric"], int(row["iteration"])
            if key in seen or key not in expected_points:
                errors.append((key, "unexpected_or_duplicate"))
                continue
            seen.add(key)
            expected = expected_points[key]
            actual = [float(row[field]) if row[field] else None
                      for field in ("numerator", "denominator", "raw", "smoothed", "env_steps")]
            if any((left is None) != (right is None) or (left is not None
                    and not math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-7))
                    for left, right in zip(actual, expected)):
                errors.append((key, "value"))
    check("six_panel_csv_all_values_and_500_iteration_pooled_windows",
          not errors and seen == expected_points.keys(),
          {"rows": len(seen), "expected_rows": len(expected_points), "errors": errors[:30]})
    check("cpu_only_no_cuda", not torch.cuda.is_initialized())
    sources.extend((trial / "status.json", trial / "plan.json", trial / "run/receipt.json",
                    trial / "run/training_evidence.jsonl", csv_path))
    result = {"status": "HP_Q1_RESULT_AUDIT_PASSED" if all(row["passed"] for row in checks)
              else "HP_Q1_RESULT_AUDIT_FAILED", "created_utc": datetime.now(timezone.utc).isoformat(),
              "checks": checks, "source": identity(__file__),
              "sources": [identity(path) for path in sources],
              "physics_integration_steps": 0, "policy_inference_calls": 0, "optimizer_updates": 0,
              "scope": "Saved frame alignment, outcome arithmetic and dashboard values. Returns "
              "are normalized from saved rollout sums, not newly recomputed per-step rewards."}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "checks": len(checks),
                      "failed": [row for row in checks if not row["passed"]]}))
    return 0 if all(row["passed"] for row in checks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
