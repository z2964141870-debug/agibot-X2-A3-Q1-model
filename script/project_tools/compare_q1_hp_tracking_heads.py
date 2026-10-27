"""Score old/new analytic tracking heads on the same saved physical states."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

from audit_q1_hp_artifacts import identity


def trace(path):
    with np.load(path/"final_trajectory.npz", allow_pickle=False) as saved:
        return {key: saved[key] for key in ("initial", "states", "frames", "steps")}


def summarize(rows):
    result = {}
    for split in ("train", "validation"):
        result[split] = {}
        for group in ("all", *sorted({row["group"] for row in rows if row["split"] == split})):
            chosen = [row for row in rows if row["split"] == split and (group == "all" or row["group"] == group)]
            result[split][group] = {"clips": len(chosen), "mean_common_seconds": float(np.mean([r["steps"]*.02 for r in chosen]))}
            for label in ("baseline", "candidate"):
                result[split][group][label] = {key: float(np.mean([row[label][key] for row in chosen]))
                    for key in chosen[0][label]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "library", "candidate-evaluation", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--baseline", action="append", required=True, help="LABEL=EVALUATION_DIRECTORY; repeat for separate baselines")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch

    torch.set_num_threads(2)
    sys.path[:0] = [str(args.runtime/"src"), str(args.runtime)]
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.mdp import MDP
    from unisupermimic.tasks.motion_tracking.q1.supermimic.mdp_base import MDP as BaseMDP
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib

    library = json.loads(args.library.read_text())
    rows = library["splits"]["train"]+library["splits"]["validation"]
    kin = Q1Kinematics(args.runtime/"q1_assets/scene.xml", "cpu")
    motion = MotionLib([row["path"] for row in rows], kin, "cpu", src_fps=30.)
    adapter = json.loads((args.runtime/"q1_adapter.json").read_text())
    default = torch.tensor([adapter["control"][name]["default"] for name in adapter["joints"]])
    original, candidate = (cls(kin, motion, default, observation_mode="root_xy") for cls in (BaseMDP, MDP))
    candidate_trace = trace(args.candidate_evaluation)
    sources = [args.library, args.runtime/"candidate_source.json", args.candidate_evaluation/"final_trajectory.npz"]
    results = {}
    with torch.inference_mode():
        for specification in args.baseline:
            label, directory = specification.split("=", 1)
            if not label or label in results:
                raise ValueError("Baseline labels must be nonempty and unique")
            folder = Path(directory)
            baseline = trace(folder)
            initial_error = float(np.max(np.abs(baseline["initial"]-candidate_trace["initial"])))
            if baseline["initial"].shape != (len(rows), 57) or initial_error > 2e-6:
                raise ValueError("Baseline physical initial states differ")
            compared = []
            for index, row in enumerate(rows):
                count = min(int(baseline["steps"][index]), int(candidate_trace["steps"][index]))
                frames = baseline["frames"][:count, index]
                if count < 1 or not np.array_equal(frames, candidate_trace["frames"][:count, index]):
                    raise ValueError("Different reference timeline: "+row["name"])
                expected = np.arange(int(motion.clip_start[index])+1, int(motion.clip_start[index])+count+1)
                if not np.array_equal(frames, expected):
                    raise ValueError("Trace is not the reference's contiguous frame-zero prefix")
                item = {"name": row["name"], "split": row["split"], "group": row["group"], "steps": count}
                f = torch.tensor(frames)
                zero = torch.zeros(count, 22)
                for name, saved in (("baseline", baseline), ("candidate", candidate_trace)):
                    state = unflatten_state(torch.tensor(saved["states"][:count, index]))
                    old = original.reward(state, f, zero, zero, terms=True)[1]["track"]/.1
                    new_terms = candidate.reward(state, f, zero, zero, terms=True)[1]
                    new = new_terms["track"]/.1
                    if not bool(torch.isfinite(old).all() and torch.isfinite(new).all()):
                        raise ValueError("Nonfinite analytic tracking score")
                    error = state[4]-motion.q[f]
                    bias = error[:, 14:].mean(0)
                    item[name] = {"old_tracking_fraction": float(old.mean()),
                                  "new_tracking_fraction": float(new.mean()),
                                  "upper_rmse_rad": float(error[:, 14:].square().mean().sqrt()),
                                  "upper_rms_temporal_bias_rad": float(bias.square().mean().sqrt()),
                                  "leg_rmse_rad": float(error[:, :12].square().mean().sqrt()),
                                  "upper_joint_reward_fraction": float((new_terms["joint_upper"]/.01).mean())}
                compared.append(item)
            results[label] = {"initial_state_max_abs": initial_error, "summary": summarize(compared), "per_clip": compared}
            sources.append(folder/"final_trajectory.npz")
    if torch.cuda.is_initialized():
        raise AssertionError("Saved-state scoring should remain CPU-only")
    result = {"status": "SAME_STATE_OLD_NEW_TRACKING_HEAD_COMPARISON_COMPLETED",
              "created_utc": datetime.now(timezone.utc).isoformat(), "comparisons": results,
              "sources": [identity(path) for path in sources], "source": identity(__file__),
              "physics_integration_steps": 0, "policy_inference_calls": 0, "optimizer_updates": 0,
              "normalization": "Analytic tracking / 0.1 maximum per 20ms, equal clip weights on pairwise common executed prefixes",
              "limits": "Excludes action-rate and joint-limit penalties. Higher new reward alone does not establish improved tracking. Baseline durations and nondeterministic rollout variation must be disclosed."}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "summary": {label: value["summary"] for label, value in results.items()}}), flush=True)


if __name__ == "__main__":
    main()
