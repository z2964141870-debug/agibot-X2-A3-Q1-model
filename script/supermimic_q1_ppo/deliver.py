"""Plot PPO evidence and render fixed saved trajectories without new physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from resources import identity, save, stamp

HERE = Path(__file__).resolve().parent


def dashboard(trial, evidence_path=None, max_iteration=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    output = trial / "dashboard"
    output.mkdir(exist_ok=False)
    evidence_path = evidence_path or trial / "run/training_evidence.jsonl"
    evidence = pd.DataFrame(json.loads(line) for line in evidence_path.read_text().splitlines())
    if max_iteration is not None:
        evidence = evidence[evidence.step <= max_iteration]
    if evidence.empty:
        raise ValueError("No completed PPO iterations to plot")
    episodes = evidence["completed_episodes"].replace(0, np.nan)
    evidence["episode_normalized_return"] = evidence["normalized_return_sum"] / episodes
    evidence["hard_termination_per_completed_episode"] = evidence["hard_terminations"] / episodes
    evidence["completed_episode_steps"] = evidence["length_steps_sum"] / episodes
    evidence["step_reward_over_current_max"] = evidence["normalized_step_reward_sum"] / evidence["transitions"]
    metrics = (("episode_normalized_return", "Completed Episode Normalized Return"),
               ("hard_termination_per_completed_episode", "Hard Terminations / Completed Episodes"),
               ("completed_episode_steps", "Completed Episode Length (steps)"),
               ("step_reward_over_current_max", "Step Reward / Current Maximum"),
               (None, "World Model Step-8 Error"), (None, "Imagined Endpoint Survival"))
    for key, _ in metrics:
        if key:
            evidence[key + "_mean500"] = evidence[key].rolling(500, min_periods=1).mean()
    evidence.to_csv(output / "training_dashboard.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, metrics):
        axis.set_title(title, fontsize=11)
        if key is None:
            axis.text(.5, .5, "N/A - real-environment PPO", ha="center", va="center", transform=axis.transAxes)
        else:
            axis.plot(evidence.step, evidence[key], linewidth=.6, alpha=.3, color="#137c82", label="Raw")
            axis.plot(evidence.step, evidence[key + "_mean500"], linewidth=1.8, color="#137c82", label="500-iteration mean")
            axis.legend(fontsize=8)
        axis.set_xlabel("PPO iteration (24 x 4096 transitions)")
        axis.grid(alpha=.2)
    seconds = float(evidence.training_seconds.iloc[-1])
    fig.suptitle(f"RF12 | 4096 environments | minibatch 24576 | {seconds:.1f} actual training seconds", fontsize=13)
    for suffix in ("png", "pdf"):
        fig.savefig(output / ("training_dashboard." + suffix), dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, key, title in zip(axes.flat,
                                 ("ppo_value", "ppo_surrogate", "reward_alpha", "new_motion_fraction"),
                                 ("Value Loss", "PPO Surrogate Loss", "Added Reward Ramp", "New Motion Sampling Mass")):
        axis.plot(evidence.step, evidence[key], linewidth=.8, color="#b25b26")
        axis.set_title(title)
        axis.set_xlabel("PPO iteration")
        axis.grid(alpha=.2)
    fig.savefig(output / "ppo_diagnostics.png", dpi=160)
    plt.close(fig)
    save(output / "receipt.json", {"status": "PPO_SIX_PANEL_AND_DIAGNOSTICS_COMPLETED", "training_seconds": seconds,
                                    "source_evidence": identity(evidence_path), "through_iteration": int(evidence.step.iloc[-1]),
                                    "iterations": len(evidence), "rolling_window_iterations": 500,
                                    "episode_denominator": "completed episodes, not instantaneous termination fraction",
                                    "reward_normalization": "Each step divided by its then-current maximum .1+alpha*.03; episode mean across its actual ramp history",
                                    "not_applicable": ["world model step-8 error", "imagined endpoint survival"],
                                    "artifacts": [identity(path) for path in sorted(output.iterdir()) if path.is_file()]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--video-suite", type=Path, required=True)
    parser.add_argument("--stage", type=int)
    args = parser.parse_args()
    original_trial = args.trial
    original_evaluation = original_trial / "evaluation"
    if args.stage is not None:
        decisions = json.loads((original_trial / "run/stage_decisions.json").read_text())
        decision = decisions[args.stage - 1]
        args.trial = original_trial / "stages" / f"stage_{args.stage:02d}"
        args.trial.mkdir(parents=True, exist_ok=False)
        (args.trial / "evaluation").mkdir()
        receipt = {"rows": json.loads((original_evaluation / "evaluation_library.json").read_text())["splits"]["train"],
                   "baseline_traces": [str(original_evaluation / f"RF11_baseline_repeat{r}.npz") for r in range(3)],
                   "final_traces": [item["path"] for item in decision["traces"]]}
        save(args.trial / "stage.json", decision)
        save(args.trial / "evaluation/receipt.json", receipt)
        cutoff = decision["iteration"]
    else:
        cutoff = None
    save(args.trial / "delivery_status.json", {"status": "RUNNING", "phase": "dashboard", "updated_utc": stamp()})
    dashboard(args.trial, original_trial / "run/training_evidence.jsonl", cutoff)
    sys.path[:0] = [str(args.runtime / "src"), str(args.runtime), str(HERE.parent / "supermimic_q1_hp")]
    import torch
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
    from video_suite import render_suite

    torch.set_num_threads(8)
    evaluation = args.trial / "evaluation"
    receipt = json.loads((evaluation / "receipt.json").read_text())
    rows = receipt["rows"]
    kin = Q1Kinematics(args.runtime / "q1_assets/scene.xml", "cpu")
    motion = MotionLib([row["path"] for row in rows], kin, "cpu", src_fps=30.)
    env = SimpleNamespace(supermimic_mdp=SimpleNamespace(motion=motion), device=torch.device("cpu"))
    suite = json.loads(args.video_suite.read_text())
    outputs = {}
    for label, path in (("initial", receipt["baseline_traces"][0]), ("final", receipt["final_traces"][0])):
        save(args.trial / "delivery_status.json", {"status": "RUNNING", "phase": label + "_video", "updated_utc": stamp()})
        with np.load(path, allow_pickle=False) as stored:
            trace = {key: stored[key] for key in ("initial", "states", "steps", "failed")}
        outputs[label] = render_suite(args.runtime, trace, env, rows, suite, evaluation / (label + ".mp4"), "RF12 " + label)
        save(evaluation / (label + "_media.json"), outputs[label])
    save(args.trial / "delivery_status.json", {"status": "COMPLETED", "updated_utc": stamp(),
                                               "new_physical_steps": 0, "new_optimizer_steps": 0, "media": outputs,
                                               "dashboard": identity(args.trial / "dashboard/receipt.json")})


if __name__ == "__main__":
    main()
