"""RF13 plots and fixed original-speed videos from existing saved evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_ppo"))
from resources import identity, save, stamp


def dashboard(trial):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    output = trial / "dashboard"
    output.mkdir(exist_ok=True)
    source = trial / "run/training_evidence.jsonl"
    evidence = pd.DataFrame(json.loads(line) for line in source.read_text().splitlines())
    if evidence.empty:
        raise ValueError("No completed update evidence")
    episodes = evidence.completed_episodes.replace(0, np.nan)
    evidence["episode_normalized_return"] = evidence.normalized_return_sum / episodes
    evidence["hard_termination_per_completed_episode"] = evidence.hard_terminations / episodes
    evidence["completed_episode_steps"] = evidence.length_steps_sum / episodes
    evidence["step_reward_over_maximum"] = evidence.normalized_step_reward_sum / evidence.transitions
    evidence["mean_applied_alpha"] = evidence.alpha_sum / evidence.transitions
    evidence["fallback_fraction"] = evidence.fallback_rows / evidence.transitions
    evidence["residual_rms_rad"] = np.sqrt(evidence.residual_squared_sum / (14 * evidence.transitions))
    evidence["lower_action_KL_proxy"] = evidence.lower_action_kl_proxy_sum / evidence.transitions
    metrics = (("episode_normalized_return", "Completed Episode Normalized Return"),
        ("hard_termination_per_completed_episode", "Hard Terminations / Completed Episodes"),
        ("completed_episode_steps", "Completed Episode Length (steps)"),
        ("step_reward_over_maximum", "Step Reward / Maximum (0.13)"),
        (None, "World Model Step-8 Error"), (None, "Imagined Endpoint Survival"))
    for key, _ in metrics:
        if key:
            evidence[key + "_mean500"] = evidence[key].rolling(500, min_periods=1).mean()
    evidence.to_csv(output / "training_dashboard.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, metrics):
        axis.set_title(title, fontsize=11)
        if key is None:
            axis.text(.5, .5, "N/A - physical PPO", ha="center", va="center", transform=axis.transAxes)
        else:
            axis.plot(evidence.step, evidence[key], linewidth=.6, alpha=.3, color="#187b80", label="Raw")
            axis.plot(evidence.step, evidence[key + "_mean500"], linewidth=1.8, color="#187b80", label="500-iteration mean")
            axis.legend(fontsize=8)
        axis.set_xlabel("PPO iteration (24 x 4096 transitions)")
        axis.grid(alpha=.2)
    seconds = float(evidence.training_seconds.iloc[-1])
    fig.suptitle(f"RF13 | residual14 | 4096 environments | minibatch24576 | {seconds:.1f} actual training seconds", fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(output / ("training_dashboard." + suffix), dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, key, title in zip(axes.flat, ("ppo_value", "ppo_surrogate", "mean_applied_alpha", "residual_rms_rad", "lower_action_KL_proxy", "curriculum_cap"),
        ("Value Loss", "PPO Surrogate", "Applied Alpha", "Lower Residual RMS (rad)", "Fixed Action-Gaussian KL Proxy", "Upper Intervention Cap")):
        axis.plot(evidence.step, evidence[key], linewidth=.8, color="#a35b31")
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("PPO iteration")
        axis.grid(alpha=.2)
    fig.savefig(output / "ppo_diagnostics.png", dpi=160)
    plt.close(fig)
    reward_columns = ("original_reward_sum", "root_bonus_sum", "leg_bonus_sum", "residual_cost_sum")
    fig, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for key in reward_columns:
        axis.plot(evidence.step, evidence[key] / evidence.transitions, linewidth=.8, label=key.removesuffix("_sum"))
    axis.set_xlabel("PPO iteration")
    axis.set_ylabel("Mean per physical step")
    axis.legend()
    axis.grid(alpha=.2)
    fig.savefig(output / "reward_decomposition.png", dpi=160)
    plt.close(fig)
    save(output / "receipt.json", {"status": "RF13_DASHBOARD_COMPLETED", "source": identity(source),
        "training_seconds": seconds, "through_iteration": int(evidence.step.iloc[-1]), "rolling_window_iterations": 500,
        "reward_maximum": .13, "episode_denominator": "completed episodes; not snapshot terminations",
        "not_applicable": ["world model step8", "imagined survival"],
        "KL_proxy_limit": "Equal0.1 action-unit Gaussians around lower physical action means; diagnostic, not latent policy KL or constraint.",
        "artifacts": [identity(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "receipt.json"]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "trial", "video-suite"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    status = a.trial / "delivery_status.json"
    save(status, {"status": "RUNNING", "phase": "dashboard", "updated_utc": stamp()})
    dashboard(a.trial)
    sys.path[:0] = [str(a.runtime / "src"), str(a.runtime), str(HERE.parent / "supermimic_q1_hp")]
    import torch
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
    from video_suite import render_suite
    torch.set_num_threads(8)
    evaluation = a.trial / "evaluation"
    receipt = json.loads((evaluation / "receipt.json").read_text())
    rows = receipt["rows"]
    kin = Q1Kinematics(a.runtime / "q1_assets/scene.xml", "cpu")
    motion = MotionLib([row["path"] for row in rows], kin, "cpu", src_fps=30.)
    env = SimpleNamespace(supermimic_mdp=SimpleNamespace(motion=motion), device=torch.device("cpu"))
    suite = json.loads(a.video_suite.read_text())
    media = {}
    for label, source in (("initial", receipt["initial_traces"][0]), ("final", receipt["final_traces"][0])):
        save(status, {"status": "RUNNING", "phase": label + "_video", "updated_utc": stamp()})
        with np.load(source, allow_pickle=False) as saved:
            trace = {key: saved[key] for key in ("initial", "states", "steps", "failed")}
        media[label] = render_suite(a.runtime, trace, env, rows, suite, evaluation / (label + ".mp4"),
                                    "RF13 " + label + (" zero residual, alpha0.75" if label == "initial" else " trained lower residual"))
        save(evaluation / (label + "_media.json"), media[label])
    save(status, {"status": "COMPLETED", "updated_utc": stamp(), "new_physical_steps": 0,
                  "new_optimizer_steps": 0, "media": media, "dashboard": identity(a.trial / "dashboard/receipt.json")})


if __name__ == "__main__":
    main()
