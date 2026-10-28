"""Physical PPO six-panel dashboard and fixed initial/final54s evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_ppo"))
from resources import save, stamp, identity


def dashboard(trial):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    target = trial / "dashboard"
    target.mkdir(exist_ok=True)
    source = trial / "run/training_evidence.jsonl"
    frame = pd.DataFrame(json.loads(line) for line in source.read_text().splitlines())
    episodes = frame.completed_episodes.replace(0, np.nan)
    frame["episode_normalized_return"] = frame.normalized_return_sum / episodes
    frame["hard_termination_fraction"] = frame.hard_terminations / episodes
    frame["completed_episode_steps"] = frame.length_steps_sum / episodes
    frame["normalized_step_reward"] = frame.normalized_step_reward_sum / frame.transitions
    frame["residual_rms_rad"] = np.sqrt(frame.residual_squared_sum / (14*frame.transitions))
    frame["mean_alpha"] = frame.alpha_sum / frame.transitions
    panels = (("episode_normalized_return", "Completed Episode Normalized Return"),
              ("hard_termination_fraction", "Hard Terminations / Completed Episodes"),
              ("completed_episode_steps", "Completed Episode Length (steps)"),
              ("normalized_step_reward", "Step Reward / Maximum (0.15)"),
              (None, "World Model Step-8 Error"), (None, "Imagined Endpoint Survival"))
    fig, axes = plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for axis,(key,title) in zip(axes.flat, panels):
        axis.set_title(title, fontsize=11)
        if key:
            frame[key+"_mean500"] = frame[key].rolling(500,min_periods=1).mean()
            axis.plot(frame.step, frame[key], linewidth=.6, alpha=.3, color="#23765e", label="Raw")
            axis.plot(frame.step, frame[key+"_mean500"], linewidth=1.8, color="#23765e", label="500-iteration mean")
            axis.legend(fontsize=8)
        else:
            axis.text(.5,.5,"N/A - physical PPO",ha="center",va="center",transform=axis.transAxes)
        axis.set_xlabel("PPO iteration (24 x 4096 transitions)")
        axis.grid(alpha=.2)
    seconds = float(frame.training_seconds.iloc[-1])
    fig.suptitle(f"RF16 | lower residual14 | 4096 environments | minibatch24576 | {seconds:.1f} training seconds",fontsize=12)
    for suffix in ("png","pdf"):
        fig.savefig(target / ("training_dashboard."+suffix),dpi=160)
    plt.close(fig)
    frame.to_csv(target / "training_dashboard.csv",index=False)
    fig,axes = plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for axis,key in zip(axes.flat,("ppo_value","ppo_surrogate","residual_rms_rad","mean_alpha")):
        axis.plot(frame.step,frame[key],linewidth=.8,color="#9f4e34")
        axis.set_title(key)
        axis.grid(alpha=.2)
    fig.savefig(target / "ppo_diagnostics.png",dpi=160)
    plt.close(fig)
    fig,axes = plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for axis,key in zip(axes.flat,("policy_update_kl_mean","policy_update_kl_p99","learning_rate","latent_mean_rms")):
        axis.plot(frame.step,frame[key],linewidth=.8,color="#23765e")
        axis.set_title(key)
        axis.grid(alpha=.2)
    fig.savefig(target / "ppo_update_diagnostics.png",dpi=160)
    plt.close(fig)
    fig,axis = plt.subplots(figsize=(11,5),constrained_layout=True)
    for key in ("original_reward_sum","stance_bonus_sum","swing_bonus_sum","stationary_bonus_sum","smooth_cost_sum","residual_cost_sum"):
        axis.plot(frame.step,frame[key]/frame.transitions,linewidth=.8,label=key.removesuffix("_sum"))
    axis.legend(fontsize=8)
    axis.set_xlabel("PPO iteration")
    axis.set_ylabel("Per physical step (costs shown positive)")
    axis.grid(alpha=.2)
    fig.savefig(target / "reward_decomposition.png",dpi=160)
    plt.close(fig)
    save(target / "receipt.json", {"status":"RF16_DASHBOARD_COMPLETED","source":identity(source),
        "training_seconds":seconds,"rolling_window_iterations":500,"reward_maximum":.15,
        "not_applicable":["world model step8","imagined survival"],
        "artifacts":[identity(p) for p in sorted(target.iterdir()) if p.name != "receipt.json"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime","trial","video-suite"):
        parser.add_argument("--"+name,type=Path,required=True)
    a = parser.parse_args()
    status = a.trial / "delivery_status.json"
    save(status,{"status":"RUNNING","phase":"dashboard","updated_utc":stamp()})
    dashboard(a.trial)
    sys.path[:0] = [str(a.runtime / "src"),str(a.runtime),str(HERE.parent / "supermimic_q1_hp")]
    import torch
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
    from video_suite import render_suite
    torch.set_num_threads(8)
    evaluation = a.trial / "evaluation"
    receipt = json.loads((evaluation / "receipt.json").read_text())
    rows = receipt["rows"]
    kin = Q1Kinematics(a.runtime / "q1_assets/scene.xml","cpu")
    motion = MotionLib([r["path"] for r in rows],kin,"cpu",src_fps=30.)
    env = SimpleNamespace(supermimic_mdp=SimpleNamespace(motion=motion),device=torch.device("cpu"))
    suite = json.loads(a.video_suite.read_text())
    media = {}
    versions = (("base",receipt["base_traces"][0]), ("initial",receipt["initial_traces"][0]),
                ("final",receipt["final_traces"][0]), ("recommended",receipt["recommended_traces"][0]))
    for label,source in versions:
        save(status,{"status":"RUNNING","phase":label+"_video","updated_utc":stamp()})
        with np.load(source,allow_pickle=False) as saved:
            trace = {key:saved[key] for key in ("initial","states","steps","failed")}
        media[label] = render_suite(a.runtime,trace,env,rows,suite,evaluation / (label+".mp4"),
                                    "RF16 "+label+(f" static upper{receipt['upper_reference_alpha']:.2f}" if label == "initial" else " gait compensation"))
        save(evaluation / (label+"_media.json"),media[label])
    save(status,{"status":"COMPLETED","updated_utc":stamp(),"new_optimizer_steps":0,"new_physical_steps":0,"media":media})

