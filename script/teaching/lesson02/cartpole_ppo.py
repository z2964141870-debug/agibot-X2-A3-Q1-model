"""YUANQI lesson 02: train/reload the installed Isaac Lab Cartpole PPO recipe.

Imports the existing BSD-3-Clause Isaac Lab and RSL-RL implementations; does
not copy or modify their PPO or environment code. All outputs stay in YUANQI.
"""

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=["train", "evaluate"], default="train")
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--log-dir", type=Path, required=True)
parser.add_argument("--iterations", type=int, default=150)
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--eval-seed", type=int, default=10001)
parser.add_argument("--pole-angle-weight", type=float, default=1.0,
                    help="Positive magnitude of the pole-angle penalty; reward term is -weight * angle_rad**2.")
parser.add_argument("--checkpoint", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not math.isfinite(args.pole_angle_weight) or args.pole_angle_weight <= 0:
    parser.error("pole-angle-weight must be finite and positive")
args.run_dir = args.run_dir.resolve()
args.log_dir = args.log_dir.resolve()
if args.mode == "train" and args.run_dir.exists():
    raise FileExistsError(f"Choose a new run directory: {args.run_dir}")
if args.mode == "evaluate" and (args.checkpoint is None or not args.checkpoint.is_file()):
    raise ValueError("Evaluation requires an existing checkpoint.")
args.run_dir.mkdir(parents=True, exist_ok=True)
args.log_dir.mkdir(parents=True, exist_ok=True)
app = AppLauncher(args).app

import copy
import hashlib
import importlib.metadata
import json
import subprocess
import time
import traceback

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

import isaaclab_tasks  # noqa: F401; registers the official environment
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

TASK = "Isaac-Cartpole-Direct-v0"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


class ArtifactRunner(OnPolicyRunner):
    """Keep checkpoints in data/ and TensorBoard/terminal records in logs/."""

    def save(self, path, infos=None):
        destination = args.run_dir / "checkpoints" / Path(path).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        super().save(str(destination), infos)


@torch.inference_mode()
def evaluate(env, policy, label):
    """One first episode per environment, including every early failure.

    Identical reset seed/count for before/after/reload. Later auto-reset
    episodes do not contribute, avoiding short-episode selection bias.
    """
    policy.eval()
    raw_obs, _ = env.env.reset(seed=args.eval_seed)
    obs = TensorDict(raw_obs, batch_size=[env.num_envs])
    initial = obs["policy"].clone()
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    returns = torch.zeros(env.num_envs, device=env.device)
    lengths = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    failed = torch.zeros_like(active)
    angle_sum = torch.zeros_like(returns)
    trace = []
    for step in range(env.max_episode_length + 2):
        actions = policy.act_inference(obs)
        if env.clip_actions is not None:
            actions = actions.clamp(-env.clip_actions, env.clip_actions)
        if active[0].item():
            trace.append({"step": step, "observation": obs["policy"][0].tolist(),
                          "action": actions[0].tolist()})
        angle_sum += obs["policy"][:, 0].abs() * active
        raw_obs, rewards, terminated, truncated, _ = env.env.step(actions)
        returns += rewards * active
        lengths += active.long()
        failed |= active & terminated
        active &= ~(terminated | truncated)
        obs = TensorDict(raw_obs, batch_size=[env.num_envs])
        if not active.any().item():
            break
    if active.any().item():
        raise RuntimeError("Evaluation ended with unfinished first episodes.")
    result = {
        "label": label, "eval_seed": args.eval_seed, "episodes": env.num_envs,
        "control_dt_s": env.unwrapped.step_dt,
        "time_limit_s": env.cfg.episode_length_s,
        "mean_return": returns.mean().item(),
        "mean_episode_steps": lengths.float().mean().item(),
        "mean_survival_s": lengths.float().mean().item() * env.unwrapped.step_dt,
        "survived_time_limit_fraction": (~failed).float().mean().item(),
        "mean_abs_pole_angle_rad_before_termination": (angle_sum / lengths).mean().item(),
        "initial_observations_sha256": hashlib.sha256(initial.cpu().numpy().tobytes()).hexdigest(),
        "evaluation_policy": "deterministic actor mean; first episode per reset; no parameter updates",
        "per_episode_return": returns.tolist(), "per_episode_steps": lengths.tolist(),
        "per_episode_failed": failed.tolist(), "trace_env_0": trace,
    }
    write_json(args.run_dir / f"evaluation_{label}.json", result)
    print("YUANQI_EVALUATION " + json.dumps({k: v for k, v in result.items()
          if not k.startswith("per_episode") and k != "trace_env_0"}), flush=True)
    return result


def main():
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device or "cuda:0"
    env_cfg.rew_scale_pole_pos = -args.pole_angle_weight
    agent_cfg.seed = args.seed
    agent_cfg.device = env_cfg.sim.device
    agent_cfg.logger = "tensorboard"
    agent_cfg.max_iterations = args.iterations
    agent_cfg.save_interval = 50
    train_cfg = agent_cfg.to_dict()
    config_path = args.run_dir / "run_config.json"
    config = {
        "task": TASK, "seed": args.seed, "eval_seed": args.eval_seed,
        "num_envs": args.num_envs, "iterations": args.iterations,
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        "device": agent_cfg.device,
        "pole_angle_weight": args.pole_angle_weight,
        "rew_scale_pole_pos": env_cfg.rew_scale_pole_pos,
        "asset_usd": env_cfg.robot_cfg.spawn.usd_path,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "packages": {p: importlib.metadata.version(p) for p in
                     ["torch", "rsl-rl-lib", "isaacsim", "isaaclab", "gymnasium"]},
        "isaaclab_commit": subprocess.check_output(
            ["git", "-C", "/home/yu/projects/IsaacLab", "rev-parse", "HEAD"], text=True).strip(),
    }
    if args.mode == "evaluate":
        original = json.loads(config_path.read_text())
        for key in ["task", "seed", "eval_seed", "num_envs", "device", "script_sha256",
                    "packages", "isaaclab_commit", "asset_usd",
                    "pole_angle_weight", "rew_scale_pole_pos"]:
            if original[key] != config[key]:
                raise ValueError(f"Evaluation configuration differs: {key}")
    else:
        write_json(config_path, config)
        dump_yaml(str(args.run_dir / "environment.yaml"), env_cfg)
        dump_yaml(str(args.run_dir / "agent.yaml"), agent_cfg)
    env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    try:
        runner = ArtifactRunner(env, copy.deepcopy(train_cfg), log_dir=str(args.log_dir),
                                device=agent_cfg.device)
        if args.mode == "evaluate":
            runner.load(str(args.checkpoint), load_optimizer=False)
            result = evaluate(env, runner.alg.policy, "reloaded")
            reference = json.loads((args.run_dir / "evaluation_trained.json").read_text())
            if result["initial_observations_sha256"] != reference["initial_observations_sha256"]:
                raise AssertionError("Reload evaluation did not use identical initial states.")
            probes = torch.load(args.run_dir / "policy_probes.pt", weights_only=True,
                                map_location=agent_cfg.device)
            with torch.inference_mode():
                actions = runner.alg.policy.act_inference(
                    TensorDict({"policy": probes["observations"]}, batch_size=[len(probes["observations"])]))
            max_error = (actions - probes["actions"]).abs().max().item()
            if max_error > 1e-6:
                raise AssertionError(f"Reload action mismatch: {max_error}")
            write_json(args.run_dir / "reload_verification.json", {
                "separate_process": True, "checkpoint": str(args.checkpoint),
                "max_probe_action_error": max_error,
                "initial_states_match": True,
                "mean_return_delta": result["mean_return"] - reference["mean_return"],
                "success_fraction_delta": result["survived_time_limit_fraction"] - reference["survived_time_limit_fraction"],
            })
            print("YUANQI_RELOAD_VERIFIED max_action_error=" + str(max_error), flush=True)
            return

        model = runner.alg.policy
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        checkpoint_dir = args.run_dir / "checkpoints"
        checkpoint_dir.mkdir()
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": runner.alg.optimizer.state_dict(),
                    "iter": 0, "infos": {"phase": "before_training"}}, checkpoint_dir / "initial.pt")
        initial = evaluate(env, model, "initial")
        with torch.inference_mode():
            env.seed(args.seed)
            env.reset()
        started = time.monotonic()
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        duration = time.monotonic() - started
        runner.save("final.pt", infos={"phase": "after_training", "iterations": args.iterations})
        changed = {}
        for group in ["actor", "critic"]:
            differences = [(param - before[name]).abs().max().item()
                           for name, param in model.named_parameters() if name.startswith(group + ".")]
            changed[group] = {"max_parameter_change": max(differences),
                              "changed_tensors": sum(value > 0 for value in differences)}
            if changed[group]["max_parameter_change"] <= 0:
                raise AssertionError(f"No verified {group} update")
        write_json(args.run_dir / "training_summary.json", {
            "iterations": args.iterations, "transitions": args.iterations * args.num_envs * runner.num_steps_per_env,
            "train_wall_seconds": duration, "parameter_updates": changed,
            "training_seed_count": 1, "phase": "teaching_baseline_not_research_validation",
        })
        trained = evaluate(env, model, "trained")
        if initial["initial_observations_sha256"] != trained["initial_observations_sha256"]:
            raise AssertionError("Before/after initial states differ")
        probe_obs = env.get_observations()["policy"].clone()
        with torch.inference_mode():
            probe_actions = model.act_inference(TensorDict({"policy": probe_obs}, batch_size=[env.num_envs]))
        torch.save({"observations": probe_obs, "actions": probe_actions}, args.run_dir / "policy_probes.pt")
        runner.writer.flush()
        runner.writer.close()
        print("YUANQI_TRAIN_COMPLETE " + json.dumps(changed), flush=True)
    finally:
        env.close()


try:
    main()
except BaseException:
    error = traceback.format_exc()
    (args.log_dir.parent / f"error_{args.mode}.txt").write_text(error)
    print(error, flush=True)
    raise
finally:
    app.close()
