"""Bounded Q1 runs using the pinned HP learner and Manager-Based Warp backend."""

from __future__ import annotations

import argparse
from dataclasses import fields
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def identity(path):
    path = Path(path)
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def process_identity():
    return {"pid": os.getpid(), "start_ticks": int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def prepare(args, split="train", num_envs=None, eval_mode=False):
    sys.path[:0] = [str(args.runtime / "src"), str(args.runtime)]
    import torch
    from omegaconf import OmegaConf
    from unilab.base.config_adapter import create_env
    from unilab.base.process_device import configure_backend_process_device
    from unilab.training.seed import apply_configured_training_seed
    from unisupermimic.tasks.motion_tracking.q1.supermimic import registry
    from unisupermimic.scripts.train_supermimic import build_env_override, build_learner_config

    cfg = OmegaConf.load(args.runtime / "q1_config.yaml")
    library = json.loads(args.library.read_text())
    rows = library["splits"][split]
    for row in rows:
        if identity(row["path"])["sha256"] != row["sha256"]:
            raise ValueError("Frozen Q1 source changed")
    cfg.env.commands.motion.motion_files = [r["path"] for r in rows]
    cfg.algo.seed = 0
    cfg.env.seed = 0
    cfg.algo.iterations = 320000
    cfg.training.log_dir = str(args.output)
    apply_configured_training_seed(cfg, torch_runtime=True, cuda=True)
    configure_backend_process_device("mjwarp", "cuda:0")
    override = build_env_override(cfg, eval_mode=eval_mode, learner_device="cuda:0")
    env = create_env(cfg, num_envs=num_envs or int(cfg.algo.num_envs), env_cfg_override=override)
    return cfg, env, build_learner_config(cfg)


def check_contract(env):
    import torch
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state, yaw_mat
    from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY

    if env.state is None:
        env.init_state()
    info = env.state.info[SUPERMIMIC_STATE_KEY]
    mdp = env.supermimic_mdp
    state = unflatten_state(info["s"])
    obs = mdp.obs(state, info["a_prev"], info["xi"], info["f"])
    goal = mdp.goal(info["f"], 10)
    spec = mdp.policy_input_spec
    assert spec["observation_mode"] == "root_xy"
    joint_error = "reference_minus_robot_joint_position" in spec["observation_fields"]
    expected_obs_dim = 102 if joint_error else 80
    assert obs.shape == (env.num_envs, expected_obs_dim) and goal.shape == (env.num_envs, 590)
    assert spec["obs_dim"] == expected_obs_dim
    assert info["xi"].shape == (env.num_envs, 26) and info["s"].shape == (env.num_envs, 57)
    expected = (yaw_mat(state[1]).transpose(-1,-2) @ (mdp.motion.root_p[info["f"]] - state[0]).unsqueeze(-1)).squeeze(-1)[..., :2]
    torch.testing.assert_close(obs[..., 78:80], expected)
    if joint_error:
        torch.testing.assert_close(obs[..., 80:102], mdp.motion.q[info["f"]]-state[4])
    future = mdp.motion.future_frames(info["f"], 10)
    end = mdp.motion.clip_end[mdp.motion.clip_id[info["f"]]]
    torch.testing.assert_close(future[:, 0], torch.minimum(info["f"]+1, end))
    before = {k: v.clone() for k, v in info.items()}
    action = torch.zeros(env.num_envs, 22, device=env.device)
    stepped = env.step(action)
    transition = stepped.info[SUPERMIMIC_TRANSITION_KEY]
    torch.testing.assert_close(transition["s"], before["s"])
    torch.testing.assert_close(transition["f"], before["f"])
    expected_reward = mdp.reward(unflatten_state(transition["s1"]), transition["f"]+1, transition["a"], transition["a_prev"])
    torch.testing.assert_close(transition["r"], expected_reward, atol=2e-6, rtol=2e-5)
    if not all(torch.isfinite(v).all() for v in transition.values()):
        raise ValueError("Nonfinite physical transition")
    return {"status": "Q1_REAL_GPU_OBSERVATION_GOAL_TRANSITION_CONTRACT_PASSED", "num_envs": env.num_envs, "obs_dim": expected_obs_dim, "goal_dim": 590, "state_dim": 57, "context_dim": 26, "root_feedback": "heading XY and full relative orientation in observation history", "current_joint_error": joint_error, "future_frames": "current+1 through current+10; full relative root position/rotation", "reward_frame": "action result compared against f+1", "physical_control_dt": .02}


def learner_class():
    import torch
    from unisupermimic.rl.algos.supermimic.learner import SuperMimicLearner

    class BudgetReached(Exception):
        pass

    class BoundedLearner(SuperMimicLearner):
        def __init__(self, *args, budget_seconds=None, maximum_seconds=3600, **kwargs):
            super().__init__(*args, **kwargs)
            self.budget_seconds = budget_seconds
            self.maximum_seconds = maximum_seconds
            self.started_monotonic = None
            self.stop_requested = False
            self.wm_updates = self.actor_updates = 0
            self.last_reported = 0
            self.latest_wm = self.latest_policy = None
            self.reporting = dict(completed_episodes=0, hard_terminations=0, normalized_return_sum=0., length_steps_sum=0., reward_sum=0., transitions=0)

        def elapsed(self):
            return 0 if self.started_monotonic is None else time.monotonic() - self.started_monotonic

        def flush_reporting(self):
            if not self.budget_seconds or self.iteration <= self.last_reported:
                return
            row = {"step": self.iteration, "training_seconds": self.elapsed(), "sim_steps": self.sim_steps, "wm_updates": self.wm_updates, "actor_updates": self.actor_updates, **self.reporting}
            row["reward_mean"] = row["reward_sum"]/max(row["transitions"], 1)
            row["wm_err8"] = self.latest_wm[1] if self.latest_wm else None
            row["imag_survival_H"] = self.latest_policy[4] if self.latest_policy else None
            with (self.log_dir / "training_evidence.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
            self.last_reported = self.iteration
            self.reporting = {k: (0. if isinstance(v, float) else 0) for k,v in self.reporting.items()}

        def collect(self, steps):
            self.flush_reporting()
            if self.started_monotonic is None:
                self.started_monotonic = time.monotonic()
                save(self.log_dir / "training_clock.json", {"started_utc": now(), "started_monotonic": self.started_monotonic, "process": process_identity(), "maximum_seconds": self.maximum_seconds, "soft_stop_seconds": self.budget_seconds})
            if self.stop_requested or (self.budget_seconds and self.elapsed() >= self.budget_seconds):
                raise BudgetReached()
            original_step = self.env.step
            warming_up = self.sim_steps == 0
            def observed_step(action):
                from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_TRANSITION_KEY
                state = original_step(action)
                transition = state.info[SUPERMIMIC_TRANSITION_KEY]
                if self.budget_seconds and not warming_up:
                    report = self.reporting
                    report["transitions"] += self.env.num_envs
                    report["reward_sum"] += float(transition["r"].sum())
                    count = int(transition["done"].sum())
                    report["completed_episodes"] += count
                    report["hard_terminations"] += int(transition["term"].sum())
                    if count:
                        if transition["ep_len"].numel() != count:
                            raise ValueError("Completed episode reporting contract changed")
                        report["normalized_return_sum"] += float(transition["ep_ret_norm"].sum())
                        report["length_steps_sum"] += float(transition["ep_len"].sum())
                return state
            self.env.step = observed_step
            try:
                result = super().collect(steps)
            finally:
                self.env.step = original_step
            save(self.log_dir / "status.json", {"status": "RUNNING", "updated_utc": now(), "process": process_identity(), "iteration": self.iteration, "sim_steps": self.sim_steps, "wm_updates": self.wm_updates, "actor_updates": self.actor_updates, "training_seconds": self.elapsed()})
            return result

        def wm_update(self):
            result = super().wm_update()
            if not all(torch.isfinite(x).all() for x in result):
                raise FloatingPointError("Nonfinite world model update")
            self.wm_updates += 1
            self.latest_wm = tuple(float(x) for x in result)
            return result

        def policy_update(self):
            result = super().policy_update()
            if not all(torch.isfinite(x).all() for x in result):
                raise FloatingPointError("Nonfinite policy or critic update")
            self.actor_updates += 1
            self.latest_policy = tuple(float(x) for x in result)
            return result

        def _sample_valid_starts(self, batch_size):
            # Same rejection sampler as HP, bounded for a library/replay with no
            # legal H32 start. This changes no successful sample distribution.
            starts, histories, count = [], [], 0
            for _ in range(128):
                start, history = self.buffer.sample_starts(2*batch_size, self.cfg.hist)
                valid = self.mdp.valid_imagination_start(start["f"], self.cfg.horizon)
                ids = valid.nonzero(as_tuple=False).squeeze(-1)[:batch_size-count]
                starts.append({k: v[ids] for k, v in start.items()})
                histories.append({k: v[ids] for k, v in history.items()})
                count += len(ids)
                if count == batch_size:
                    return tuple({k: torch.cat([p[k] for p in rows]) for k in rows[0]} for rows in (starts, histories))
            raise RuntimeError("No sufficient legal H32 replay starts after 128 proposals")

    return BoundedLearner, BudgetReached


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "profile", "train"), required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--code-commit")
    p.add_argument("--training-seconds",type=float,default=3500)
    p.add_argument("--maximum-training-seconds",type=float,default=3600)
    args = p.parse_args()
    if not 0 < args.training_seconds < args.maximum_training_seconds:
        p.error("Training soft target must be positive and below the external maximum")
    args.output.mkdir(parents=True, exist_ok=False)
    env = None
    status = {"status": "INITIALIZING", "started_utc": now(), "process": process_identity(), "mode": args.mode, "code_commit": args.code_commit, "library": identity(args.library)}
    save(args.output / "status.json", status)
    try:
        import torch
        from omegaconf import OmegaConf
        torch.set_num_threads(8)
        cfg, env, learner_cfg = prepare(args, num_envs=16 if args.mode == "smoke" else 4096)
        OmegaConf.save(cfg, args.output / "config.yaml", resolve=True)
        contract = check_contract(env)
        save(args.output / "contract.json", contract)
        if args.mode == "smoke":
            status.update(status="SMOKE_COMPLETED", contract=contract, optimizer_updates=0)
        else:
            BoundedLearner, BudgetReached = learner_class()
            learner = BoundedLearner(env, learner_cfg, args.output, device="cuda:0", backend="mjwarp", budget_seconds=args.training_seconds if args.mode == "train" else None, maximum_seconds=args.maximum_training_seconds)
            torch.cuda.reset_peak_memory_stats()
            if args.mode == "profile":
                started = time.monotonic()
                learner.collect(64)
                wm_values = [tuple(float(x) for x in learner.wm_update()) for _ in range(10)]
                policy_values = [float(x) for x in learner.policy_update()]
                torch.cuda.synchronize()
                status.update(status="PROFILE_COMPLETED", elapsed_seconds=time.monotonic()-started, sim_steps=learner.sim_steps, wm_updates=learner.wm_updates, actor_updates=learner.actor_updates, wm_values=wm_values, policy_values=policy_values, actual_config={"num_envs": env.num_envs, "wm_batch": learner_cfg.wm_batch, "pi_batch": learner_cfg.pi_batch, "horizon": learner_cfg.horizon, "wm_batches": learner_cfg.wm_batches, "buffer_steps": learner_cfg.buffer_steps, "amp": learner_cfg.amp, "compile": learner_cfg.compile})
            else:
                learner.save(args.output / "initial.pt", include_optimizers=False)
                def request_stop(signum, frame):
                    learner.stop_requested = True
                signal.signal(signal.SIGTERM, request_stop)
                signal.signal(signal.SIGINT, request_stop)
                try:
                    learner.train()
                except BudgetReached:
                    pass
                elapsed = learner.elapsed()
                learner.wall_time = elapsed
                learner.save(args.output / "final.pt")
                learner.save(args.output / "last.pt")
                if learner.stop_requested:
                    raise InterruptedError("External stop requested")
                if not args.training_seconds <= elapsed <= args.maximum_training_seconds:
                    raise RuntimeError(f"Actual training budget outside [{args.training_seconds},{args.maximum_training_seconds}]: {elapsed}")
                status.update(status="TRAINING_COMPLETED", training_seconds=elapsed, iteration=learner.iteration, sim_steps=learner.sim_steps, wm_updates=learner.wm_updates, actor_updates=learner.actor_updates, final=identity(args.output / "final.pt"), initial=identity(args.output / "initial.pt"))
            status.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:
        status.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if env is not None:
            env.close()
            env.backend.close()
        status["ended_utc"] = now()
        save(args.output / "status.json", status)
        save(args.output / "receipt.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
