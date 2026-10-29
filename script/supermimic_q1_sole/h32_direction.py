"""Capture the native H32 actor direction and compare matched closed loops.

cpu-contract checks the objective and parameter/RNG protection on reference
poses. native-direction additionally uses an independent native Warp env.
No optimizer step or checkpoint write is permitted in either mode.
"""

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def digest(module):
    value = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value.update(name.encode("ascii"))
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(status="INITIALIZING", captured_utc=datetime.now(timezone.utc).isoformat(),
                  mode=args.mode, horizon=32, optimizer_updates=0, training_launched=False,
                  user_accepted_checkpoint=None)
    env = None
    cpu_rng = cuda_rng = None
    control_steps = 0
    try:
        import torch
        sys.path[:0] = [str(HERE), str(args.runtime / "src"), str(args.runtime),
                       str(HERE.parent / "project_tools"), str(HERE.parent / "supermimic_q1_hp")]
        from check_contract import identity
        from prepare_q1_gait_review import FIXED
        from unisupermimic.rl.algos.supermimic import networks
        from unisupermimic.rl.algos.supermimic.geometry import flatten_state, unflatten_state
        from unisupermimic.rl.algos.supermimic.learner import SuperMimicConfig, SuperMimicLearner
        from unisupermimic.rl.algos.supermimic.networks import WorldModel, Policy, Critic
        from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
        from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
        from unisupermimic.tasks.motion_tracking.q1.supermimic.mdp import MDP
        from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import default_joint_pos_list
        from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY

        torch.set_num_threads(8)
        cpu_rng = torch.get_rng_state().clone()
        device = torch.device("cpu" if args.mode == "cpu-contract" else "cuda:0")
        if device.type == "cuda":
            busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
            if busy:
                raise RuntimeError("GPU has an existing compute process; native probe was not launched")
            cuda_rng = torch.cuda.get_rng_state_all()
        torch.manual_seed(args.seed)
        scene = args.runtime / "q1_assets/scene.xml"
        library = json.loads(args.library.read_text())
        rows = [r for values in library["splits"].values() for r in values if r["name"] in FIXED]
        if len(rows) != 3 or len({r["name"] for r in rows}) != 3:
            raise ValueError("Expected the three original fixed clips")
        for row in rows:
            if identity(row["path"])["sha256"] != row["sha256"]:
                raise ValueError("Fixed reference changed")
        native = args.runtime / "src/unisupermimic"
        source_paths = [Path(__file__), args.checkpoint, args.library, scene,
                        native / "rl/algos/supermimic/learner.py",
                        native / "rl/algos/supermimic/networks.py",
                        native / "tasks/motion_tracking/q1/supermimic/mdp_base.py"]
        sources = [identity(path) for path in source_paths]
        saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
        cfg = SuperMimicConfig(**{f.name: saved["args"][f.name]
                                 for f in fields(SuperMimicConfig) if f.name in saved["args"]})
        if cfg.horizon != 32 or cfg.hist != 10 or cfg.no_critic:
            raise ValueError("Expected native H32/hist10 actor with a saved critic")
        cfg.pi_batch = args.envs
        cfg.compile = False
        networks.AMP_DTYPE = torch.bfloat16 if cfg.amp == "bf16" else None
        if device.type == "cpu":
            kin = Q1Kinematics(scene, device)
            motion = MotionLib([r["path"] for r in rows], kin, device, src_fps=30.)
            mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list()), observation_mode="root_xy")
        else:
            from types import SimpleNamespace
            from worker import prepare
            subset = args.output / "evaluation_library.json"
            subset.write_text(json.dumps({"splits": {"train": rows}}, indent=2) + "\n")
            preparation_args = SimpleNamespace(runtime=args.runtime, library=subset, output=args.output)
            _, env, _ = prepare(preparation_args, num_envs=args.envs, eval_mode=True)
            mdp = env.supermimic_mdp
            motion = mdp.motion

        learner = SuperMimicLearner.__new__(SuperMimicLearner)
        learner.cfg, learner.device, learner.mdp = cfg, device, mdp
        learner.pi = Policy(mdp.obs_dim, mdp.goal_frame_dim * cfg.goal_frames, hist=cfg.hist,
                            z_dim=cfg.z_dim, num_joints=22, hidden=cfg.policy_hidden).to(device)
        learner.wm = WorldModel(hidden=cfg.wm_hidden, num_joints=22, context_dim=26).to(device)
        sample_frames = motion.clip_start[:1]
        context = torch.zeros(1, 26, device=device)
        context[:, 0] = 1
        feature_dim = mdp.critic_features(motion.ref_state(sample_frames), sample_frames,
                                       mdp.normalize_context(context)).shape[-1] + mdp.goal_frame_dim * cfg.goal_frames
        learner.critic = Critic(feature_dim, hidden=cfg.critic_hidden).to(device)
        for name in ("pi", "wm", "critic"):
            getattr(learner, name).load_state_dict(saved[name])
        learner.wm.requires_grad_(False)
        learner.critic.requires_grad_(False)
        learner._imag_step_fn = learner._imag_step
        learner._critic_input_fn = learner._critic_input
        initial_policy = {k: v.clone() for k, v in learner.pi.state_dict().items()}
        protected = {name: digest(getattr(learner, name)) for name in ("pi", "wm", "critic")}
        base_parameters = [p.detach().clone() for p in learner.pi.parameters()]
        env_ids = torch.arange(args.envs, device=device)
        clips = env_ids % len(rows)
        offsets = 30 * (env_ids // len(rows))
        initial_frames = torch.minimum(motion.clip_start[clips] + offsets,
                                       motion.clip_end[clips] - cfg.hist - cfg.horizon - 1)
        if bool((initial_frames < motion.clip_start[clips]).any()):
            raise ValueError("Fixed clip cannot accommodate history and H32")

        def set_parameters(step, gradients=None):
            with torch.no_grad():
                for index, parameter in enumerate(learner.pi.parameters()):
                    parameter.copy_(base_parameters[index] if gradients is None else
                                    base_parameters[index] - step * gradients[index])

        def warm(fresh_environment=False):
            nonlocal control_steps, env
            set_parameters(0.)
            records = []
            if env is None:
                for index in range(cfg.hist):
                    frames = initial_frames + index
                    records.append(dict(s=flatten_state(motion.ref_state(frames)), f=frames,
                                        xi=context.expand(args.envs, -1).clone(),
                                        a_prev=torch.zeros(args.envs, 22, device=device)))
            else:
                if fresh_environment:
                    env.close()
                    env.backend.close()
                    env = None
                    _, env, _ = prepare(preparation_args, num_envs=args.envs, eval_mode=True)
                env.reset_to_frames(initial_frames)
                import warp as wp
                wp.to_torch(env.backend._device_data.qacc_warmstart).zero_()
                history = None
                with torch.no_grad():
                    for index in range(cfg.hist):
                        info = env.state.info[SUPERMIMIC_STATE_KEY]
                        records.append({k: info[k].clone() for k in ("s", "f", "xi", "a_prev")})
                        measured = mdp.obs(unflatten_state(info["s"]), info["a_prev"], info["xi"], info["f"])
                        history = measured[:, None].repeat(1, cfg.hist, 1) if history is None else torch.cat((history[:, 1:], measured[:, None]), 1)
                        if index < cfg.hist - 1:
                            action = learner.pi(history, mdp.goal(info["f"], cfg.goal_frames))
                            stepped = env.step(action)
                            control_steps += args.envs
                            if bool(stepped.info[SUPERMIMIC_TRANSITION_KEY]["done"].any()):
                                raise ValueError("Warm-up terminated a fixed H32 sample")
                records[-1]["warmstart"] = wp.to_torch(env.backend._device_data.qacc_warmstart).clone()
            start = {k: v.clone() for k, v in records[-1].items()}
            history = {k: torch.stack([row[k] for row in records], 1) for k in ("s", "f", "xi", "a_prev")}
            return start, history

        start, history = warm()
        if not bool(mdp.valid_imagination_start(start["f"], 32).all()):
            raise ValueError("Illegal H32 start")
        learner._sample_valid_starts = lambda size: (start, history)
        captured = {}

        def reward_record(*values):
            reward, done = learner._reward_term(*values)
            captured["rewards"] = reward.detach().view(32, args.envs)
            captured["dones"] = done.detach().view(32, args.envs)
            return reward, done

        class DirectionCaptured(Exception):
            pass

        class CaptureWithoutStep:
            def zero_grad(self, set_to_none=True):
                for parameter in learner.pi.parameters():
                    parameter.grad = None

            def step(self):
                captured["gradients"] = [p.grad.detach().clone() for p in learner.pi.parameters()]
                raise DirectionCaptured()

        learner._reward_term_fn = reward_record
        learner.opt_pi = CaptureWithoutStep()
        hook = learner.critic.register_forward_hook(lambda module, inputs, output:
                                                     captured.update(tail=output.detach().clone()))
        torch.manual_seed(args.seed)
        try:
            learner.policy_update()
        except DirectionCaptured:
            pass
        finally:
            hook.remove()
            learner.wm.requires_grad_(False)
            learner.critic.requires_grad_(False)
        gradients = captured["gradients"]
        if not all(bool(torch.isfinite(g).all()) for g in gradients):
            raise ValueError("Nonfinite native actor direction")
        gradient_norm = sum(float(g.square().sum()) for g in gradients)**.5
        if gradient_norm == 0:
            raise ValueError("Native H32 direction is zero")
        if protected != {name: digest(getattr(learner, name)) for name in protected}:
            raise AssertionError("Native gradient capture changed parameters or normalization")

        def objective(rewards, dones, tail):
            alive = torch.cumprod(1 - dones, 0)
            continuation = torch.cat((torch.ones_like(alive[:1]), alive[:-1]))
            discount = cfg.gamma**torch.arange(32, device=device).float()[:, None]
            reward_value = (discount * continuation * rewards).sum(0)
            bootstrap = cfg.gamma**32 * alive[-1] * tail
            return reward_value, bootstrap, alive[-1]

        @torch.no_grad()
        def score(step, physical=False):
            nonlocal control_steps
            if physical:
                current, matched_history = warm(fresh_environment=True)
                result.setdefault("branch_initialization_checks", []).append({
                    "step": step,
                    "maximum_absolute_difference": {
                        key: float((current[key] - start[key]).abs().max()) for key in start
                    }})
                for key in start:
                    torch.testing.assert_close(current[key], start[key], atol=2e-6, rtol=2e-5)
                for key in history:
                    torch.testing.assert_close(matched_history[key], history[key], atol=2e-6, rtol=2e-5)
            set_parameters(step, gradients)
            torch.manual_seed(args.seed)
            state = unflatten_state(start["s"])
            frames, previous, xi = start["f"].clone(), start["a_prev"].clone(), start["xi"]
            normalized = mdp.normalize_context(xi)
            obs_hist = mdp.add_noise(mdp.obs(unflatten_state(history["s"]), history["a_prev"], history["xi"], history["f"]))
            goal = mdp.goal(frames, cfg.goal_frames)
            rewards, dones, hard = [], [], torch.zeros(args.envs, dtype=torch.bool, device=device)
            first_action = learner.pi(obs_hist, goal)
            for _ in range(32):
                action = learner.pi(obs_hist, goal)
                if physical:
                    output = env.step(action)
                    control_steps += args.envs
                    transition = output.info[SUPERMIMIC_TRANSITION_KEY]
                    state = unflatten_state(transition["s1"])
                    frames = transition["f"] + 1
                    hard |= transition["done"]
                    obs = mdp.add_noise(mdp.obs(state, action, xi, frames))
                    obs_hist = torch.cat((obs_hist[:, 1:], obs[:, None]), 1)
                    goal = mdp.goal(frames, cfg.goal_frames)
                else:
                    state, action, obs_hist, frames, goal = learner._imag_step(
                        state, obs_hist, previous, frames, xi, normalized, goal)
                reward, done = learner._reward_term(state, frames, action, previous)
                rewards.append(reward)
                dones.append(torch.maximum(done, hard.float()) if physical else done)
                previous = action
            tail = learner.critic(learner._critic_input(state, frames, normalized, goal))
            value, bootstrap, survival = objective(torch.stack(rewards), torch.stack(dones), tail)
            return dict(step=step, objective=(value + bootstrap).cpu().tolist(),
                        discounted_reward=value.cpu().tolist(), critic_tail=bootstrap.cpu().tolist(),
                        survival=survival.cpu().tolist(), hard_completions=int(hard.sum())), first_action

        reward_value, bootstrap, survival = objective(captured["rewards"], captured["dones"], captured["tail"])
        native_objective = reward_value + bootstrap
        model_base, base_action = score(0.)
        torch.testing.assert_close(torch.tensor(model_base["objective"], device=device), native_objective, atol=2e-6, rtol=2e-5)
        model_rows = [model_base]
        physical_rows = []
        for step in (-args.direction_step, args.direction_step):
            row, action = score(step)
            row["first_action_rms_change"] = float((action - base_action).square().mean().sqrt())
            model_rows.append(row)
        if env is not None:
            for step in (0., -args.direction_step, args.direction_step):
                row, action = score(step, physical=True)
                row["first_action_rms_change"] = float((action - base_action).square().mean().sqrt())
                physical_rows.append(row)
        learner.pi.load_state_dict(initial_policy)
        if protected != {name: digest(getattr(learner, name)) for name in protected}:
            raise AssertionError("Branches changed source parameters or normalization")
        if sources != [identity(path) for path in source_paths]:
            raise ValueError("Checkpoint or native source changed during H32 diagnostic")
        result.update(status="H32_CPU_OBJECTIVE_CONTRACT_PASSED" if env is None else "H32_NATIVE_DIRECTION_MEASURED",
                      sources=sources, native_learner_method="SuperMimicLearner.policy_update",
                      batch=args.envs, source_pi_batch=saved["args"]["pi_batch"],
                      actual_network_precision="fp32" if device.type == "cpu" else cfg.amp,
                      source_network_precision=cfg.amp, compile=False,
                      start_frames=start["f"].cpu().tolist(), fixed_clips=[r["name"] for r in rows],
                      history="Reference poses only" if env is None else "Fresh independent env per branch;9 identical native warm-up steps; exact state/history/warmstart checks",
                      gradient="Native clipped actor-loss gradient; parameter ascent uses its negative; not an Adam/momentum update",
                      clipped_gradient_norm=gradient_norm, parameter_direction_step=args.direction_step,
                      native_objective_reproduced=True, parameters_and_normalizers_restored=True,
                      common_observation_noise_seed=args.seed,
                      imagined=model_rows, physical=physical_rows,
                      new_control_transitions=control_steps, new_integration_substeps=4*control_steps,
                      limits=["Diagnostic small batch, not the production1024-start training gradient.",
                              "No optimizer step: this checks native gradient direction, not saved Adam momentum.",
                              "Critic tail is reported separately; increased value is not physical gait success.",
                              "CPU contract uses reference poses/fp32 and does not validate native Warp physics or BF16.",
                              "H32 is0.64s, not a full walking/stability evaluation.",
                              "Physical hard termination truncates the remaining objective and is reported separately."])
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result.update(new_control_transitions=control_steps, new_integration_substeps=4*control_steps)
        if env is not None:
            env.close()
            env.backend.close()
        if cpu_rng is not None:
            import torch
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            result["torch_rng_restored"] = bool(torch.equal(cpu_rng, torch.get_rng_state()))
        (args.output / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "library", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("cpu-contract", "native-direction"), required=True)
    parser.add_argument("--envs", type=int, default=16, choices=(16,))
    parser.add_argument("--seed", type=int, default=32100)
    parser.add_argument("--direction-step", type=float, default=.01)
    parsed = parser.parse_args()
    if not 0 < parsed.direction_step <= .1:
        parser.error("direction-step must be positive and at most.1")
    run(parsed)
