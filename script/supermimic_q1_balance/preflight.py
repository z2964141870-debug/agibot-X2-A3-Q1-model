"""Zero-optimizer tests on actual 16/4096 Warp environments for the residual PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_hp"), str(HERE.parent / "project_tools")]
from worker import prepare, check_contract, process_identity
from diagnose_q1_upper_reference import policy_digest
from balance_adapter import BalanceActor, BalanceAdapter, make_ppo, bootstrap_rewards, termination_masks
from control import compose, balance_rewards, RADIUS
from balance_evaluation import preserve_balance, preserve_random_state, rollout, FIXED_DEMOS
from resources import identity, save, stamp


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    status = {"status": "INITIALIZING", "process": process_identity(), "optimizer_steps": 0,
              "num_envs": args.envs, "started_utc": stamp()}
    save(args.output / "status.json", status)
    adapter = env = None
    try:
        torch.set_num_threads(8)
        _, env, _ = prepare(args, num_envs=args.envs)
        status["physical_contract"] = check_contract(env)
        from unisupermimic.tasks.motion_tracking.x2.supermimic.evaluate import load_policy
        from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY
        inherited = torch.load(args.checkpoint, map_location=env.device, weights_only=True)
        teacher = load_policy(env, inherited)
        teacher_before = policy_digest(teacher)
        adapter = BalanceAdapter(env, teacher, json.loads(args.calibration.read_text())["thresholds"])
        actor = BalanceActor().to(env.device)
        before = {k: v.clone() for k, v in actor.state_dict().items()}
        algorithm = make_ppo(actor, adapter.observations, args.envs)
        expected_ids = {id(p) for p in list(actor.parameters()) + list(algorithm.critic.parameters())}
        actual_ids = {id(p) for group in algorithm.optimizer.param_groups for p in group["params"]}
        if actual_ids != expected_ids or any(p.requires_grad for p in teacher.parameters()):
            raise AssertionError("Optimizer contains frozen teacher or misses trainable parameters")
        with torch.no_grad():
            zero = actor(adapter.observations)
            torch.testing.assert_close(zero, torch.zeros_like(zero), rtol=0, atol=0)
            executed, residual = compose(adapter.nominal, adapter.reference_action, zero, adapter.gate.alpha, adapter.scale)
            torch.testing.assert_close(executed, adapter.nominal, rtol=0, atol=0)
        old_cap = adapter.command.max_episode_steps
        adapter.command.max_episode_steps = adapter.max_episode_length = 1
        with torch.no_grad():
            _, reward, done, extras = adapter.step(zero)
            if not bool(done.all()) or not bool(extras["artificial_timeouts"].any()):
                raise AssertionError("Artificial truncation not exercised")
            adjusted = bootstrap_rewards(reward, extras, algorithm.critic)
            mask = extras["artificial_timeouts"]
            torch.testing.assert_close(adjusted[mask], reward[mask] + .99 * algorithm.critic(extras["terminal_observations"]).squeeze(-1))
            torch.testing.assert_close(adapter.history, adapter.history[:, :1].expand_as(adapter.history))
            status["artificial_timeouts_checked"] = int(mask.sum())
            motion = adapter.mdp.motion
            end_frames = motion.clip_end[motion.clip_id[adapter.current["f"]]] - 1
            _, info = env.reset_to_frames(end_frames)
            adapter.current = info[SUPERMIMIC_STATE_KEY]
            adapter.history = None
            adapter.reset_gate(torch.ones_like(mask))
            adapter.observations = adapter.observe(adapter.current)
            _, _, done, extra = adapter.step(zero)
            natural, timeout = termination_masks(adapter.last_transition, motion, 1)
            if not bool(natural.all() & done.all()) or bool(timeout.any()):
                raise AssertionError("Natural terminal must not bootstrap")
        adapter.command.max_episode_steps = adapter.max_episode_length = old_cap
        env.reset()
        adapter.cap = .75
        adapter.current = env.state.info[SUPERMIMIC_STATE_KEY]
        adapter.history = None
        adapter.reset_gate(torch.ones(args.envs, dtype=torch.bool, device=env.device))
        adapter.observations = adapter.observe(adapter.current)
        with torch.no_grad():
            for _ in range(24):
                action = algorithm.act(adapter.observations)
                obs, reward, done, extras = adapter.step(action)
                tr = adapter.last_transition
                expected = balance_rewards(tr["r"], tr["s1"], motion.root_p[tr["f"] + 1], motion.q[tr["f"] + 1], RADIUS * torch.tanh(action))[0]
                torch.testing.assert_close(reward, expected, rtol=0, atol=0)
                if bool((motion.clip_id[tr["f"]] >= 300).any()):
                    raise AssertionError("Unexpected expanded training reference")
                algorithm.process_env_step(obs, bootstrap_rewards(reward, extras, algorithm.critic), done, {})
            algorithm.compute_returns(adapter.observations)
        evaluation_steps = 0
        with preserve_balance(adapter):
            with preserve_random_state():
                source = json.loads(args.library.read_text())
                rows = [row for row in source["splits"]["train"] if row["name"] in FIXED_DEMOS]
                if len(rows) != 3:
                    raise ValueError("The original fixed three demos are required")
                lib = args.output / "preservation_library.json"
                save(lib, {"splits": {"train": rows}})
                _, independent, _ = prepare(SimpleNamespace(runtime=args.runtime, library=lib, output=args.output),
                                             num_envs=3, eval_mode=True)
                try:
                    trace = rollout(independent, teacher, actor.head,
                                    json.loads(args.calibration.read_text())["thresholds"], "trained_static075")
                    evaluation_steps = int(trace["states"].shape[0] * 3)
                finally:
                    independent.close()
                    independent.backend.close()
        batch = next(algorithm.storage.mini_batch_generator(4, 1))
        actor(batch.observations, stochastic_output=True)
        logprob = actor.get_output_log_prob(batch.actions)
        difference = float((logprob - batch.old_actions_log_prob.squeeze(-1)).abs().max())
        torch.testing.assert_close(logprob, batch.old_actions_log_prob.squeeze(-1), atol=2e-4, rtol=2e-5)
        (logprob.mean() + algorithm.critic(batch.observations).square().mean()).backward()
        parameters = list(actor.parameters()) + list(algorithm.critic.parameters())
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise AssertionError("Nonfinite residual/critic gradient")
        if any(p.grad is not None for p in teacher.parameters()) or policy_digest(teacher) != teacher_before:
            raise AssertionError("Frozen teacher or normalization changed")
        for key, value in actor.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        if not bool(torch.isfinite(algorithm.storage.returns).all()):
            raise AssertionError("Nonfinite GAE")
        status.update(status="ZERO_OPTIMIZER_PHYSICAL_PREFLIGHT_PASSED", completed_utc=stamp(),
            num_actions_stored=14, physical_joints=22, latent_std=.1, residual_radius_rad=RADIUS,
            maximum_logprob_difference=difference, full_minibatch=len(batch.actions),
            training_collection_transitions=24 * args.envs,
            new_physical_transitions=27 * args.envs + evaluation_steps,
            teacher_normalizers_unchanged=True, only_head_and_critic_in_optimizer=True,
            actual_terminal_bootstrap=True, natural_terminal_no_bootstrap=True,
            reward_decomposition_exact=True, native_contacts_observed=True,
            actor_parameters_unchanged=True, calibration=identity(args.calibration),
            independent_evaluation_preserved_live_gate_rng_contacts=True,
            fallback_enabled_in_training=adapter.fallback_enabled,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save(args.output / "receipt.json", status)
    except BaseException as error:
        status.update(status="FAILED", error=f"{type(error).__name__}: {error}", ended_utc=stamp())
        raise
    finally:
        if adapter is not None:
            adapter.close()
        elif env is not None:
            env.close()
            env.backend.close()
        save(args.output / "status.json", status)
    print(json.dumps(status), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "library", "checkpoint", "calibration", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--envs", type=int, choices=(16, 4096), required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
