"""Zero-update RSL-RL contracts for foot rewards and physical gait evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_gait_comp"), str(HERE.parent / "supermimic_q1_hp"), str(HERE.parent / "project_tools")]
from worker import prepare, check_contract, process_identity
from contact_adapter import Adapter, Actor, make_ppo, bootstrap_rewards, termination_masks
from contact_reference import configure, ContactReference
from physical import preserve_balance, rollout, FootReference, FIXED_DEMOS, preserve_random_state
from diagnose_q1_upper_reference import policy_digest
from resources import save, identity, stamp


def run(a):
    configure(a.library)
    a.output.mkdir(parents=True, exist_ok=False)
    status = {"status": "INITIALIZING", "process": process_identity(), "started_utc": stamp(),
              "num_envs": a.envs, "optimizer_steps": 0}
    save(a.output / "status.json", status)
    adapter = env = independent = None
    try:
        torch.set_num_threads(8)
        _, env, _ = prepare(a, num_envs=a.envs)
        status["contract"] = check_contract(env)
        from unisupermimic.tasks.motion_tracking.x2.supermimic.evaluate import load_policy
        from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY
        teacher = load_policy(env, torch.load(a.checkpoint, map_location=env.device, weights_only=True)).requires_grad_(False).eval()
        digest = policy_digest(teacher)
        thresholds = json.loads(a.calibration.read_text())["thresholds"]
        adapter = Adapter(env, teacher, thresholds, a.upper_alpha)
        status["phase_contract"] = adapter.foot_reference.contract
        actor = Actor().to(env.device)
        before = {k:v.clone() for k,v in actor.state_dict().items()}
        algorithm = make_ppo(actor, adapter.observations, a.envs)
        actual = {id(p) for group in algorithm.optimizer.param_groups for p in group["params"]}
        expected = {id(p) for p in list(actor.parameters())+list(algorithm.critic.parameters())}
        if actual != expected:
            raise AssertionError("Optimizer must contain only new head and critic")
        old_limit = adapter.command.max_episode_steps
        adapter.command.max_episode_steps = adapter.max_episode_length = 1
        with torch.no_grad():
            zero = actor(adapter.observations)
            torch.testing.assert_close(zero, torch.zeros_like(zero), atol=0, rtol=0)
            _, reward, done, extras = adapter.step(zero)
            mask = extras["artificial_timeouts"]
            if not bool(done.all()) or not bool(mask.any()):
                raise AssertionError("Artificial truncation was not exercised")
            adjusted = bootstrap_rewards(reward, extras, algorithm.critic)
            torch.testing.assert_close(adjusted[mask], reward[mask]+.99*algorithm.critic(extras["terminal_observations"]).squeeze(-1))
            frames = adapter.mdp.motion.clip_end[adapter.mdp.motion.clip_id[adapter.current["f"]]]-1
            _, info = env.reset_to_frames(frames)
            adapter.current = info[SUPERMIMIC_STATE_KEY]
            adapter.history = None
            adapter.reset_gate(torch.ones(a.envs, dtype=torch.bool, device=env.device))
            adapter.observations = adapter.observe(adapter.current)
            _, _, done, extras = adapter.step(zero)
            natural, timeouts = termination_masks(adapter.last_transition, adapter.mdp.motion, 1)
            if not bool(natural.all() & done.all()) or bool(timeouts.any()):
                raise AssertionError("Natural clip completion must not bootstrap")
        adapter.command.max_episode_steps = adapter.max_episode_length = old_limit
        env.reset()
        adapter.current = env.state.info[SUPERMIMIC_STATE_KEY]
        adapter.history = None
        adapter.reset_gate(torch.ones(a.envs, dtype=torch.bool, device=env.device))
        adapter.observations = adapter.observe(adapter.current)
        with torch.no_grad():
            for _ in range(24):
                latent = algorithm.act(adapter.observations)
                obs, reward, done, extras = adapter.step(latent)
                parts = adapter.last_reward_parts
                expected_reward = parts["original"]+parts["stance_bonus"]+parts["swing_bonus"]+parts["stationary_bonus"]-parts["smooth_cost"]-parts["residual_cost"]
                torch.testing.assert_close(reward, expected_reward, atol=2e-8, rtol=2e-6)
                if not bool(torch.isfinite(obs["features"]).all()) or obs["features"].shape != (a.envs, 1420):
                    raise AssertionError("Invalid foot-conditioned observations")
                algorithm.process_env_step(obs, bootstrap_rewards(reward, extras, algorithm.critic), done, {})
            algorithm.compute_returns(adapter.observations)
        with preserve_balance(adapter):
            with preserve_random_state():
                rows = [r for r in json.loads(a.library.read_text())["splits"]["train"] if r["name"] in FIXED_DEMOS]
                if len(rows) != 3:
                    raise ValueError("Fixed demonstrations missing")
                lib = a.output / "evaluation_library.json"
                save(lib, {"splits": {"train": rows}})
                _, independent, _ = prepare(SimpleNamespace(runtime=a.runtime, library=lib, output=a.output),
                                             num_envs=3, eval_mode=True)
                foot = ContactReference(independent.supermimic_mdp)
                trace = rollout(independent, teacher, actor.head, thresholds, foot, upper_alpha=a.upper_alpha)
                for key in ("states", "feet", "reference_feet"):
                    if not bool(torch.isfinite(torch.from_numpy(trace[key])).all()):
                        raise AssertionError("Nonfinite physical gait evidence")
        batch = next(algorithm.storage.mini_batch_generator(4, 1))
        actor(batch.observations, stochastic_output=True)
        probability = actor.get_output_log_prob(batch.actions)
        old_probability = batch.old_actions_log_prob.squeeze(-1)
        torch.testing.assert_close(probability, old_probability, atol=2e-4, rtol=2e-5)
        (probability.mean()+algorithm.critic(batch.observations).square().mean()).backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in list(actor.parameters())+list(algorithm.critic.parameters())):
            raise AssertionError("Nonfinite PPO gradients")
        if any(p.grad is not None for p in teacher.parameters()) or policy_digest(teacher) != digest:
            raise AssertionError("Frozen base changed")
        for key,value in actor.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        status.update(status="ZERO_OPTIMIZER_PHYSICAL_PREFLIGHT_PASSED", completed_utc=stamp(),
                      source_base=identity(a.checkpoint), input_features=1420, latent_actions=14,
                      maximum_logprob_difference=float((probability-old_probability).abs().max()),
                      full_minibatch=len(batch.actions), physical_transitions=27*a.envs,
                      extra_independent_evaluation_steps=int(trace["states"].shape[0]*3),
                      reward_decomposition_verified=True, artificial_terminal_bootstrap_verified=True,
                      natural_terminal_no_bootstrap=True, native_contacts_and_pre_reset_states=True,
                      independent_eval_preserved_live_state_rng=True, only_new_head_and_critic_in_optimizer=True,
                      teacher_and_normalizers_unchanged=True, actor_parameters_unchanged=True)
        status.update(upper_reference_alpha=a.upper_alpha, desired_kl=algorithm.desired_kl,
                      adaptive_learning_rate_bounds=[1e-5, 5e-5])
    except BaseException as error:
        status.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if adapter is not None:
            adapter.close()
        elif env is not None:
            env.close()
            env.backend.close()
        if independent is not None:
            independent.close()
            independent.backend.close()
        save(a.output / "status.json", status)
        save(a.output / "receipt.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "library", "checkpoint", "calibration", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--envs", type=int, choices=(16,4096), required=True)
    p.add_argument("--upper-alpha", type=float, default=.5)
    run(p.parse_args())

