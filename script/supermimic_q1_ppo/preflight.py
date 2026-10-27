"""Zero-optimizer physical contracts for the RF12 RSL-RL adapter."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_hp"))
from worker import prepare, check_contract
from adapter import HPActor, PhysicalAdapter, make_ppo, bootstrap_rewards, reward_components, termination_masks
from resources import save, identity, stamp


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "RUNNING", "started_utc": stamp(), "num_envs": args.envs,
              "optimizer_updates": 0, "checkpoint": identity(args.checkpoint)}
    save(args.output / "status.json", result)
    env = adapter = None
    try:
        torch.set_num_threads(8)
        _, env, _ = prepare(args, num_envs=args.envs)
        result["HP_contract"] = check_contract(env)
        from unisupermimic.tasks.motion_tracking.x2.supermimic.evaluate import load_policy
        from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY
        from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
        checkpoint = torch.load(args.checkpoint, map_location=env.device, weights_only=True)
        policy = load_policy(env, checkpoint)
        actor = HPActor(policy).to(env.device)
        adapter = PhysicalAdapter(env, actor)
        separate = load_policy(env, checkpoint)
        with torch.no_grad():
            expected = HPActor(separate).to(env.device)(adapter.observations)
            measured = actor(adapter.observations)
            torch.testing.assert_close(expected, measured, rtol=0, atol=0)
        result["policy_transfer_max_abs"] = float((expected - measured).abs().max())
        del separate
        before = {k: v.clone() for k, v in actor.state_dict().items()}
        with torch.no_grad():
            reference = adapter.mdp.motion.ref_state(adapter.current["f"])
            joint, wrist, bonus = reward_components(adapter.mdp, reference, adapter.current["f"], 1.)
            torch.testing.assert_close(joint, torch.ones_like(joint), atol=2e-6, rtol=0)
            torch.testing.assert_close(wrist, torch.ones_like(wrist), atol=2e-6, rtol=0)
            torch.testing.assert_close(bonus, torch.full_like(bonus, .03), atol=2e-6, rtol=0)
        algorithm = make_ppo(actor, adapter.observations, args.envs)
        old_cap = adapter.command.max_episode_steps
        adapter.command.max_episode_steps = adapter.max_episode_length = 1
        with torch.no_grad():
            next_obs, rewards, dones, extras = adapter.step(torch.zeros(args.envs, 22, device=env.device))
            if not bool(dones.all()) or not bool(extras["artificial_timeouts"].any()):
                raise AssertionError("Artificial timeout contract was not exercised")
            terminal = extras["terminal_observations"]
            adjusted = bootstrap_rewards(rewards, extras, algorithm.critic)
            mask = extras["artificial_timeouts"]
            torch.testing.assert_close(adjusted[mask], rewards[mask] + .99 * algorithm.critic(terminal).squeeze(-1))
            if "time_outs" in extras:
                raise AssertionError("Stock current-state timeout bootstrap must stay disabled")
            torch.testing.assert_close(adapter.history, adapter.history[:, :1].expand_as(adapter.history))
            result["artificial_timeouts_checked"] = int(mask.sum())
            # A natural clip end remains finite-task terminal even at a time cap.
            motion = adapter.mdp.motion
            end_frames = motion.clip_end[motion.clip_id[adapter.current["f"]]] - 1
            _, info = env.reset_to_frames(end_frames)
            adapter.current = info[SUPERMIMIC_STATE_KEY]
            adapter.history = None
            adapter.observations = adapter.observe(adapter.current)
            _, _, done, final_extras = adapter.step(torch.zeros(args.envs, 22, device=env.device))
            natural, timeout = termination_masks(adapter.last_transition, motion, 1)
            if not bool(natural.all() & done.all()) or bool(timeout.any()):
                raise AssertionError("Natural clip-end bootstrap classification failed")
        adapter.command.max_episode_steps = adapter.max_episode_length = old_cap
        env.reset()
        adapter.current = env.state.info[SUPERMIMIC_STATE_KEY]
        adapter.history = None
        adapter.observations = adapter.observe(adapter.current)
        with torch.no_grad():
            for _ in range(24):
                action = algorithm.act(adapter.observations)
                obs, reward, done, extras = adapter.step(action)
                tr = adapter.last_transition
                original = adapter.mdp.reward(unflatten_state(tr["s1"]), tr["f"] + 1, tr["a"], tr["a_prev"])
                torch.testing.assert_close(reward, original, atol=2e-6, rtol=2e-5)
                if bool((motion.clip_id[tr["f"]] >= 300).any()):
                    raise AssertionError("AP03 sampled during Stage A")
                algorithm.process_env_step(obs, bootstrap_rewards(reward, extras, algorithm.critic), done, {})
            algorithm.compute_returns(adapter.observations)
        if not bool(torch.isfinite(algorithm.storage.returns).all()):
            raise AssertionError("Nonfinite physical GAE returns")
        batch = next(algorithm.storage.mini_batch_generator(4, 1))
        actor(batch.observations, stochastic_output=True)
        log_prob = actor.get_output_log_prob(batch.actions)
        result["unchanged_policy_logprob_max_abs"] = float((log_prob - batch.old_actions_log_prob.squeeze(-1)).detach().abs().max())
        torch.testing.assert_close(log_prob, batch.old_actions_log_prob.squeeze(-1), atol=2e-4, rtol=2e-5)
        loss = log_prob.mean() + algorithm.critic(batch.observations).square().mean()
        loss.backward()
        if not all(bool(torch.isfinite(p.grad).all()) for p in list(actor.parameters()) + list(algorithm.critic.parameters()) if p.grad is not None):
            raise AssertionError("Nonfinite full-minibatch backward")
        for key, value in actor.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        result.update(status="ZERO_OPTIMIZER_PHYSICAL_PREFLIGHT_PASSED", completed_utc=stamp(),
                      transition_count=24 * args.envs, rollout_steps=24, minibatch_size=len(batch.actions),
                      action_std=.1, actual_reward_frame="f+1 pre-reset state", natural_terminal_no_bootstrap=True,
                      artificial_terminal_bootstrap="V(actual pre-reset s1), not V(reset state) or V(s_t)",
                      normalization="RF11 copied and fixed", raw_action_logprob_and_executed_clip_consistent=True,
                      actor_parameters_unchanged=True, full_minibatch_forward_backward_finite=True,
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save(args.output / "receipt.json", result)
        print(json.dumps(result), flush=True)
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(args.output / "status.json", result)
        if adapter is not None:
            adapter.close()
        elif env is not None:
            env.close()
            env.backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "library", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--envs", type=int, choices=(16, 4096), required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
