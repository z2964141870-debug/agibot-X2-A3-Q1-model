"""Exercise the actual native action, actuator, delayed memory and reward on CUDA."""

import argparse
import json
from pathlib import Path
import sys
from types import MethodType


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot", choices=("q1", "g1"), default="q1")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the original preflight result")
    sys.path.insert(0, str(args.source.resolve(strict=True) / "src"))
    import torch
    from bootstrap import install
    adapter = install(args.robot)
    from assets import presence
    from morphmimic.env.env_cfg import morphmimic_env_cfg
    from morphmimic.env.morphmimic_env import MorphMimicEnv
    from morphmimic.model.world_model import WorldModel, WorldModelCfg
    from morphmimic.model.state import StateNormalizer

    cfg = morphmimic_env_cfg(args.robot, subset_yaml=str(args.subset), num_envs=16)
    cfg.seed = 0
    cfg.auto_reset = False
    torque_audit = dict(calls=0, max_abs_error_nm=0.)
    if args.robot == "q1":
        actuator_cfg = cfg.scene.entities["robot"].articulation.actuators[0]
        original_build = type(actuator_cfg).build

        def audited_build(self, *build_args, **kwargs):
            actuator = original_build(self, *build_args, **kwargs)
            original_compute = actuator.compute

            def compute(cmd):
                actual = original_compute(cmd)
                request = actuator.stiffness * (cmd.position_target - cmd.pos) - actuator.damping * cmd.vel
                pos = torch.minimum(actuator.peak, torch.clamp(
                    actuator.peak * (actuator.cutoff - cmd.vel) /
                    (actuator.cutoff - actuator.knee), min=0))
                neg = -torch.minimum(actuator.peak, torch.clamp(
                    actuator.peak * (actuator.cutoff + cmd.vel) /
                    (actuator.cutoff - actuator.knee), min=0))
                expected = request.maximum(neg).minimum(pos).clamp(
                    min=-actuator.force_limit, max=actuator.force_limit)
                error = float((actual - expected).abs().max())
                torque_audit["calls"] += 1
                torque_audit["max_abs_error_nm"] = max(torque_audit["max_abs_error_nm"], error)
                if error > 1e-4 or not torch.isfinite(actual).all():
                    raise AssertionError("Native softwarePD/torque-speed callback mismatch")
                return actual

            actuator.compute = compute
            return actuator

        actuator_cfg.build = MethodType(audited_build, actuator_cfg)
    env = MorphMimicEnv(cfg, device="cuda:0")
    samples, reward_error, observation_error = [], 0., 0.
    try:
        spec = adapter["robot"]
        term = env.action_manager.get_term("joint_pos")
        robot = env.scene["robot"]
        assert env.physics_dt == (.002 if args.robot == "q1" else .005)
        assert env.step_dt == .02 and env.num_envs == 16
        assert env.sim.mj_model.nu == len(spec.joint_names)
        normalizer = StateNormalizer()
        wm = WorldModel(WorldModelCfg(hidden=(32, 32), morph_dim=154), normalizer,
            spec.translator.limits_to_canonical(robot.data.soft_joint_pos_limits[0]),
            spec.translator.to_canonical(robot.data.default_joint_pos[0]),
            imu_idx=spec.imu_slot).to("cuda:0")
        wm.set_morphology(adapter["morphology_population"]()[0][("g1", "q1").index(args.robot)])
        command = env.command_manager.get_term("motion")
        obs, _ = env.reset()
        mask = presence(spec, obs["state"])
        assert obs["state"].shape == (16, 269) and obs["context"].shape == (16, 62)
        assert torch.count_nonzero(obs["state"][..., 240:]) == 0
        target_error = 0.
        original_apply = term.apply_actions

        def apply_actions():
            nonlocal target_error
            index = term._substep
            original_apply()
            expected = torch.where(index >= term.delay_steps,
                term._processed_actions, term._previous_target) - robot.data.encoder_bias[:, term._target_ids]
            actual = robot.data.joint_pos_target[:, term._target_ids]
            target_error = max(target_error, float((actual - expected).abs().max()))
            if target_error > 1e-6:
                raise AssertionError("Native applied target does not follow the delay oracle")

        term.apply_actions = apply_actions
        for i in range(24):
            previous = env.action_manager.action.clone()
            frame = command.global_frame.clone()
            ref = command.library.state(frame)
            action = torch.full((16, 29), .1 if i % 2 else -.1, device="cuda:0")
            action[:, ~mask] = float("nan")
            lag_before = term.delay_steps.clone()
            with torch.no_grad():
                after, reward, terminated, timeout, _ = env.step(action)
                analytic_reward = wm.reward(after["state"], ref, term.project(action), previous)
                reward_error = max(reward_error, float((reward - analytic_reward).abs().max()))
                analytic_obs = wm.obs(after["state"], after["context"][:, 1:30], term.project(action))
                observation_error = max(observation_error,
                    float((after["policy_clean"] - analytic_obs).abs().max()))
            assert torch.equal(term.delay_steps, lag_before)
            assert torch.equal(after["state"][:, 240:], term.executed)
            assert not after["state"][:, :29][:, ~mask].any()
            assert not after["policy"][:, :29][:, ~mask].any()
            assert all(torch.isfinite(x).all() for x in after.values())
            samples.append(dict(step=i, terminated=int(terminated.sum()), timeout=int(timeout.sum()),
                delay_min=int(term.delay_steps.min()), delay_max=int(term.delay_steps.max())))
            ids = (terminated | timeout).nonzero(as_tuple=False).flatten()
            if len(ids):
                obs, _ = env.reset(env_ids=ids)
                assert torch.count_nonzero(obs["state"][ids, 240:]) == 0
        if reward_error > 3e-5 or observation_error > 3e-5:
            raise AssertionError(f"Native reward/observation mismatch {reward_error}, {observation_error}")
        if args.robot == "q1" and torque_audit["calls"] < 24 * 10:
            raise AssertionError("Q1 did not execute the softwarePD callback each substep")
        torch.cuda.synchronize()
        result = dict(status="PASSED", robot=args.robot, environments=16,
            control_steps=24, batched_physics_steps=24 * cfg.decimation,
            environment_transitions=24 * 16, new_optimizer_updates=0,
            reward_max_abs_error=reward_error, observation_max_abs_error=observation_error,
            target_max_abs_error=target_error, torque_callback=torque_audit,
            contracts=adapter["contracts"], physics_dt=env.physics_dt,
            delay_groups=term.delay_groups, samples=samples,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            quality_claim="interface and dynamics execution only; not a gait evaluation",
            user_accepted_checkpoint=None)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(status="PASSED", robot=args.robot,
            reward_error=reward_error, observation_error=observation_error, torque=torque_audit)), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
