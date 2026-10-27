"""CPU tensor contracts for a candidate; no simulator steps or optimization."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys

from build_tracking_candidate import TASK, identity


def check_runtime(runtime, files):
    import torch

    torch.set_num_threads(1)
    sys.path[:0] = [str(runtime/"src"), str(runtime)]
    from unisupermimic.rl.contracts.supermimic import validate_policy_input_spec
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.mdp import MDP, policy_input_spec
    from unisupermimic.tasks.motion_tracking.q1.supermimic.mdp_base import MDP as BaseMDP
    from unisupermimic.tasks.motion_tracking.q1.supermimic.mdp_base import policy_input_spec as base_spec
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import BIN_FRAMES, CONTROL_DT, MotionLib
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion_base import MotionLib as BaseMotionLib

    options = json.loads((runtime/TASK/"candidate_options.json").read_text())
    adapter = json.loads((runtime/"q1_adapter.json").read_text())
    default = torch.tensor([adapter["control"][name]["default"] for name in adapter["joints"]])
    kin = Q1Kinematics(runtime/"q1_assets/scene.xml", "cpu")
    motion = MotionLib([str(path) for path in files], kin, "cpu")
    original_motion = BaseMotionLib([str(path) for path in files], kin, "cpu")
    mdp = MDP(kin, motion, default, observation_mode="root_xy")
    original = BaseMDP(kin, original_motion, default, observation_mode="root_xy")
    joint_error = options["joint_error_observation"]
    checks = []

    for name in ("root_p", "root_R", "root_v", "root_w", "q", "qd", "body_p", "body_R", "body_v", "body_w", "clip_start", "clip_end", "clip_id"):
        torch.testing.assert_close(getattr(motion, name), getattr(original_motion, name), atol=0, rtol=0)
    ends = motion.clip_end[motion.bin_clip]
    assert bool((motion.bin_start < ends).all())
    assert motion.num_bins == int(((motion.clip_len-2)//BIN_FRAMES+1).sum())
    torch.testing.assert_close(motion.frame_bin[motion.clip_end], motion.frame_bin[motion.clip_end-1])
    checks.append("reference_tensors_preserved_and_bins_exclude_terminal_starts")

    frames = torch.cat((motion.clip_start, torch.minimum(motion.clip_start+1, motion.clip_end), motion.clip_end-1))
    state = list(motion.ref_state(frames))
    state[4] = state[4] + torch.linspace(-.15, .15, 22)
    state = tuple(state)
    previous = torch.zeros(len(frames), 22)
    context = torch.zeros(len(frames), 26)
    context[:, 0], context[:, 1:23] = 1., .003
    obs = mdp.obs(state, previous, context, frames)
    torch.testing.assert_close(obs[:, :80], original.obs(state, previous, context, frames))
    assert mdp.policy_input_spec == policy_input_spec("root_xy")
    assert obs.shape == (len(frames), 102 if joint_error else 80)
    if joint_error:
        torch.testing.assert_close(obs[:, 80:], motion.q[frames]-state[4])
    torch.testing.assert_close(mdp.goal(frames, 10), original.goal(frames, 10), atol=0, rtol=0)
    torch.testing.assert_close(mdp.goal(motion.clip_end, 10), original.goal(motion.clip_end, 10), atol=0, rtol=0)
    checks.append("current_joint_error_and_unchanged_base_observation_goal")

    for values in (obs, obs[:, None].expand(-1, 10, -1)):
        torch.manual_seed(20261027)
        noisy = mdp.add_noise(values)
        delta = noisy-values
        if joint_error:
            torch.testing.assert_close(delta[..., :22], -delta[..., 80:], atol=5e-7, rtol=0)
        assert bool((delta.abs() <= mdp.obs_noise+5e-7).all())
    differentiable_q = state[4].detach().clone().requires_grad_(True)
    grad_state = (*state[:4], differentiable_q, state[5])
    if joint_error:
        gradient = torch.autograd.grad(mdp.obs(grad_state, previous, context, frames)[..., 80:].sum(), differentiable_q)[0]
        torch.testing.assert_close(gradient, -torch.ones_like(gradient))
    checks.append("batch_history_noise_consistency_and_observation_derivative")

    action = torch.full_like(previous, .05)
    value, terms = mdp.reward(grad_state, frames, action, previous, terms=True)
    torch.testing.assert_close(value, mdp.reward(grad_state, frames, action, previous))
    reconstructed = terms["track"] + CONTROL_DT*(mdp.cfg.w_action_rate*terms["action_rate"] + mdp.cfg.w_joint_limit*terms["joint_limit"])
    torch.testing.assert_close(value, reconstructed, atol=2e-7, rtol=1e-5)
    if options["joint_tracking_reward"]:
        error = differentiable_q-motion.q[frames]
        upper = .5*torch.exp(-error[:, 14:].square().mean(-1)/mdp.cfg.upper_joint_std**2)*CONTROL_DT
        lower = .5*torch.exp(-error[:, :14].square().mean(-1)/mdp.cfg.lower_joint_std**2)*CONTROL_DT
        torch.testing.assert_close(terms["joint_upper"], upper)
        torch.testing.assert_close(terms["joint_lower"], lower)
        gradient = torch.autograd.grad((terms["joint_upper"]+terms["joint_lower"]).sum(), differentiable_q)[0]
        assert bool(torch.isfinite(gradient).all()) and bool((gradient*error <= 1e-9).all())
        assert float(gradient.abs().sum()) > 0
        reference_terms = mdp.reward(motion.ref_state(frames), frames, previous, previous, terms=True)[1]
        torch.testing.assert_close(reference_terms["track"], torch.full_like(value, .1), atol=5e-6, rtol=0)
    else:
        torch.testing.assert_close(value, original.reward(grad_state, frames, action, previous), atol=0, rtol=0)
    checks.append("reward_decomposition_pose_gradient_and_tracking_maximum")

    validate_policy_input_spec({"policy_input_spec": mdp.policy_input_spec}, mdp)
    old_schema_rejected = False
    try:
        validate_policy_input_spec({"policy_input_spec": base_spec("root_xy")}, mdp)
    except ValueError:
        old_schema_rejected = True
    assert old_schema_rejected == joint_error
    checks.append("checkpoint_input_schema_contract")
    assert not torch.cuda.is_initialized()
    return {"status": "CPU_TENSOR_CONTRACTS_PASSED_NOT_PHYSICALLY_VALIDATED", "checks": checks,
            "runtime": str(runtime), "candidate_source": identity(runtime/"candidate_source.json"),
            "reference_files": [{"path": str(path), **identity(path)} for path in files],
            "options": options, "obs_dim": mdp.obs_dim, "goal_dim": int(mdp.goal(frames, 10).shape[-1]),
            "tested_states": len(frames), "old_input_schema_rejected": old_schema_rejected,
            "physics_steps": 0, "optimizer_updates": 0, "cuda_initialized": False,
            "limits": "Tensor and analytic-head checks only. Native command/reset, replay/imagine rollout, physical action directions and training remain unvalidated."}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--motion", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.motion) > 4:
        parser.error("Use at most four fixed clips for this bounded CPU check")
    if args.output.exists():
        raise FileExistsError(args.output)
    result = {"status": "FAILED", "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        result.update(check_runtime(args.runtime.resolve(), [path.resolve() for path in args.motion]))
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
