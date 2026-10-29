"""Validate the isolated base MDP on real Q1 saved poses without integration."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def run(args):
    sys.path[:0] = [str(HERE), str(args.runtime / "src"), str(args.runtime),
                   str(HERE.parent / "project_tools")]
    import torch
    from check_contract import identity
    from base_mdp import SoleMDP
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
    from unisupermimic.tasks.motion_tracking.q1.supermimic.mdp import MDP
    from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import default_joint_pos_list
    from prepare_q1_gait_review import FIXED

    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(status="INITIALIZING", captured_utc=datetime.now(timezone.utc).isoformat(),
                  new_integration_steps=0, policy_inference_calls=0, optimizer_updates=0)
    try:
        torch.set_num_threads(8)
        library = json.loads(args.library.read_text())
        rows = [r for values in library["splits"].values() for r in values if r["name"] in FIXED]
        if len(rows) != 3 or len({r["name"] for r in rows}) != 3:
            raise ValueError("Expected the three unique original fixed clips")
        files = [args.scene, args.library, Path(__file__), HERE / "base_mdp.py",
                 HERE / "sole_geometry.py", HERE / "tensor_geometry.py"]
        native = args.runtime / "src/unisupermimic/tasks/motion_tracking/q1/supermimic"
        files.extend(native / name for name in ("mdp.py", "mdp_base.py", "kinematics.py", "candidate_options.json"))
        for row in rows:
            if identity(row["path"])["sha256"] != row["sha256"]:
                raise ValueError("Original reference identity changed")
            files.append(Path(row["path"]))
        before = [identity(path) for path in files]
        kin = Q1Kinematics(args.scene, "cpu")
        motion = MotionLib([r["path"] for r in rows], kin, "cpu", src_fps=30.)
        default = torch.tensor(default_joint_pos_list())
        old = MDP(kin, motion, default, observation_mode="root_xy")
        new = SoleMDP(kin, motion, default, scene=args.scene, observation_mode="root_xy")
        if old.policy_input_spec != new.policy_input_spec or old.max_step_reward != new.max_step_reward:
            raise AssertionError("Policy or reward scale contract changed")
        frames = torch.unique(torch.cat([torch.linspace(int(s), int(e), 17).long()
                                        for s, e in zip(motion.clip_start, motion.clip_end)]))
        reference = motion.ref_state(frames)
        actions = torch.full((len(frames), 22), .03)
        previous = actions + .007
        context = torch.zeros(len(frames), 26)
        context[:, 0] = 1
        stats = {}
        for name, root_z, ankle in (("self_reference", 0., 0.), ("root_z", .01, 0.),
                                    ("right_ankle_roll", 0., .12), ("combined", -.015, -.12)):
            state = tuple(value.clone() for value in reference)
            state[0][:, 2] += root_z
            state[4][:, 11] += ankle
            reward, parts = old.reward(state, frames, actions, previous, terms=True)
            actual, changed = new.reward(state, frames, actions, previous, terms=True)
            for key in parts.keys() - {"track", "e_body_pos"}:
                torch.testing.assert_close(parts[key], changed[key], atol=0, rtol=0)
            torch.testing.assert_close(actual - reward, changed["track"] - parts["track"], atol=2e-8, rtol=2e-5)
            torch.testing.assert_close(old.obs(state, actions, context, frames), new.obs(state, actions, context, frames), atol=0, rtol=0)
            torch.testing.assert_close(old.critic_features(state, frames, context), new.critic_features(state, frames, context), atol=0, rtol=0)
            torch.testing.assert_close(old.termination(state, frames), new.termination(state, frames), atol=0, rtol=0)
            torch.testing.assert_close(old.soft_termination(state, frames), new.soft_termination(state, frames), atol=0, rtol=0)
            # Reconstruct the exact mean with only the two foot-Z slots replaced.
            full = kin.forward(*state, all_bodies=True)
            old_z = (motion.body_p[frames][:, new.foot_slots, 2] -
                     full[0][:, kin.tracked_idx][:, new.foot_slots, 2]).square().sum(-1)
            sole_z = (new.sole(full[0], full[1]) - new.reference_sole[frames]).square().sum(-1)
            expected = parts["e_body_pos"] + (sole_z - old_z) / len(kin.tracked_idx)
            torch.testing.assert_close(changed["e_body_pos"], expected, atol=2e-8, rtol=2e-5)
            if name == "self_reference":
                torch.testing.assert_close(actual, reward, atol=2e-7, rtol=2e-5)
            stats[name] = float((actual - reward).abs().max())
        # The new reward must supply a real FK/sole derivative, not just finite values.
        pose = [value.clone() for value in reference]
        joint = pose[4].clone()
        joint[:, 11] += .123
        joint.requires_grad_()
        pose[4] = joint
        gradient = torch.autograd.grad(new.reward(tuple(pose), frames, actions, previous).sum(), joint)[0]
        if not bool(torch.isfinite(gradient).all()):
            raise AssertionError("Nonfinite base reward gradient")
        epsilon = 1e-3
        sides = []
        for sign in (-1, 1):
            shifted = list(pose)
            shifted[4] = joint.detach().clone()
            shifted[4][:, 11] += sign * epsilon
            sides.append(new.reward(tuple(shifted), frames, actions, previous).detach())
        difference = (sides[1] - sides[0]) / (2 * epsilon)
        torch.testing.assert_close(gradient[:, 11], difference, atol=3e-5, rtol=2e-2)
        if before != [identity(path) for path in files]:
            raise ValueError("Source changed during base contract")
        result.update(status="BASE_SOLE_CPU_CONTRACT_PASSED", sources=before,
                      fixed_clips=3, sampled_poses=len(frames),
                      only_two_foot_z_position_slots_changed=True,
                      policy_critic_features_termination_other_reward_unchanged=True,
                      max_step_reward_unchanged=True, reward_change_at_perturbations=stats,
                      maximum_ankle_roll_gradient_finite_difference_error=float((gradient[:, 11] - difference).abs().max()),
                      sole_contract=new.sole_contract,
                      limits=["Original reference poses and synthetic perturbations only; no physical gait benefit proven.",
                              "Native16/4096 integration and H32 direction checks remain separate."])
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (args.output / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "scene", "library", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
