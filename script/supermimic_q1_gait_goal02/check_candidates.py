"""Check native GOAL02 reward composition on CPU without a control environment."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(status="INITIALIZING", new_integration_steps=0,
                  policy_inference_calls=0, optimizer_updates=0)
    try:
        import torch
        sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_gait_goal"),
                       str(HERE.parent / "supermimic_q1_sole"),
                       str(HERE.parent / "supermimic_q1_contact_comp"),
                       str(args.runtime / "src"), str(args.runtime)]
        from check_contract import identity
        from goal02_config import contract
        from reward_candidates import PhaseSoleMDP, PhaseSoleReference
        from coupled_rewards import CoupledMDP, CoupledReference
        from unisupermimic.rl.algos.supermimic.geometry import flatten_state
        from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
        from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
        from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import default_joint_pos_list

        torch.set_num_threads(1)
        fixed = {"walk_ff_stop_180_R_002__A095", "walk_ff_start_180_R_002__A071",
                 "triumph_two_handed_001__A058"}
        rows = [row for split in json.loads(args.library.read_text())["splits"].values()
                for row in split if row["name"] in fixed]
        if len(rows) != 3 or {row["name"] for row in rows} != fixed:
            raise ValueError("Original fixed3 references required")
        paths = [*HERE.glob("*.py"), args.scene, args.library]
        for row in rows:
            path = Path(row["path"])
            if identity(path)["sha256"] != row["sha256"]:
                raise ValueError("Reference identity mismatch")
            paths.extend((path, path.with_suffix(".json")))
        before = [identity(path) for path in paths]
        kin = Q1Kinematics(args.scene, "cpu")
        motion = MotionLib([row["path"] for row in rows], kin, "cpu", src_fps=30.)
        default = torch.tensor(default_joint_pos_list())
        old = PhaseSoleMDP(kin, motion, default, scene=args.scene, library=args.library)
        new = CoupledMDP(kin, motion, default, scene=args.scene, library=args.library)
        old_ppo, new_ppo = PhaseSoleReference(old, args.scene), CoupledReference(old, args.scene)
        frames = torch.unique(torch.cat([torch.linspace(int(start), int(end), 17).long()
                                        for start, end in zip(motion.clip_start, motion.clip_end)]))
        reference = motion.ref_state(frames)
        action = torch.full((len(frames), 22), .03)
        previous = action + .007
        context = torch.zeros(len(frames), 26)
        context[:, 0] = 1
        comparisons = []
        for name, z, roll in (("reference", 0., 0.), ("root_z", .011, 0.),
                              ("roll_positive", 0., .08), ("roll_negative", -.01, -.08)):
            state = tuple(value.clone() for value in reference)
            state[0][:, 2] += z
            angle = torch.tensor(roll)
            rotation = torch.eye(3)
            rotation[1, 1] = rotation[2, 2] = angle.cos()
            rotation[1, 2], rotation[2, 1] = -angle.sin(), angle.sin()
            state[1][:] = rotation @ state[1]
            old_reward, old_parts = old.reward(state, frames, action, previous, terms=True)
            new_reward, new_parts = new.reward(state, frames, action, previous, terms=True)
            for key in old_parts.keys() - {"track", "phase_bonus"}:
                torch.testing.assert_close(old_parts[key], new_parts[key], atol=0, rtol=0)
            torch.testing.assert_close(new_parts["phase_bonus_uncoupled"], old_parts["phase_bonus"], atol=0, rtol=0)
            expected = old_parts["phase_bonus"] * new_parts["posture_factor"] * new_parts["pair_factor"]
            torch.testing.assert_close(new_parts["phase_bonus"], expected, atol=1e-9, rtol=2e-6)
            torch.testing.assert_close(new_reward - old_reward,
                new_parts["phase_bonus"] - old_parts["phase_bonus"], atol=2e-8, rtol=2e-5)
            if bool((new_parts["phase_bonus"] < 0).any()) or bool((new_parts["phase_bonus"] > .01000001).any()):
                raise AssertionError("Bounded base bonus exceeded")
            for method, values in (("obs", (state, action, context, frames)),
                    ("critic_features", (state, frames, context)),
                    ("termination", (state, frames)), ("soft_termination", (state, frames))):
                torch.testing.assert_close(getattr(old, method)(*values), getattr(new, method)(*values), atol=0, rtol=0)
            flat = flatten_state(state)
            torch.testing.assert_close(old_ppo.features(flat, frames), new_ppo.features(flat, frames), atol=0, rtol=0)
            residual = torch.zeros(len(frames), 14)
            force, slip = torch.full((len(frames), 2), 10.), torch.full((len(frames), 2), .05)
            p0, parts0 = old_ppo.reward(old_reward, flat, frames, action, previous, residual, force, slip)
            p1, parts1 = new_ppo.reward(old_reward, flat, frames, action, previous, residual, force, slip)
            for key in ("stationary_bonus", "smooth_cost", "residual_cost"):
                torch.testing.assert_close(parts0[key], parts1[key], atol=0, rtol=0)
            expected_ppo = old_reward + parts1["stance_bonus"] + parts1["swing_bonus"] + parts1["stationary_bonus"]
            expected_ppo -= parts1["smooth_cost"] + parts1["residual_cost"]
            torch.testing.assert_close(p1, expected_ppo, atol=3e-8, rtol=3e-6)
            if bool((p1 > p0 + 3e-8).any()):
                raise AssertionError("Coupling cannot increase an existing foot bonus")
            comparisons.append(dict(perturbation=name, base_delta_min=float((new_reward-old_reward).min()),
                base_delta_max=float((new_reward-old_reward).max()), ppo_delta_min=float((p1-p0).min()),
                posture_factor_min=float(new_parts["posture_factor"].min()),
                pair_factor_min=float(new_parts["pair_factor"].min())))
        pose = [value.clone() for value in reference]
        pose[0][:, 2] += .011
        pose[0].requires_grad_()
        gradient = torch.autograd.grad(new.reward(tuple(pose), frames, action, previous).sum(), pose[0])[0][:, 2]
        values = []
        for sign in (-1, 1):
            shifted = list(pose)
            shifted[0] = pose[0].detach().clone()
            shifted[0][:, 2] += sign * 1e-4
            values.append(new.reward(tuple(shifted), frames, action, previous).detach())
        finite = (values[1] - values[0]) / 2e-4
        torch.testing.assert_close(gradient, finite, atol=2e-4, rtol=1e-2)
        if before != [identity(path) for path in paths]:
            raise ValueError("CPU contract source changed")
        result.update(status="GOAL02_NATIVE_CPU_REWARD_CONTRACT_PASSED", completed_utc=datetime.now(timezone.utc).isoformat(),
            sampled_poses=len(frames), perturbations=comparisons, sources=before, recipe=contract(),
            features_costs_terminations_original_tracking_preserved=True,
            maximum_root_z_gradient_finite_difference_error=float((gradient-finite).abs().max()),
            new_base_maximum=new.max_step_reward, physical_gait_improvement_established=False)
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (args.output / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "sampled_poses",
          "maximum_root_z_gradient_finite_difference_error")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "scene", "library", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
