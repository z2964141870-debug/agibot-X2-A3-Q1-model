"""Check phase/sole candidates on original CPU poses without integration."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(status="INITIALIZING", captured_utc=datetime.now(timezone.utc).isoformat(),
                  new_integration_steps=0, policy_inference_calls=0, optimizer_updates=0)
    try:
        import torch
        sys.path[:0] = [str(HERE), str(args.runtime / "src"), str(args.runtime),
                       str(HERE.parent / "supermimic_q1_sole"), str(HERE.parent / "supermimic_q1_contact_comp")]
        from check_contract import identity
        from physical_gate import FIXED
        from reward_candidates import PhaseSoleMDP, PhaseSoleReference, lowest_material_velocity, phase_mean
        from contact_reference import ContactReference, configure
        from unisupermimic.rl.algos.supermimic.geometry import flatten_state
        from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
        from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
        from unisupermimic.tasks.motion_tracking.q1.supermimic.mdp import MDP
        from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import default_joint_pos_list

        torch.set_num_threads(8)
        source = json.loads(args.library.read_text())
        rows = [row for split in source["splits"].values() for row in split if row["name"] in FIXED]
        if len(rows) != 3 or len({row["name"] for row in rows}) != 3:
            raise ValueError("Expected original fixed3 references")
        paths = [Path(__file__), HERE / "reward_candidates.py", HERE / "support_geometry.py", args.scene, args.library]
        for row in rows:
            if identity(row["path"])["sha256"] != row["sha256"]:
                raise ValueError("Original reference changed")
            paths.extend((Path(row["path"]), Path(row["path"]).with_suffix(".json")))
        before = [identity(path) for path in paths]
        kin = Q1Kinematics(args.scene, "cpu")
        motion = MotionLib([row["path"] for row in rows], kin, "cpu", src_fps=30.)
        default = torch.tensor(default_joint_pos_list())
        original = MDP(kin, motion, default, observation_mode="root_xy")
        candidate = PhaseSoleMDP(kin, motion, default, scene=args.scene, library=args.library)
        configure(args.library)
        original_reference = ContactReference(original)
        residual_reference = PhaseSoleReference(original, args.scene)
        frames = torch.unique(torch.cat([torch.linspace(int(start), int(end), 17).long()
                                        for start, end in zip(motion.clip_start, motion.clip_end)]))
        reference = motion.ref_state(frames)
        from tensor_geometry import TensorSole
        from sole_geometry import SoleGeometry
        full = kin.forward(*reference, all_bodies=True)
        original_sole = TensorSole(SoleGeometry.from_scene(args.scene), kin.body_names, "cpu")
        torch.testing.assert_close(candidate.foot_reference.sole(full[0], full[1]),
                                   original_sole(full[0], full[1]), atol=2e-7, rtol=2e-6)
        action = torch.full((len(frames), 22), .03)
        previous = action + .007
        context = torch.zeros(len(frames), 26)
        context[:, 0] = 1
        comparisons = []
        for name, root_z, ankle in (("reference", 0., 0.), ("root_z", .01, 0.),
                                   ("ankle_roll", 0., .123), ("combined", -.015, -.123)):
            state = tuple(value.clone() for value in reference)
            state[0][:, 2] += root_z
            state[4][:, 11] += ankle
            old, old_parts = original.reward(state, frames, action, previous, terms=True)
            new, new_parts = candidate.reward(state, frames, action, previous, terms=True)
            for key in old_parts.keys() - {"track"}:
                torch.testing.assert_close(old_parts[key], new_parts[key], atol=0, rtol=0)
            torch.testing.assert_close(new - old, new_parts["phase_bonus"], atol=2e-8, rtol=2e-5)
            if bool((new_parts["phase_bonus"] < 0).any()) or bool((new_parts["phase_bonus"] > .01000001).any()):
                raise AssertionError("Phase bonus exceeds predeclared maximum")
            for method, values in (("obs", (state, action, context, frames)),
                                   ("critic_features", (state, frames, context)),
                                   ("termination", (state, frames)), ("soft_termination", (state, frames))):
                torch.testing.assert_close(getattr(original, method)(*values),
                                           getattr(candidate, method)(*values), atol=0, rtol=0)
            flat = flatten_state(state)
            torch.testing.assert_close(original_reference.features(flat, frames),
                                       residual_reference.features(flat, frames), atol=0, rtol=0)
            residual = torch.zeros(len(frames), 14)
            force, slip = torch.full((len(frames), 2), 10.), torch.full((len(frames), 2), .05)
            old_ppo, old_extra = original_reference.reward(old, flat, frames, action, previous, residual, force, slip)
            new_ppo, new_extra = residual_reference.reward(old, flat, frames, action, previous, residual, force, slip)
            for key in old_extra.keys() - {"swing_bonus"}:
                torch.testing.assert_close(old_extra[key], new_extra[key], atol=0, rtol=0)
            full = kin.forward(*state, all_bodies=True)
            ankle_error = full[0][:, residual_reference.ids, 2] - residual_reference.positions[frames, :, 2]
            sole_error = residual_reference.sole(full[0], full[1]) - residual_reference.reference_sole[frames]
            correction = .01 * phase_mean(torch.exp(-(sole_error / .02).square()) -
                torch.exp(-(ankle_error / .04).square()), residual_reference.swing[frames])
            torch.testing.assert_close(new_ppo - old_ppo, correction, atol=3e-8, rtol=3e-5)
            comparisons.append(dict(perturbation=name, base_bonus_min=float((new-old).min()),
                                    base_bonus_max=float((new-old).max()), ppo_z_delta_max=float(correction.abs().max())))
        # A synthetic translating/rotating body gives an independent rigid-point velocity check.
        full = tuple(value.clone() for value in kin.forward(*reference, all_bodies=True))
        full[2].zero_()
        full[3].zero_()
        full[2][..., 0] = .2
        measured = lowest_material_velocity(candidate.foot_reference.sole, full)
        torch.testing.assert_close(measured[..., 0], torch.full_like(measured[..., 0], .2), atol=1e-8, rtol=0)
        torch.testing.assert_close(measured[..., 1:], torch.zeros_like(measured[..., 1:]), atol=0, rtol=0)
        neutral = torch.zeros(len(frames), 2, dtype=torch.bool)
        torch.testing.assert_close(phase_mean(torch.ones(len(frames), 2), neutral), torch.zeros(len(frames)))
        # Root-Z does not switch the lowest material point, so finite differences are well defined here.
        pose = [value.clone() for value in reference]
        pose[0][:, 2] += .011
        pose[0].requires_grad_()
        gradient = torch.autograd.grad(candidate.reward(tuple(pose), frames, action, previous).sum(), pose[0])[0][:, 2]
        values = []
        for sign in (-1, 1):
            shifted = list(pose)
            shifted[0] = pose[0].detach().clone()
            shifted[0][:, 2] += sign * 1e-4
            values.append(candidate.reward(tuple(shifted), frames, action, previous).detach())
        finite = (values[1]-values[0]) / 2e-4
        torch.testing.assert_close(gradient, finite, atol=2e-4, rtol=1e-2)
        if before != [identity(path) for path in paths]:
            raise ValueError("Candidate contract sources changed")
        result.update(status="GAIT_PHASE_SOLE_CPU_CONTRACTS_PASSED", sources=before,
                      sampled_poses=len(frames), fixed_clips=3, perturbations=comparisons,
                      original_base_terms_features_termination_preserved=True,
                      exact_original_collision_point_heights_verified=True,
                      ppo_only_swing_z_geometry_and_width_changed=True,
                      base_max_step_reward=candidate.max_step_reward,
                      base_original_max_step_reward=original.max_step_reward,
                      maximum_root_z_gradient_finite_difference_error=float((gradient-finite).abs().max()),
                      phase_contract=candidate.phase_contract,
                      limits=["CPU reference poses and synthetic perturbations, no physical gait claim.",
                          "Material-point stance speed is differentiable intent-conditioned feedback, not measured load.",
                          "Production sampler reward bound must be updated to .11 only for the independent base candidate.",
                          "Native16/4096, H32 interaction and formal training remain separate gates."])
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (args.output / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "sampled_poses", "base_max_step_reward",
        "maximum_root_z_gradient_finite_difference_error")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "scene", "library", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
