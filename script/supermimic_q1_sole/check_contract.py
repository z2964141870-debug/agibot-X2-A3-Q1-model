"""Check the sole candidate on original saved poses, with no physics or optimizer."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def identity(path):
    data = Path(path).read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def run(args):
    import mujoco
    import numpy as np
    import torch
    from types import SimpleNamespace

    sys.path[:0] = [str(HERE), str(args.runtime / "src"), str(args.runtime),
                   str(HERE.parent / "project_tools"),
                   str(HERE.parent / "supermimic_q1_contact_comp")]
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state, mat_to_quat
    from contact_reference import configure, ContactReference
    from sole_reference import SoleReference
    from audit_q1_saved_sole_geometry import foot_geometry, measure
    from prepare_q1_gait_review import FIXED

    args.output.mkdir(parents=True, exist_ok=False)
    status = dict(status="INITIALIZING", captured_utc=datetime.now(timezone.utc).isoformat(),
                  new_integration_steps=0, policy_inference_calls=0, optimizer_updates=0)
    path = args.output / "receipt.json"
    try:
        sources = [identity(p) for p in (args.scene, args.library, Path(__file__),
                    HERE / "sole_geometry.py", HERE / "tensor_geometry.py",
                    HERE / "sole_reference.py", HERE / "sole_adapter.py")]
        library = json.loads(args.library.read_text())
        all_rows = [row for values in library["splits"].values() for row in values]
        rows = [row for row in all_rows if row["name"] in FIXED]
        if len(rows) != 3 or len({row["name"] for row in all_rows}) != len(all_rows):
            raise ValueError("Expected unique original library including the three fixed clips")
        for row in rows:
            source = identity(Path(row["path"]))
            if source["sha256"] != row["sha256"]:
                raise ValueError("Fixed reference identity changed")
            sources.extend((source, identity(Path(row["path"]).with_suffix(".json"))))
        before = list(sources)
        torch.set_num_threads(8)
        kin = Q1Kinematics(args.scene, "cpu")
        motion = MotionLib([row["path"] for row in rows], kin, "cpu", src_fps=30.)
        mdp = SimpleNamespace(kin=kin, motion=motion)
        configure(args.library)
        baseline = ContactReference(mdp)
        candidate = SoleReference(mdp, args.scene)
        for name in ("positions", "velocities", "floor", "stance", "swing"):
            torch.testing.assert_close(getattr(candidate, name), getattr(baseline, name), atol=0, rtol=0)
        model = mujoco.MjModel.from_xml_path(str(args.scene))
        data = mujoco.MjData(model)
        geometry = [foot_geometry(model, side) for side in ("left", "right")]
        indices = torch.unique(torch.cat([torch.linspace(int(start), int(end), 17).long()
                                         for start, end in zip(motion.clip_start, motion.clip_end)]))
        state_tuple = motion.ref_state(indices)
        p, rotation, v, w, q, qd = state_tuple
        poses = torch.cat((p, mat_to_quat(rotation), q), -1).numpy()
        expected, _ = measure(model, data, poses, geometry)
        fk = kin.forward(*state_tuple, all_bodies=True)
        actual = candidate.sole(fk[0], fk[1])
        np.testing.assert_allclose(actual.numpy(), expected, atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(candidate.reference_sole[indices], actual, atol=2e-6, rtol=2e-5)
        # The production state contract stores quaternions, world v/w and joint q/qd.
        state = torch.cat((p, mat_to_quat(rotation), v, w, q, qd), -1)
        restored = unflatten_state(state)
        for original, decoded in zip(state_tuple, restored):
            torch.testing.assert_close(original, decoded, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(candidate.features(state, indices), baseline.features(state, indices), atol=0, rtol=0)
        count = len(indices)
        original = torch.linspace(-.02, .1, count)
        targets = q.clone()
        previous = q + .003
        residual = torch.zeros(count, 14)
        force = torch.full((count, 2), 20.)
        slip = torch.full((count, 2), .07)
        joint = model.joint("right_ankle_roll_joint")
        joint_column = 13 + int(model.jnt_qposadr[joint.id]) - 7
        errors = {}
        cases = (("self_reference", 0., 0.), ("root_z_plus10mm", .01, 0.),
                 ("root_z_minus15mm", -.015, 0.), ("right_ankle_roll_plus012rad", 0., .12))
        for label, offset, ankle_roll in cases:
            altered = state.clone()
            altered[:, 2] += offset
            altered[:, joint_column] += ankle_roll
            old_reward, old_parts = baseline.reward(original, altered, indices, targets, previous, residual, force, slip)
            reward, parts = candidate.reward(original, altered, indices, targets, previous, residual, force, slip)
            for name in old_parts.keys() - {"swing_bonus"}:
                torch.testing.assert_close(parts[name], old_parts[name], atol=0, rtol=0)
            torch.testing.assert_close(reward - old_reward, parts["swing_bonus"] - old_parts["swing_bonus"], atol=2e-8, rtol=2e-5)
            no_swing = ~candidate.swing[indices].any(-1)
            torch.testing.assert_close(reward[no_swing], old_reward[no_swing], atol=0, rtol=0)
            error = float((reward - old_reward).abs().max())
            if label == "self_reference":
                torch.testing.assert_close(reward, old_reward, atol=2e-7, rtol=2e-5)
            errors[label] = error
        sources = [identity(Path(row["path"])) for row in before]
        if sources != before:
            raise ValueError("Source changed during candidate contract check")
        status.update(status="SOLE_CPU_POSE_CONTRACT_PASSED", sources=sources,
                      fixed_clips=len(rows), sampled_poses=count,
                      maximum_sole_geometry_error_m=float(np.abs(actual.numpy() - expected).max()),
                      features_phase_and_other_reward_parts_unchanged=True,
                      self_reference_reward_equivalence=True, no_swing_exact_equivalence=True,
                      maximum_reward_change_at_pose_perturbations=errors,
                      phase_and_geometry=candidate.contract,
                      limits=["Saved reference poses and synthetic root-Z/ankle-roll offsets; no physical success evidence.",
                              "Native16/4096 zero-update checks and H32 learner-direction validation remain separate.",
                              "No teacher or trained head is loaded; no candidate checkpoint is accepted."])
    except BaseException as error:
        status.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        path.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "scene", "library", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())
