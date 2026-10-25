#!/usr/bin/env python3
"""Evaluate readable Q1 references with the trained actor in real MuJoCo."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
CORE = HERE.parent / "supermimic_for_q1"
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(HERE.parent / "supermimic_q1_sources"))
sys.path.insert(0, str(HERE.parent / "supermimic_q1_stability"))

def select_clips(available, requested=None, count=8):
    names = list(requested) if requested is not None else sorted(available)[:count]
    if count < 1 or not names or len(names) != len(set(names)) or not set(names) <= set(available):
        raise ValueError("Select distinct readable references from this library")
    return names


def summarize_trace(state, rewards, joints, body, terminated, reference_root, initial_root, max_reward):
    count, expected = len(state), len(reference_root)
    if not 0 < count <= expected:
        raise ValueError("A physical trace must contain 1..reference-length steps")
    root = np.asarray(state, dtype=np.float64)[:, :3]
    target = np.asarray(reference_root, dtype=np.float64)
    initial = np.asarray(initial_root, dtype=np.float64)
    trajectory = np.concatenate([initial[None], root], axis=0)
    failed = bool(terminated[-1])
    return {
        "expected_steps": expected, "executed_steps": count, "failed": failed,
        "early_terminated": failed and count < expected,
        "termination_step": count if failed else None,
        "reward_sum": float(np.asarray(rewards, dtype=np.float64).sum()),
        "normalized_return": float(np.asarray(rewards, dtype=np.float64).sum()) / (max_reward * expected),
        "full_clip_completion": count / expected,
        "executed_joint_rmse_rad": float(np.sqrt(np.mean(np.square(np.asarray(joints, dtype=np.float64))))),
        "executed_body_error_mean_m": float(np.asarray(body, dtype=np.float64).mean()),
        "root_xy_error_mean_m": float(np.linalg.norm(root[:, :2] - target[:count, :2], axis=-1).mean()),
        "root_xyz_error_mean_m": float(np.linalg.norm(root - target[:count], axis=-1).mean()),
        "root_xy_final_error_m": float(np.linalg.norm(root[-1, :2] - target[count - 1, :2])),
        "root_xy_displacement_m": float(np.linalg.norm(root[-1, :2] - initial[:2])),
        "root_xy_path_length_m": float(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=-1).sum()),
        "reference_root_xy_displacement_full_m": float(np.linalg.norm(target[-1, :2] - target[0, :2])),
        "reference_root_xy_path_length_full_m": float(np.linalg.norm(np.diff(target[:, :2], axis=0), axis=-1).sum()),
        "reference_root_xy_displacement_executed_m": float(np.linalg.norm(target[count - 1, :2] - target[0, :2])),
        "initial_root_position_w_m": initial.tolist(),
        "final_root_position_w_m": root[-1].tolist(),
    }


def aggregate_rows(rows):
    groups = {}
    for split in sorted({row["split"] for row in rows}):
        selected = [row for row in rows if row["split"] == split]
        groups[split] = {
            "clips": len(selected), "failures": sum(row["failed"] for row in selected),
            "early_terminations": sum(row["early_terminated"] for row in selected),
            **{"mean_" + key: float(np.mean([row[key] for row in selected])) for key in (
                "normalized_return", "full_clip_completion", "executed_joint_rmse_rad",
                "executed_body_error_mean_m", "root_xy_error_mean_m", "root_xyz_error_mean_m",
                "root_xy_displacement_m", "root_xy_path_length_m",
                "reference_root_xy_displacement_full_m")},
        }
    return groups


def load_references(motions, manifest, view):
    from reference_admission import POLICY
    from supermimic.q1_assets import identity

    source_path = motions / "manifest.json"
    source = json.loads(source_path.read_text())
    receipt_record = manifest["reference_receipt"]
    view.verify(receipt_record)
    checked = json.loads(Path(receipt_record["path"]).read_text())
    if (source["admission_policy"] != POLICY or checked["admission_policy"] != POLICY
            or checked["status"] != "Q1_REFERENCE_IO_ADMISSION_COMPLETED"):
        raise ValueError("The reproduction IO reference policy is required")
    if identity(source_path) != checked["input_manifest"]:
        raise ValueError("Evaluation library differs from the training reference receipt")
    if "motion_manifest" in manifest and identity(source_path) != manifest["motion_manifest"]:
        raise ValueError("Training and evaluation motion manifests differ")
    io_valid = {row["clip"] for row in checked["clips"] if row["io_valid"]}
    available = {row["clip"]: row for row in source["motions"]
                 if row.get("admitted_for_training", False) and row["clip"] in io_valid}
    return source_path, checked, available


def main():
    import torch

    from contact_protocol import ContactCorrectedAssets
    from bounded_policy import ACTION_LIMIT, BoundedPolicy
    from relocated_io import JsonView, migration_view
    from reference_admission import POLICY, validate_arrays
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.policy.nets import PolicyCfg
    from supermimic.q1_assets import TRACKED, identity
    from supermimic.q1_sim import Q1Sim, SimCfg

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--clips", nargs="+", default=None)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    view = migration_view(args.root)
    receipts = JsonView(view)
    checkpoint_identity = identity(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    manifest = view.rebase(checkpoint["manifest"])
    policy = BoundedPolicy(PolicyCfg(**manifest["policy"])).to(args.device).eval()
    policy.load_state_dict(checkpoint["policy"], strict=True)
    assets = ContactCorrectedAssets(args.root)
    if assets.manifest() != manifest["assets"]:
        raise ValueError("Evaluation assets/control differ from the trained Q1 contract")
    view.verify(manifest["backend_receipt"])
    backend = receipts.loads(Path(manifest["backend_receipt"]["path"]).read_text())
    if backend["status"] != "BACKEND_AND_GRADIENT_CONTRACT_PASSED":
        raise ValueError("Unrecognized backend receipt")
    for source in backend["source_files"]:
        view.verify(source)
    source_path, checked, available = load_references(args.motions, manifest, view)
    names = select_clips(available, args.clips, args.count)
    heads = assets.heads(args.device)
    training_names = {Path(record["path"]).stem for record in manifest["train_clips"]}
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    started = time.monotonic()
    with torch.no_grad():
        for name in names:
            reference_file = args.motions / (name + ".npz")
            if identity(reference_file) != available[name]["output"]:
                raise ValueError("Prepared Q1 reference identity changed: " + name)
            validate_arrays(reference_file)
            library = MotionLibrary([reference_file], TRACKED, assets.q0,
                                    device=args.device, active_indices=assets.active_indices)
            expected = library.num_frames
            config = SimCfg(num_envs=1, threads=1, domain_randomization=False,
                            observation_noise=False, action_noise=0., episode_steps=expected + 1)
            sim = Q1Sim(assets, library, config, heads, seed=0)
            sim.reset(sim.ids, torch.tensor([0], device=args.device))
            initial_state = sim.state[0].cpu().numpy().copy()
            states, actions, rewards, joints, body, terminated = [], [], [], [], [], []
            try:
                for _ in range(expected):
                    observation, goal = sim.policy_inputs()
                    action = policy(observation, goal)
                    result = sim.step(action, auto_reset=False)
                    states.append(result["state"][0].cpu().numpy())
                    actions.append(action[0].cpu().numpy())
                    rewards.append(float(result["reward"][0]))
                    joints.append(float(result["joint_rmse"][0]))
                    body.append(float(result["body_error_m"][0]))
                    terminated.append(bool(result["terminated"][0]))
                    if terminated[-1]:
                        break
            finally:
                sim.close()
            reference_root = library.root_pos_w.cpu().numpy()
            trace_path = args.output / (name + ".npz")
            np.savez_compressed(trace_path, state=np.asarray(states), actions=np.asarray(actions),
                                reward=np.asarray(rewards), joint_rmse_rad=np.asarray(joints),
                                body_error_m=np.asarray(body), terminated=np.asarray(terminated),
                                initial_state=initial_state, reference_frames=np.arange(len(states)))
            row = {"clip": name, "split": "training" if name in training_names else "heldout",
                   **summarize_trace(states, rewards, joints, body, terminated,
                                     reference_root, initial_state[:3], heads.reward_cfg.max_step_reward),
                   "reference": identity(reference_file), "trace": identity(trace_path)}
            rows.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
    receipt = {
        "status": "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED", "checkpoint": checkpoint_identity,
        "training_step": checkpoint["step"], "policy_updates": checkpoint["policy_updates"],
        "rows": rows, "groups": aggregate_rows(rows), "clips": names,
        "seconds": time.monotonic() - started, "assets": assets.manifest(),
        "simulator": "MuJoCo CPU native single-environment rollout",
        "protocol": "Reference frame0 initial state; one control per reference frame; full-clip return/completion denominator includes early failure; no noise/DR; friction1; free base",
        "metric_scope": "Executed error metrics cover executed controls only; movement includes the frame0 initial root; full-reference movement is reported separately",
        "action_limit": ACTION_LIMIT, "max_step_reward": heads.reward_cfg.max_step_reward,
        "reference_admission_policy": POLICY, "local_geometry_gates_used_for_admission": False,
        "reference_manifest": identity(source_path), "reference_receipt": manifest["reference_receipt"],
        "evaluation_entry": identity(__file__), "relocation": view.evidence(),
        "storage_status": "LOCAL_ONLY", "not_hardware_validated": True,
    }
    with (args.output / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt["groups"], indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
