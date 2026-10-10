"""E19 frozen deployment-policy sensitivity at common saved CV robot states."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.corrected_trial import OFFICIAL
from script.a3.fullchain_support import ROOT, VENDOR
from script.a3.reference_contract import load_sim
from script.a3.verify_joint_dynamics import independent_window


SOURCE = ROOT / "data/experiments/a3_causal_reference_20261010_E18"
OUTPUT = ROOT / "data/experiments/a3_controller_reference_20261010_E19"
MODES = ("hold", "cv", "smooth_cv", "E12", "E14")


def tensor_hash(policy):
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        digest.update(name.encode("ascii"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def forward(policy, packed, proprio, device):
    per_frame = np.concatenate((packed[:580].reshape(10, 58), packed[580:].reshape(10, 6)), axis=1)
    x = torch.as_tensor(per_frame, dtype=torch.float32, device=device).reshape(1, 640)
    prop = torch.as_tensor(proprio, dtype=torch.float32, device=device).reshape(1, 930)
    latent = policy.encoder(x).view(1, 2, 32)
    tokens = policy.fsq(latent).reshape(1, 64)
    action = policy.decoder(torch.cat((tokens, prop), dim=-1))
    return latent.reshape(64), tokens.reshape(64), action.reshape(29), x, prop


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "result.json").exists():
        print("E19 completed; receipt preserved")
        return 0
    sim = load_sim()
    device = torch.device("cuda")
    policy = sim.A3Policy(OFFICIAL, device, encoder="a3_fast")
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    before = tensor_hash(policy)
    asset = VENDOR / "gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml"
    permutation = sim.build_loop_runtime(mujoco.MjModel.from_xml_path(str(asset)), sim.DEFAULT_URDF).mapping.mj29_to_il
    source_job = json.loads((SOURCE / "job.json").read_text())
    models = {}
    for mode in ("E12", "E14"):
        with np.load(ROOT / f"data/experiments/a3_future_reference_20261010_{mode}/ridge_joint_future.npz", allow_pickle=False) as stored:
            models[mode] = {key: stored[key].copy() for key in ("mean", "std", "weights", "bias")}
    per_motion, sums = [], {str(s): {m: dict(q=0., velocity=0., latent=0., token=0., action=0., count=0) for m in MODES} for s in (0., .01)}
    teacher_pairs, gradient = [], None
    replay_max, checked = 0., 0
    for sigma in (0., .01):
        for source in source_job["motions"]:
            name = Path(source["path"]).stem
            directory = SOURCE / f"cv_noise{sigma:g}_seed0"
            with np.load(directory / f"{name}.inputs.npz", allow_pickle=False) as stored:
                observations, recorded_actions = stored["observations"].copy(), stored["actions"].copy()
            with np.load(directory / f"{name}.joint.npz", allow_pickle=False) as stored:
                anchors = stored["robot_anchor_quat_wxyz"].copy()
            _, quaternion, q, _ = sim.load_a3_flat_csv(Path(source["path"]), source_fps=30, frame_stride=4)
            times = np.arange(len(q))/30
            sample_times = np.arange(0., times[-1]-1e-12, .02)
            clean = np.column_stack([np.interp(sample_times, times, q[:, j]) for j in range(29)])[:, permutation]
            velocity = np.diff(clean, axis=0)*50
            velocity = np.concatenate((velocity, velocity[-1:]))
            clean_root = Slerp(times, Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]))(sample_times).as_quat()[:, [3, 0, 1, 2]]
            observed = q+np.random.default_rng(0).normal(0, sigma, q.shape)
            indices = np.linspace(0, len(observations)-1, 100, dtype=int)
            local = {m: dict(q=0., velocity=0., latent=0., token=0., action=0., count=0) for m in MODES}
            for tick in indices:
                current = observations[tick]
                native = tick*3//5
                base = native*5//3
                ids = np.minimum(base+np.arange(10), len(clean)-1)
                oracle_ori = sim.root_ori_diff_6d(anchors[tick], clean_root[ids]).reshape(-1)
                oracle = np.concatenate((clean[ids].reshape(-1), velocity[ids].reshape(-1), oracle_ori)).astype(np.float32)
                with torch.no_grad():
                    tl, tt, ta, _, _ = forward(policy, oracle, current[640:], device)
                    teacher_latent, teacher_tokens, teacher_action = [v.cpu().numpy() for v in (tl, tt, ta)]
                    for mode in MODES:
                        prediction = independent_window(observed, native, base, mode, models.get(mode))[:, permutation]
                        pos, vel = prediction[:10], np.diff(prediction, axis=0)*50
                        # All candidates share the same saved CV orientation and proprioception.
                        packed = np.concatenate((pos.reshape(-1), vel.reshape(-1), current[580:640])).astype(np.float32)
                        latent, tokens, action, _, _ = forward(policy, packed, current[640:], device)
                        latent, tokens, action = [v.cpu().numpy() for v in (latent, tokens, action)]
                        if mode == "cv":
                            np.testing.assert_allclose(packed, current[:640], atol=1e-6, rtol=1e-6)
                            np.testing.assert_allclose(action, recorded_actions[tick], atol=1e-6, rtol=1e-5)
                            replay_max = max(replay_max, float(np.abs(action-recorded_actions[tick]).max()))
                            checked += 1
                        measurements = dict(q=np.mean((pos-clean[ids])**2), velocity=np.mean((vel-velocity[ids])**2),
                            latent=np.mean((latent-teacher_latent)**2), token=np.mean(tokens != teacher_tokens),
                            action=np.mean((action-teacher_action)**2), count=1)
                        for key, value in measurements.items():
                            local[mode][key] += float(value)
                            sums[str(sigma)][mode][key] += float(value)
                if gradient is None and tick > 0:
                    _, _, _, x, prop = forward(policy, current[:640], current[640:], device)
                    x = x.detach().requires_grad_(True)
                    z = policy.encoder(x).view(1, 2, 32)
                    quantized = policy.fsq(z).reshape(1, 64)
                    action = policy.decoder(torch.cat((quantized, prop), dim=-1))
                    target = torch.as_tensor(teacher_action, device=device).reshape(1, 29)
                    action_gradient = torch.autograd.grad((action-target).square().mean(), x, retain_graph=True)[0]
                    latent_target = torch.as_tensor(teacher_latent, device=device).view(1, 2, 32)
                    latent_gradient = torch.autograd.grad((z-latent_target).square().mean(), x)[0]
                    gradient = dict(action_loss_input_gradient_norm=float(action_gradient.norm().item()),
                        prequant_latent_loss_input_gradient_norm=float(latent_gradient.norm().item()),
                        input_gradients_finite=bool(torch.isfinite(action_gradient).all() and torch.isfinite(latent_gradient).all()),
                        quantizer="PINNED_DEPLOYMENT_TORCH_ROUND_NO_STRAIGHT_THROUGH",
                        scope="ONE_SAVED_STATE_FROZEN_DEPLOYMENT_NETWORK_NOT_NATIVE_PPO_DIAGNOSTIC")
                teacher_pairs.append(dict(motion=name, sigma=sigma, tick=tick, oracle=oracle,
                    causal=current.copy(), latent=teacher_latent, tokens=teacher_tokens, action=teacher_action))
            metrics = {m: {k+"_rmse": math_sqrt(v[k]/v["count"]) for k in ("q", "velocity", "latent", "action")}
                       | dict(token_mismatch_fraction=v["token"]/v["count"]) for m, v in local.items()}
            per_motion.append(dict(motion=name, noise_std_rad=sigma, states=100, metrics=metrics))
            print(json.dumps(dict(motion=name, sigma=sigma, runtime_action_replay="passed")), flush=True)
    after = tensor_hash(policy)
    if before != after or gradient is None or not gradient["input_gradients_finite"]:
        raise ValueError("Frozen policy or gradient contract failed")
    pair_path = OUTPUT / "teacher_diagnostic.npz"
    np.savez_compressed(pair_path, motion=np.asarray([r["motion"] for r in teacher_pairs]),
        noise_std_rad=np.asarray([r["sigma"] for r in teacher_pairs]), policy_tick=np.asarray([r["tick"] for r in teacher_pairs]),
        oracle_inputs=np.asarray([r["oracle"] for r in teacher_pairs]), causal_inputs=np.asarray([r["causal"] for r in teacher_pairs]),
        teacher_prequant_latent=np.asarray([r["latent"] for r in teacher_pairs]),
        teacher_tokens=np.asarray([r["tokens"] for r in teacher_pairs]), teacher_actions=np.asarray([r["action"] for r in teacher_pairs]),
        usage=np.asarray("REPEATED_VALIDATION_DIAGNOSTIC_ONLY_NOT_TRAINING"), training_allowed=False)
    with np.load(pair_path, allow_pickle=False) as stored:
        if (stored["causal_inputs"].shape != (800, 1570) or stored["oracle_inputs"].shape != (800, 640) or
                stored["teacher_actions"].shape != (800, 29) or stored["training_allowed"].item() or
                any(not np.isfinite(stored[k]).all() for k in ("causal_inputs", "oracle_inputs", "teacher_prequant_latent", "teacher_tokens", "teacher_actions"))):
            raise ValueError("Teacher diagnostic artifact reload failed")
    aggregate = {s: {m: {k+"_rmse": math_sqrt(v[k]/v["count"]) for k in ("q", "velocity", "latent", "action")}
                    | dict(token_mismatch_fraction=v["token"]/v["count"]) for m, v in ms.items()} for s, ms in sums.items()}
    result = dict(status="complete", experiment="E19", code_sha256=sha256(Path(__file__)),
        scope="FROZEN_DEPLOYMENT_MODEL_SAME_SAVED_CV_STATES_COUNTERFACTUAL_NOT_NEW_CLOSED_LOOP_OR_NATIVE_PPO",
        runtime_action_replays=checked, runtime_action_max_abs_difference=replay_max, policy_tensor_sha256=before,
        policy_tensors_unchanged=True, policy_updates=0, gradient_probe=gradient, aggregate=aggregate, per_motion=per_motion,
        diagnostic_pair_bytes=pair_path.stat().st_size, diagnostic_pair_sha256=sha256(pair_path),
        backup_status="LOCAL_ONLY", selected_long_training_method=False,
        cuda_metadata=dict(torch_version=torch.__version__, allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                           float32_matmul_precision=torch.get_float32_matmul_precision()))
    atomic_json(OUTPUT / "result.json", result)
    atomic_json(ROOT / "data/manifests/a3_controller_reference_20261010_E19.json", result)
    print(json.dumps(dict(aggregate=aggregate, gradient=gradient), indent=2), flush=True)
    return 0


def math_sqrt(value):
    return float(np.sqrt(value))


if __name__ == "__main__":
    raise SystemExit(main())
