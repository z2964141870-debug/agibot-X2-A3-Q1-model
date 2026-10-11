"""E20 CPU artifact, export, train-only data and teacher-label verification."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.controller_adapter_trial import OUTPUT
from script.a3.controller_reference_probe import tensor_hash
from script.a3.corrected_trial import OFFICIAL
from script.a3.fullchain_support import DATA
from script.a3.joint_forecast_replay import forecast_window, load_predictor
from script.a3.reference_contract import load_sim


def main():
    registration = json.loads((OUTPUT / "job.json").read_text())
    split = json.loads((DATA / "split.json").read_text())
    dataset = OUTPUT / "training_windows.npz"
    metadata = json.loads((OUTPUT / "training_windows.json").read_text())
    if sha256(dataset) != metadata["sha256"] or dataset.stat().st_size != metadata["bytes"]:
        raise ValueError("Dataset identity changed")
    if metadata["heldout_examples_used"] != 0 or set(s["name"] for s in metadata["sources"]) != set(s["name"] for s in split["train"]):
        raise ValueError("Dataset train split violated")
    if set(s["name"] for s in split["train"]) & set(s["name"] for s in split["heldout"]):
        raise ValueError("Source split overlaps")
    sim = load_sim()
    policy = sim.A3Policy(OFFICIAL, torch.device("cpu"), encoder="a3_fast")
    policy.requires_grad_(False)
    if tensor_hash(policy) != metadata["teacher_policy_tensor_sha256"]:
        raise ValueError("Official frozen tensor identity changed")
    with np.load(dataset, allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    if not arrays["training_allowed"].item() or any(not np.isfinite(v).all() for v in arrays.values() if v.dtype.kind in "fiu"):
        raise ValueError("Dataset finite/usage contract failed")
    mean, std, permutation = arrays["mean"], arrays["std"], arrays["permutation"]
    indices_for_teacher, checked, cursor = [], 0, 0
    for source in metadata["sources"]:
        raw = next(s for s in split["train"] if s["name"] == source["name"])
        if sha256(Path(raw["path"])) != raw["sha256"]:
            raise ValueError("Training source identity changed")
        _, quaternion, q, _ = sim.load_a3_flat_csv(Path(raw["path"]), source_fps=30, frame_stride=4)
        observed = q+np.random.default_rng(0).normal(0, source["noise_std_rad"], q.shape)
        native_indices = np.arange(5, len(q)-7)
        if len(native_indices) != source["windows"]:
            raise ValueError("Training window count changed")
        sample_times = np.arange(len(q))/30
        selected = sorted(set([0, len(native_indices)//2, len(native_indices)-1]))
        for local in selected:
            row = cursor+local
            native = int(native_indices[local])
            requested = (native*5//3+np.arange(11))/50
            clean = np.column_stack([np.interp(requested, sample_times, q[:, j]) for j in range(29)])
            cv = np.column_stack([np.interp(requested, np.r_[sample_times[native-1:native], native/30+np.arange(11)/50],
                np.r_[observed[native-1, j], observed[native, j]+np.arange(11)/50*(observed[native, j]-observed[native-1, j])*30])
                for j in range(29)])
            x = (observed[native-5:native]-observed[native]).reshape(145).astype(np.float32)
            np.testing.assert_allclose(arrays["x"][row], (x-mean)/std, atol=2e-5, rtol=2e-6)
            np.testing.assert_allclose(arrays["baseline"][row], cv, atol=1e-6, rtol=2e-6)
            np.testing.assert_allclose(arrays["target"][row], clean, atol=1e-6, rtol=2e-6)
            seen = Rotation.from_quat(quaternion[native-1:native+1, [1, 2, 3, 0]])
            requested_ori = requested[:10]
            past = requested_ori <= sample_times[native]+1e-10
            predicted = np.empty((10, 4))
            if past.any():
                predicted[past] = Slerp(sample_times[native-1:native+1], seen)(np.minimum(requested_ori[past], sample_times[native])).as_quat()[:, [3, 0, 1, 2]]
            if (~past).any():
                rate = (seen[-1]*seen[-2].inv()).as_rotvec()*30
                predicted[~past] = (Rotation.from_rotvec((requested_ori[~past]-sample_times[native])[:, None]*rate)*seen[-1]).as_quat()[:, [3, 0, 1, 2]]
            orientation = sim.root_ori_diff_6d(predicted[0], predicted)
            np.testing.assert_allclose(arrays["orientation"][row], orientation, atol=1e-6, rtol=1e-6)
            command = np.r_[clean[:10, permutation].reshape(-1), (np.diff(clean[:, permutation], axis=0)*50).reshape(-1)].reshape(10, 58)
            teacher = np.c_[command, orientation].reshape(640)
            np.testing.assert_allclose(arrays["target_encoder"][row], teacher, atol=2e-6, rtol=2e-6)
            indices_for_teacher.append(row)
            checked += 1
        cursor += len(native_indices)
    if cursor != len(arrays["x"]):
        raise ValueError("Dataset source ledger incomplete")
    with torch.no_grad():
        actual = policy.encoder(torch.as_tensor(arrays["target_encoder"][indices_for_teacher])).numpy()
    np.testing.assert_allclose(actual, arrays["teacher_latent"][indices_for_teacher], atol=2e-6, rtol=1e-4)
    rng = np.random.default_rng(42)
    sample_ids = rng.choice(len(arrays["x"]), min(100, len(arrays["x"])), replace=False)
    checkpoint_rows = []
    for objective in registration["objectives"]:
        expected_steps = [0, 2]+list(range(10, 201, 10))
        for step in expected_steps:
            path = OUTPUT / objective / f"adapter_step_{step:06d}.pt"
            sidecar = json.loads(path.with_suffix(".json").read_text())
            if sha256(path) != sidecar["sha256"] or path.stat().st_size != sidecar["bytes"]:
                raise ValueError("Checkpoint SHA/size mismatch")
            saved = torch.load(path, map_location="cpu", weights_only=False)
            tensors = list(saved["adapter_state_dict"].values())
            tensors += [v for state in saved["optimizer_state_dict"]["state"].values() for v in state.values() if isinstance(v, torch.Tensor)]
            if any(not torch.isfinite(t).all() for t in tensors):
                raise ValueError("Nonfinite saved model/optimizer")
            if (saved["dataset_sha256"] != metadata["sha256"] or saved["objective"] != objective or
                    saved["base_policy_tensor_sha256"] != metadata["teacher_policy_tensor_sha256"] or saved["step"] != step):
                raise ValueError("Saved checkpoint experiment contract mismatch")
            exported = path.with_suffix(".npz")
            if sha256(exported) != sidecar["export_sha256"]:
                raise ValueError("Export SHA mismatch")
            model = load_predictor(exported)
            x = arrays["x"][sample_ids]
            cpu = torch.nn.functional.linear(torch.as_tensor(x), saved["adapter_state_dict"]["weight"], saved["adapter_state_dict"]["bias"]).detach().numpy()
            numpy = x.astype(float) @ model["weights"]+model["bias"]
            np.testing.assert_allclose(cpu, numpy, atol=2e-6, rtol=1e-4)
            if step == 0 and any(torch.count_nonzero(t).item() for t in saved["adapter_state_dict"].values()):
                raise ValueError("Initial adapter is not zero")
            checkpoint_rows.append(dict(objective=objective, step=step, bytes=path.stat().st_size,
                sha256=sidecar["sha256"], export_sha256=sidecar["export_sha256"],
                model_optimizer_cpu_finite=True, export_max_abs_difference=float(np.abs(cpu-numpy).max())))
        receipt = json.loads((OUTPUT / objective / "training_receipt.json").read_text())
        if receipt["step"] != 200 or receipt["official_policy_updates"] != 0:
            raise ValueError("Training receipt does not match bounded budget")
    result = dict(status="passed", dataset_sha256=metadata["sha256"], train_sources=16, heldout_training_examples=0,
        source_windows_independently_checked=checked, cpu_teacher_labels_checked=len(indices_for_teacher),
        checkpoints=checkpoint_rows, checkpoint_count=len(checkpoint_rows), actual_training_windows=len(arrays["x"]),
        base_policy_updates=0, source_anchor_distribution="CANONICAL_REFERENCE_NOT_ROLLOUT_STATE",
        backup_status="LOCAL_ONLY")
    atomic_json(OUTPUT / "independent_verification.json", result)
    print(json.dumps({k: result[k] for k in ("status", "checkpoint_count", "actual_training_windows", "cpu_teacher_labels_checked")}), flush=True)


if __name__ == "__main__":
    main()
