"""E20 paired short supervised adapters; official policy parameters are frozen."""

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from script.a3.causal_orientation import orientation_window
from script.a3.checkpoint_store import atomic_json, fsync_directory, sha256
from script.a3.controller_reference_probe import tensor_hash
from script.a3.corrected_trial import OFFICIAL
from script.a3.fullchain_support import DATA, ROOT
from script.a3.reference_contract import load_sim


EXPERIMENT = os.environ.get("A3_ADAPTER_EXPERIMENT", "E20")
if EXPERIMENT not in ("E20", "E21"):
    raise ValueError("Only preregistered E20 GPU or E21 CPU experiments are supported")
OUTPUT = ROOT / f"data/experiments/a3_controller_adapter_20261010_{EXPERIMENT}"
DEVICE = torch.device("cpu" if EXPERIMENT == "E21" else "cuda")
LR = 1e-4


def time_matrix(phase):
    support = np.concatenate(([-1/30], np.arange(11)/50))
    query = phase+np.arange(11)/50
    upper = np.searchsorted(support, query-1e-10)
    lower = upper-1
    fraction = (query-support[lower])/(support[upper]-support[lower])
    matrix = np.zeros((11, 12))
    matrix[np.arange(11), lower] = 1-fraction
    matrix[np.arange(11), upper] += fraction
    return matrix


def make_policy():
    sim = load_sim()
    policy = sim.A3Policy(OFFICIAL, DEVICE, encoder="a3_fast")
    policy.requires_grad_(False)
    policy.eval()
    return sim, policy


def prepare():
    path = OUTPUT / "training_windows.npz"
    if path.exists():
        registered = json.loads((OUTPUT / "training_windows.json").read_text())
        if sha256(path) != registered["sha256"]:
            raise ValueError("Training data changed")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    split = json.loads((DATA / "split.json").read_text())
    sim, policy = make_policy()
    before = tensor_hash(policy)
    columns = {key: [] for key in ("x", "baseline", "future_matrix", "target", "orientation", "target_encoder")}
    sources = []
    for source in split["train"]:
        source_path = Path(source["path"])
        if sha256(source_path) != source["sha256"]:
            raise ValueError("Training source SHA changed")
        _, rotation, q, _ = sim.load_a3_flat_csv(source_path, source_fps=30, frame_stride=4)
        # The adapter is fitted in CSV/MJ29 order; encoder commands use URDF/IL order.
        import mujoco
        from script.a3.fullchain_support import VENDOR
        if not sources:
            asset = VENDOR / "gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml"
            permutation = sim.build_loop_runtime(mujoco.MjModel.from_xml_path(str(asset)), sim.DEFAULT_URDF).mapping.mj29_to_il
        indices = np.arange(5, len(q)-7)
        times = np.arange(len(q))/30
        phases = indices*5//3/50-indices/30
        matrices = np.asarray([time_matrix(phase) for phase in phases])
        query = indices[:, None]*5//3/50+np.arange(11)/50
        target = np.stack([np.interp(query, times, q[:, j]) for j in range(29)], axis=-1)
        orientations = []
        for i, requested in zip(indices, query):
            predicted = orientation_window(rotation[i-1:i+1], times[i-1:i+1], requested[:10])
            orientations.append(sim.root_ori_diff_6d(predicted[0], predicted))
        orientations = np.asarray(orientations)
        teacher_q = target[:, :10, permutation]
        teacher_v = np.diff(target[:, :, permutation], axis=1)*50
        teacher_command = np.concatenate((teacher_q.reshape(len(indices), 290), teacher_v.reshape(len(indices), 290)), axis=1).reshape(-1, 10, 58)
        teacher_encoder = np.concatenate((teacher_command, orientations), axis=-1).reshape(-1, 640)
        for sigma in (0., .01):
            observed = q+np.random.default_rng(0).normal(0, sigma, q.shape)
            current = observed[indices]
            history = observed[indices[:, None]+np.arange(-5, 1)]
            x = (history[:, :-1]-current[:, None]).reshape(-1, 145)
            slope = (current-observed[indices-1])*30
            future = current[:, None]+np.arange(1, 11)[None, :, None]/50*slope[:, None]
            nodes = np.concatenate((observed[indices-1, None], current[:, None], future), axis=1)
            baseline = np.einsum("nht,ntj->nhj", matrices, nodes)
            values = dict(x=x, baseline=baseline, future_matrix=matrices[:, :, 2:], target=target,
                          orientation=orientations, target_encoder=teacher_encoder)
            for key, value in values.items():
                columns[key].append(value)
            sources.append(dict(name=source["name"], source_sha256=source["sha256"], noise_std_rad=sigma,
                                windows=len(indices), first_native_index=5, last_native_index=int(indices[-1])))
        print(json.dumps(dict(prepared=source["name"], windows=len(indices)*2)), flush=True)
    arrays = {key: np.concatenate(value).astype(np.float32) for key, value in columns.items()}
    mean = arrays["x"].mean(0)
    std = arrays["x"].std(0)
    std = np.where(std < 1e-8, 1, std)
    arrays["x"] = (arrays["x"]-mean)/std
    latent = []
    with torch.no_grad():
        for start in range(0, len(arrays["x"]), 256):
            value = torch.as_tensor(arrays["target_encoder"][start:start+256], device=DEVICE)
            latent.append(policy.encoder(value).cpu().numpy())
    arrays["teacher_latent"] = np.concatenate(latent)
    if any(not np.isfinite(v).all() for v in arrays.values()) or tensor_hash(policy) != before:
        raise ValueError("Training data or frozen teacher contract failed")
    with tempfile.NamedTemporaryFile(dir=OUTPUT, suffix=".incomplete", delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays, mean=mean, std=std, permutation=permutation,
                            training_allowed=True, usage=np.asarray("REGISTERED_16_OFFICIAL_TRAIN_MOTIONS_ONLY"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(OUTPUT)
    with np.load(path, allow_pickle=False) as stored:
        if stored["x"].shape[1] != 145 or stored["target"].shape[1:] != (11, 29):
            raise ValueError("Training dataset reload shape mismatch")
    atomic_json(OUTPUT / "training_windows.json", dict(sha256=sha256(path), bytes=path.stat().st_size,
        windows=len(arrays["x"]), sources=sources, source_split_sha256=sha256(DATA / "split.json"),
        training_motions=16, heldout_examples_used=0, teacher_policy_tensor_sha256=before,
        anchor_assumption="CANONICAL_REFERENCE_PELVIS_AT_ARRIVED_TARGET_NOT_ROLLOUT_STATE",
        teacher_orientation="SAME_CAUSAL_ORIENTATION_AS_STUDENT; ONLY_FUTURE_JOINT_LABELS_DIFFER",
        backup_status="LOCAL_ONLY"))


def finite_checkpoint(payload):
    for tensor in payload["adapter_state_dict"].values():
        if not torch.isfinite(tensor).all():
            raise ValueError("Nonfinite adapter")
    for state in payload["optimizer_state_dict"]["state"].values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError("Nonfinite supervised optimizer")
    if not all(math.isfinite(row["loss"]) and math.isfinite(row["gradient_norm"]) for row in payload["losses"]):
        raise ValueError("Nonfinite objective history")


def save(directory, payload, mean, std):
    finite_checkpoint(payload)
    step = payload["step"]
    path = directory / f"adapter_step_{step:06d}.pt"
    if path.exists():
        raise FileExistsError("Existing adapter checkpoint must not be overwritten")
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".incomplete", delete=False) as stream:
        temporary = Path(stream.name)
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(directory)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    finite_checkpoint(loaded)
    if any(not torch.equal(loaded["adapter_state_dict"][k], v.cpu()) for k, v in payload["adapter_state_dict"].items()):
        raise ValueError("Adapter CPU reload differs")
    exported = directory / f"adapter_step_{step:06d}.npz"
    np.savez_compressed(exported, mean=mean, std=std,
        weights=loaded["adapter_state_dict"]["weight"].numpy().T,
        bias=loaded["adapter_state_dict"]["bias"].numpy(), horizons_s=np.arange(1, 11)/50,
        history_frames=6, source_fps=30, step=step, usage=np.asarray("CV_RESIDUAL_DIAGNOSTIC_NOT_ACCEPTED"))
    with exported.open("rb") as stream:
        os.fsync(stream.fileno())
    fsync_directory(directory)
    atomic_json(path.with_suffix(".json"), dict(step=step, bytes=path.stat().st_size, sha256=sha256(path),
        model_and_optimizer_finite=True, cpu_reload_exact=True, export_sha256=sha256(exported),
        source_dataset_sha256=payload["dataset_sha256"], objective=payload["objective"], backup_status="LOCAL_ONLY"))


def train(objective, limit):
    directory = OUTPUT / objective
    directory.mkdir(parents=True, exist_ok=True)
    dataset = OUTPUT / "training_windows.npz"
    dataset_sha = sha256(dataset)
    with np.load(dataset, allow_pickle=False) as stored:
        if not stored["training_allowed"].item() or not all(np.isfinite(stored[key]).all() for key in
                ("x", "baseline", "future_matrix", "target", "orientation", "teacher_latent")):
            raise ValueError("Invalid registered training data")
        tensors = {key: torch.as_tensor(stored[key].copy(), device=DEVICE) for key in
                   ("x", "baseline", "future_matrix", "target", "orientation", "teacher_latent")}
        mean, std, permutation = stored["mean"].copy(), stored["std"].copy(), stored["permutation"].copy()
    sim, policy = make_policy()
    frozen_sha = tensor_hash(policy)
    adapter = torch.nn.Linear(145, 290, device=DEVICE)
    torch.nn.init.zeros_(adapter.weight)
    torch.nn.init.zeros_(adapter.bias)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=0.)
    generator = torch.Generator().manual_seed(0)
    saved = sorted(directory.glob("adapter_step_*.json"))
    step, losses = 0, []
    if saved:
        sidecar = json.loads(saved[-1].read_text())
        path = saved[-1].with_suffix(".pt")
        if sha256(path) != sidecar["sha256"] or path.stat().st_size != sidecar["bytes"]:
            raise ValueError("Adapter resume SHA mismatch")
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        finite_checkpoint(loaded)
        if (loaded["dataset_sha256"] != dataset_sha or loaded["objective"] != objective or
                loaded["base_policy_tensor_sha256"] != frozen_sha):
            raise ValueError("Resume belongs to another objective or dataset")
        adapter.load_state_dict(loaded["adapter_state_dict"])
        optimizer.load_state_dict(loaded["optimizer_state_dict"])
        generator.set_state(loaded["generator_state"])
        step, losses = loaded["step"], loaded["losses"]
    def checkpoint():
        save(directory, dict(adapter_state_dict=adapter.state_dict(), optimizer_state_dict=optimizer.state_dict(),
            generator_state=generator.get_state(), step=step, losses=losses, dataset_sha256=dataset_sha,
            objective=objective, base_policy_tensor_sha256=frozen_sha), mean, std)
    if not saved:
        checkpoint()
    while step < limit:
        ids = torch.randint(len(tensors["x"]), (256,), generator=generator).to(DEVICE)
        correction = adapter(tensors["x"][ids]).reshape(-1, 10, 29)
        predicted = tensors["baseline"][ids]+torch.einsum("bht,btj->bhj", tensors["future_matrix"][ids], correction)
        if objective == "mse":
            loss = (predicted-tensors["target"][ids]).square().mean()
        else:
            q = predicted[:, :10, permutation]
            v = torch.diff(predicted[:, :, permutation], dim=1)*50
            command = torch.cat((q.reshape(-1, 290), v.reshape(-1, 290)), dim=1).reshape(-1, 10, 58)
            encoded = torch.cat((command, tensors["orientation"][ids]), dim=-1).reshape(-1, 640)
            loss = (policy.encoder(encoded)-tensors["teacher_latent"][ids]).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite supervised loss")
        optimizer.zero_grad()
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.)
        if not torch.isfinite(gradient):
            raise ValueError("Nonfinite adapter gradient")
        optimizer.step()
        step += 1
        losses.append(dict(step=step, loss=float(loss.item()), gradient_norm=float(gradient.item())))
        if step == 2 or step % 10 == 0:
            checkpoint()
            print(json.dumps(dict(objective=objective, saved_step=step, loss=losses[-1]["loss"])), flush=True)
    if tensor_hash(policy) != frozen_sha:
        raise ValueError("Official policy changed during adapter training")
    atomic_json(directory / "training_receipt.json", dict(step=step, objective=objective,
        teacher_and_base_policy_frozen=True, base_policy_tensor_sha256=frozen_sha, official_policy_updates=0,
        adapter_optimizer_updates=step, dataset_sha256=dataset_sha, loss_first=losses[0]["loss"] if losses else None,
        loss_last=losses[-1]["loss"] if losses else None, scope="SUPERVISED_REFERENCE_ADAPTER_NOT_PPO_OR_EFFECT_ACCEPTANCE"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train"))
    parser.add_argument("--objective", choices=("mse", "latent"))
    parser.add_argument("--limit", type=int, choices=(2, 200), default=2)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        if args.objective is None:
            parser.error("Training requires an explicit objective")
        train(args.objective, args.limit)


if __name__ == "__main__":
    main()
