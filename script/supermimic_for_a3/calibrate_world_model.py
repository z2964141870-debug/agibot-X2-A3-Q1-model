#!/usr/bin/env python3
"""Collect independent frozen-policy replay sets, then fit and validate only the world model."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from supermimic.a3_assets import A3Assets, TRACKED, identity
from supermimic.a3_sim import A3Sim, SimCfg
from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.policy.nets import Policy, PolicyCfg
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg
from supermimic.world_model.wm_trainer import WorldModelTrainer, WMTrainerCfg
from train import atomic_json, health


REPLAY_FIELDS = ("s", "a", "a_exec", "xi", "r", "done", "frame", "clip", "ep_id",
                 "terminal", "ep_start", "valid", "ptr", "ep_counter")


def save_pt(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


@torch.no_grad()
def validation_metrics(wm, windows):
    rows = {}
    for horizon, window in windows.items():
        learned, prior = window.states[0], window.states[0]
        for step in range(horizon):
            learned = wm.step(learned, window.a_exec[step], window.xi)
            prior = wm.spec.integrate(prior, torch.zeros_like(learned[:, :82]))
        target = window.states[-1]
        rows[str(horizon)] = {}
        for label, prediction in (("learned", learned), ("constant_velocity_prior", prior)):
            if not torch.isfinite(prediction).all():
                raise ValueError("Nonfinite validation prediction")
            root = prediction[:, :3] - target[:, :3]
            joints = (prediction[:, 7:42] - target[:, 7:42])[:, wm.heads.active_indices]
            rows[str(horizon)][label] = {
                "normalized_state_mse": float(wm.spec.normalized_sq_error(prediction, target).mean()),
                "root_position_rmse_m": float(root.square().sum(-1).mean().sqrt()),
                "root_height_rmse_m": float(root[:, 2].square().mean().sqrt()),
                "active_joint_rmse_rad": float(joints.square().mean().sqrt())}
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--collection-steps", type=int, default=1024)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--seed", type=int, default=4911)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.collection_steps < 64 or args.updates < 1:
        raise ValueError("Collection must admit 32-step windows and fitting must update parameters")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    source_identity = identity(args.checkpoint)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    manifest = ck["manifest"]
    backend_path = Path(manifest["backend_receipt"]["path"])
    if identity(backend_path) != manifest["backend_receipt"]:
        raise ValueError("Source backend receipt changed")
    for source in json.loads(backend_path.read_text())["source_files"]:
        if identity(source["path"])["sha256"] != source["sha256"]:
            raise ValueError("Verified backend source changed")
    assets = A3Assets(args.root)
    if assets.manifest() != manifest["assets"]:
        raise ValueError("Source asset/control contract changed")
    files = [Path(row["path"]) for row in manifest["train_clips"]]
    for path, row in zip(files, manifest["train_clips"]):
        if identity(path) != row:
            raise ValueError("Training reference changed")
    heads = assets.heads(args.device)
    lib = MotionLibrary(files, TRACKED, assets.q0, device=args.device, active_indices=assets.active_indices)
    policy = Policy(PolicyCfg(**manifest["policy"])).to(args.device).eval()
    policy.load_state_dict(ck["policy"], strict=True)
    policy.requires_grad_(False)
    frozen_policy = {name: value.clone() for name, value in policy.state_dict().items()}
    wcfg = dict(manifest["world_model"])
    wcfg["state_spec"] = StateSpecCfg(**wcfg["state_spec"])
    wm = WorldModel(WorldModelCfg(**wcfg), heads, args.device)
    wm.load_state_dict(ck["wm"], strict=True)
    cfg = WMTrainerCfg(batch_size=128, batches_per_update=10, compile="default" if args.compile else None)
    trainer = WorldModelTrainer(wm, lib, cfg)
    trainer.opt.load_state_dict(ck["wm_opt"])
    initial_optimizer_steps = sorted({int(v["step"]) for v in trainer.opt.state.values()})
    initial_wm = {name: value.clone() for name, value in wm.state_dict().items()}
    admission = {"status": "ADMITTED_NOT_COMPLETED", "source_checkpoint": source_identity,
                 "source_training_step": ck["step"], "source_policy_updates": ck["policy_updates"],
                 "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                 "assets": assets.manifest(), "training_clips": manifest["train_clips"],
                 "wm_trainer": asdict(cfg), "initial_optimizer_steps": initial_optimizer_steps,
                 "entry": identity(__file__), "backend_receipt": identity(backend_path),
                 "protocol": "Frozen R02 policy; separate collection seeds and episodes on training clips only; fixed validation windows; no actor/critic update.",
                 "decision_rule": "Final 32-step root RMSE <= 0.8*initial; 1-step state MSE and 32-step active-joint RMSE <= 1.1*initial. All must pass.",
                 "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    atomic_json(args.output / "admission.json", admission)
    del ck
    started = time.monotonic()
    status, error = "FAILED", None
    collected = updates = 0
    stop_requested = False
    baseline = final = None
    counts = {}

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def progress(phase, **values):
        row = {"status": "RUNNING", "phase": phase, "pid": os.getpid(),
               "physics_env_steps": collected, "wm_updates": updates,
               "policy_updates": 0, "seconds": time.monotonic() - started, "health": health(), **values}
        atomic_json(args.output / "status.json", row)
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    try:
        progress("collection_preflight")
        replays = {}
        for offset, split in enumerate(("fitting", "validation")):
            seed = args.seed + offset
            torch.manual_seed(seed)
            np.random.seed(seed)
            replay = ReplayBuffer(args.num_envs, 2 * args.collection_steps + 1, 83, 29, 33, args.device)
            sim = A3Sim(assets, lib, SimCfg(num_envs=args.num_envs, threads=args.threads), heads, seed)
            try:
                with torch.no_grad():
                    for step in range(1, args.collection_steps + 1):
                        obs, goal = sim.policy_inputs()
                        sim.step(policy(obs, goal), replay)
                        collected += args.num_envs
                        if step % 128 == 0:
                            progress("collection_" + split, collection_step=step)
                        if stop_requested:
                            raise InterruptedError("Stop requested")
            finally:
                sim.close()
                del sim
            transitions = replay.valid & ~replay.terminal
            count = torch.bincount(replay.clip[transitions], minlength=lib.num_clips)
            counts[split] = {files[i].stem: int(n) for i, n in enumerate(count)}
            replay_path = args.output / (split + "_replay.pt")
            save_pt(replay_path, {"seed": seed, "E": replay.E, "T": replay.T,
                                 "total_rows": replay.total_rows,
                                 "fields": {name: getattr(replay, name).cpu() for name in REPLAY_FIELDS},
                                 "source_checkpoint": source_identity, "train_clips": manifest["train_clips"]})
            replays[split] = replay
        if any(n == 0 for split in counts.values() for n in split.values()):
            raise ValueError("A training clip was absent from one collection split")
        torch.manual_seed(args.seed + 2)
        windows = {h: replays["validation"].sample_windows(256, h) for h in (1, 8, 32)}
        save_pt(args.output / "validation_windows.pt", {h: {k: v.cpu() for k, v in vars(w).items()}
                                                      for h, w in windows.items()})
        baseline = validation_metrics(wm, windows)
        atomic_json(args.output / "validation_initial.json", baseline)
        progress("validation_initial", validation=baseline, clip_transition_counts=counts)
        for updates in range(1, args.updates + 1):
            metrics = trainer.update(replays["fitting"])
            if not all(np.isfinite(value) for value in metrics.values()):
                raise ValueError("Nonfinite model fitting metrics")
            if updates % 100 == 0:
                progress("world_model_fitting", **metrics)
            if stop_requested:
                raise InterruptedError("Stop requested")
        final = validation_metrics(wm, windows)
        policy_unchanged = all(torch.equal(value, frozen_policy[name]) for name, value in policy.state_dict().items())
        changed = [name for name, value in wm.state_dict().items() if not torch.equal(value, initial_wm[name])]
        optimizer_steps = sorted({int(v["step"]) for v in trainer.opt.state.values()})
        expected_steps = [value + args.updates * cfg.batches_per_update for value in initial_optimizer_steps]
        if not policy_unchanged or not changed or optimizer_steps != expected_steps:
            raise ValueError("Frozen-policy or model-update contract failed")
        if not all(torch.isfinite(value).all() for value in wm.state_dict().values()):
            raise ValueError("Nonfinite fitted model parameters")
        if not all(torch.isfinite(value).all() for state in trainer.opt.state.values()
                   for value in state.values() if torch.is_tensor(value)):
            raise ValueError("Nonfinite fitted model optimizer state")
        decision = {
            "root_32step_improved_20pct": final["32"]["learned"]["root_position_rmse_m"] <= .8 * baseline["32"]["learned"]["root_position_rmse_m"],
            "one_step_state_regression_below_10pct": final["1"]["learned"]["normalized_state_mse"] <= 1.1 * baseline["1"]["learned"]["normalized_state_mse"],
            "joint_32step_regression_below_10pct": final["32"]["learned"]["active_joint_rmse_rad"] <= 1.1 * baseline["32"]["learned"]["active_joint_rmse_rad"]}
        save_pt(args.output / "calibrated_world_model.pt", {"wm": wm.state_dict(), "wm_opt": trainer.opt.state_dict(),
                                                           "source_checkpoint": source_identity,
                                                           "wm_config": manifest["world_model"], "admission": admission,
                                                           "additional_wm_updates": updates})
        status = "WORLD_MODEL_CALIBRATION_COMPLETED"
        progress("completed", validation=final, decision=decision, passed_for_candidate=all(decision.values()))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt = {"status": status, "error": error, "physics_env_steps": collected,
                   "wm_updates": updates, "policy_updates": 0, "critic_updates": 0,
                   "seconds": time.monotonic() - started, "source_checkpoint": source_identity,
                   "admission": identity(args.output / "admission.json"), "clip_transition_counts": counts,
                   "initial_validation": baseline, "final_validation": final,
                   "policy_unchanged": locals().get("policy_unchanged"),
                   "changed_wm_tensors": locals().get("changed"), "optimizer_steps": locals().get("optimizer_steps"),
                   "decision": locals().get("decision"), "passed_for_candidate": all(locals().get("decision", {"incomplete": False}).values()),
                   "artifacts": [identity(path) for path in sorted(args.output.glob("*.pt"))],
                   "storage_status": "LOCAL_ONLY", "not_a_policy_effect_result": True}
        atomic_json(args.output / "receipt.json", receipt)
        atomic_json(args.output / "status.json", receipt)
        print(json.dumps(receipt, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
