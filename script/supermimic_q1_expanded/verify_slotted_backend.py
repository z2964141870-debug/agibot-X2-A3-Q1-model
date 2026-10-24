#!/usr/bin/env python3
"""Compare complete Q1 transitions/replay and verify the slotted backend's H32 graph."""

import argparse
import gc
import json
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
CORE = HERE.parent / "supermimic_for_q1"
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(HERE.parent / "supermimic_q1_phuma"))

import mujoco
import numpy as np
import torch

from contact_protocol import ContactCorrectedAssets
from scan_corpus import resource_check
from supermimic.data.replay_buffer import ReplayBuffer
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.q1_assets import TRACKED, identity
from supermimic.q1_sim import Q1Sim, SimCfg
from supermimic.robot_spec import ACT_DIM, STATE_DIM, XI_DIM
from verify_backend import check_graph

from slotted_sim import SlottedQ1Sim


def collect(kind, assets, library, environments, slots, steps, seed, report):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = SimCfg(num_envs=environments, threads=4, episode_steps=7)
    heads = assets.heads("cpu")
    sim = (Q1Sim(assets, library, cfg, heads, seed) if kind == "legacy" else
           SlottedQ1Sim(assets, library, cfg, heads, seed, slots=slots))
    replay = ReplayBuffer(environments, 128, STATE_DIM, ACT_DIM, XI_DIM, "cpu")
    arrays = {}
    resets = 0
    peak_memory = 0
    started = time.monotonic()
    try:
        for step in range(steps):
            if step % 11 == 0:
                sim.reset(sim.ids[::2])
            result = sim.step(torch.randn(environments, ACT_DIM) * .2, replay)
            report["physics_control_steps"] += environments
            if step % 8 == 0:
                peak_memory = max(peak_memory, int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text()))
            resets += int((sim.steps == 0).sum())
            values = {"post_" + key: getattr(sim, key) for key in (
                "state", "frame", "hist", "context", "previous_action", "steps", "ep_start", "bin_return", "bin_memory")}
            values.update({"result_" + key: value for key, value in result.items() if isinstance(value, torch.Tensor)})
            values.update({"reward_term_" + key: value for key, value in result["terms"].items()})
            for key, value in values.items():
                arrays.setdefault(key, []).append(value.detach().cpu().numpy().copy())
        arrays = {key: np.stack(value) for key, value in arrays.items()}
        arrays.update({"replay_" + key: value.detach().cpu().numpy().copy()
                       for key, value in vars(replay).items() if isinstance(value, torch.Tensor)})
        arrays["replay_total_rows"] = np.array(replay.total_rows)
        metadata = {"kind": kind, "environments": environments, "slots": slots if kind != "legacy" else environments,
                    "control_steps": steps * environments, "automatic_resets": resets,
                    "explicit_partial_resets": (steps + 10) // 11,
                    "unique_models": len({id(model) for model in sim.models}),
                    "model_buffer_bytes": sum(model.nbuffer for model in {id(m): m for m in sim.models}.values()),
                    "sampled_peak_cgroup_memory_bytes": peak_memory,
                    "seconds": time.monotonic() - started}
        return arrays, metadata, replay
    finally:
        sim.close()


def compare_arrays(left, right):
    if left.keys() != right.keys():
        raise AssertionError("Recorded transition/replay fields differ")
    errors = {}
    for key in left:
        a, b = left[key], right[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError("Transition/replay shape or dtype differs: " + key)
        if a.dtype.kind in "fc":
            error = float(np.max(np.abs(a - b)))
            if not np.isfinite(a).all() or not np.isfinite(b).all() or error > 1e-6:
                raise AssertionError(f"Slotted/legacy numerical mismatch {key}: {error}")
        else:
            error = int(np.count_nonzero(a != b))
            if error:
                raise AssertionError("Slotted/legacy discrete mismatch: " + key)
        errors[key] = error
    return errors


def verify(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if resource_check() != args.expected_oom_kill or mujoco.__version__ != "3.3.7":
        raise RuntimeError("Expected frozen MuJoCo3.3.7 and unchanged resource/OOM state")
    original = json.loads(args.baseline_backend.read_text())
    if original["status"] != "BACKEND_AND_GRADIENT_CONTRACT_PASSED":
        raise ValueError("The original corrected Q1 backend receipt is required")
    for record in original["source_files"]:
        if identity(record["path"]) != record:
            raise ValueError("A preserved backend source changed")
    args.output.mkdir(parents=True)
    report = {"status": "FAILED", "optimizer_updates": 0, "physics_control_steps": 0,
              "model_slots": args.slots, "baseline_backend": identity(args.baseline_backend),
              "storage_status": "LOCAL_ONLY", "hardware_control": False}
    try:
        torch.set_num_threads(2)
        assets = ContactCorrectedAssets(args.root)
        if assets.manifest() != original["assets"]:
            raise ValueError("Q1 assets differ from the corrected baseline")
        report["assets"] = assets.manifest()
        manifest_path = args.motions / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        accepted = set(manifest["qualified_clips"])
        training = [row for row in manifest["motions"] if row["clip"] in accepted and row["split"] == "train"]
        selected = ([row for row in training if row["bucket"] == "locomotion"][:2]
                    + [row for row in training if row["bucket"] != "locomotion"][:2])
        if len(selected) != 4:
            raise ValueError("Need two moving and two other accepted training references")
        files = [args.motions / (row["clip"] + ".npz") for row in selected]
        for path, row in zip(files, selected):
            if identity(path) != row["output"]:
                raise ValueError("Numerical comparison reference changed")
        report["references"] = [identity(path) for path in files]
        report["motion_manifest"] = identity(manifest_path)
        library = MotionLibrary(files, TRACKED, assets.q0, device="cpu", active_indices=assets.active_indices)
        report["cases"] = []
        for index, (environments, slots) in enumerate(((7, 3), (8, args.slots))):
            legacy, old_meta, replay = collect("legacy", assets, library, environments, slots, 96, 2026102409 + index, report)
            del replay
            gc.collect()
            actual, new_meta, replay = collect("slotted", assets, library, environments, slots, 96, 2026102409 + index, report)
            errors = compare_arrays(legacy, actual)
            if old_meta["automatic_resets"] < environments or old_meta["automatic_resets"] != new_meta["automatic_resets"]:
                raise AssertionError("Repeated automatic reset coverage is missing")
            np.savez_compressed(args.output / f"case{index}_legacy.npz", **legacy)
            np.savez_compressed(args.output / f"case{index}_slotted.npz", **actual)
            report["cases"].append({"legacy": old_meta, "slotted": new_meta, "max_abs_errors": errors,
                                    "arrays": [identity(args.output / f"case{index}_{name}.npz") for name in ("legacy", "slotted")]})
            if resource_check() != args.expected_oom_kill:
                raise RuntimeError("Resource/OOM state changed during backend comparison")
            del legacy, actual
            gc.collect()
        graph_library = MotionLibrary(files, TRACKED, assets.q0, device="cpu", dtype=torch.float64,
                                      active_indices=assets.active_indices)
        report["graph"] = check_graph(assets, graph_library, replay)
        del replay, graph_library
        gc.collect()
        probe, probe_meta, replay = collect("slotted", assets, library, 64, args.slots, 32, 2026102411, report)
        if (probe_meta["unique_models"] != min(64, args.slots)
                or not np.isfinite(probe["post_state"]).all()):
            raise AssertionError("64-environment model-slot resource probe failed")
        np.savez_compressed(args.output / "probe64.npz", **probe)
        report["resource_probe64"] = {**probe_meta, "arrays": identity(args.output / "probe64.npz"),
                                      "policy_training_executed": False, "device": "cpu"}
        report["versions"] = {"mujoco": mujoco.__version__, "torch": torch.__version__}
        report["oom_kill"] = resource_check()
        if report["oom_kill"] != args.expected_oom_kill:
            raise RuntimeError("OOM state changed during H32 graph verification")
        report["status"] = "BACKEND_AND_GRADIENT_CONTRACT_PASSED"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        files = {Path(row["path"]) for row in original["source_files"]}
        files.update((HERE / "slotted_sim.py", Path(__file__), args.baseline_backend))
        report["source_files"] = [identity(path) for path in sorted(files)]
        with (args.output / "receipt.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({k: v for k, v in report.items()
                          if k not in ("assets", "source_files", "traceback", "cases")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--baseline-backend", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    verify(parser.parse_args())
