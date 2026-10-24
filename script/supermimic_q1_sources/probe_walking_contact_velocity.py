#!/usr/bin/env python3
"""P12B: retain the P12 engine and add a train-only stance velocity condition."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys

import numpy as np

import probe_walking_contact as engine
from walking_contact_targets import runs


SPEED_MAX_M_S = .15


def refine_stance(positions, height_mask):
    values, mask = np.asarray(positions, float), np.asarray(height_mask, bool).copy()
    if (values.ndim != 3 or values.shape[1:] != (2, 3) or len(values) < 5
            or mask.shape != values.shape[:2] or not np.isfinite(values).all()):
        raise ValueError("Finite 50Hz ankle positions and matching bilateral labels required")
    speed = np.linalg.norm(np.gradient(values[:, :, :2], .02, axis=0, edge_order=2), axis=-1)
    mask &= speed <= SPEED_MAX_M_S
    for side in range(2):
        for start, end in runs(mask[:, side]):
            if end - start < 4 and start > 0 and end < len(mask):
                mask[start:end, side] = False
    return mask


def worker(args):
    adapter_inputs = [engine.identity(__file__), engine.identity(Path(__file__).with_name("test_walking_contact_velocity.py"))]
    original_targets, original_save = engine.contact_targets, engine.save_json

    def velocity_targets(positions, rotations, mask, feet, floor, level_rotations):
        # The engine saves this diagnostic label array; source/task arrays remain untouched.
        mask[:] = refine_stance(positions, mask)
        return original_targets(positions, rotations, mask, feet, floor, level_rotations)

    def save_effective(path, value):
        if any(engine.identity(item["path"]) != item for item in adapter_inputs):
            raise ValueError("The P12B adapter changed during execution")
        value = dict(value)
        value["effective_stance_adapter"] = {"protocol": "P12B_HEIGHT_AND_XY_SPEED_AT_MOST_0P15",
            "xy_speed_max_m_s": SPEED_MAX_M_S, "velocity_method": "50Hz central gradient, second-order edges",
            "source": adapter_inputs[0], "controls": adapter_inputs[1],
            "entrypoint": velocity_targets.__qualname__, "base_producer": engine.identity(engine.__file__),
            "runtime_target_hook_verified": engine.contact_targets is velocity_targets,
            "limits": "Additional low-speed condition on inferred training stance, not contact ground truth."}
        if path.name == "selection.json":
            value["stance_definition"] += " Also require mapped ankle XY speed at most 0.15m/s and remove short interior runs again."
        if "frozen_inputs" in value:
            value["frozen_inputs"] = [*value["frozen_inputs"], *adapter_inputs]
        original_save(path, value)

    engine.contact_targets, engine.save_json = velocity_targets, save_effective
    try:
        engine.worker(args)
    finally:
        engine.contact_targets, engine.save_json = original_targets, original_save


def main(args):
    args.root, args.output = args.root.resolve(strict=True), args.output.resolve()
    if args.worker:
        worker(args)
        return 0
    if args.output.exists() or not args.output.is_relative_to(args.root / "data/supermimic_for_q1"):
        raise ValueError("Use a new P12B train diagnostic directory")
    budget = engine.MemoryBoundBudget(engine.StorageBudget(args.output,
        args.root / "data/references/phuma_source_2faf15b5", 100 * 1024 ** 3,
        128 * 1024 ** 2, 512 * 1024 ** 2), int(28.75 * 1024 ** 3), args.expected_oom_kill)
    budget.memory()
    args.output.mkdir()
    os.nice(10)
    stop = False

    def requested(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, requested)
    signal.signal(signal.SIGINT, requested)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--root", str(args.root),
        "--output", str(args.output), "--expected-oom-kill", str(args.expected_oom_kill), "--worker"]
    record = engine.guarded_run(command, args.root, budget, args.output / "guard.json",
        args.root / "logs" / (args.output.name + ".log"), check_interval_s=.25, grace_s=5.,
        stop_requested=lambda: stop, progress_path=args.output / "probe", maximum_idle_s=900.)
    print(json.dumps(record), flush=True)
    return 0 if record["status"] == "WORKER_COMPLETED" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    sys.exit(main(parser.parse_args()))
