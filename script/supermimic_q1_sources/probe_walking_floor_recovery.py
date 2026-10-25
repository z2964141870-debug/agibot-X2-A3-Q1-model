#!/usr/bin/env python3
"""P15: require signed-distance recovery from the P14 penetrated floor poses."""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys

import probe_walking_root_offset as engine


def restoring_upper_bound(distance, clearance, gain, dt, relaxation=0.):
    if (not all(math.isfinite(x) for x in (distance, clearance, gain, dt, relaxation))
            or dt <= 0 or gain <= 0 or clearance < 0 or relaxation != 0):
        raise ValueError("Finite signed distance, positive gain/dt and zero relaxation required")
    return gain * (distance - clearance) / dt


def configure_floor(model, height, mj):
    floor = model.geom("floor").id
    if (not math.isfinite(height) or model.geom_type[floor] != mj.mjtGeom.mjGEOM_PLANE
            or model.geom_bodyid[floor] != 0):
        raise ValueError("A finite world-plane floor height is required")
    model.geom_pos[floor, 2] = height
    # Runtime position changes invalidate the compiled same-frame shortcut.
    model.geom_sameframe[floor] = mj.mjtSameFrame.mjSAMEFRAME_NONE


class RecoveringFloorLimit:
    """Reuse Mink's signed normal Jacobian, including a restoring bound below clearance."""

    def __init__(self, native, mj):
        self.native, self.mj = native, mj
        self.geom_id_pairs = native.geom_id_pairs
        self.calls, self.restoring_rows = 0, 0

    @property
    def gain(self):
        return self.native.gain

    @gain.setter
    def gain(self, value):
        self.native.gain = value

    def compute_qp_inequalities(self, configuration, dt):
        floor = self.native.model.geom("floor").id
        if max(abs(configuration.data.geom_xpos[floor] - self.native.model.geom_pos[floor])) > 1e-12:
            raise ValueError("The effective floor position does not match its configured world position")
        constraint = self.native.compute_qp_inequalities(configuration, dt)
        h = constraint.h.copy()
        for index, (first, second) in enumerate(self.geom_id_pairs):
            if not math.isfinite(float(h[index])):
                continue
            distance = self.mj.mj_geomDistance(self.native.model, configuration.data, first, second,
                                              self.native.collision_detection_distance, None)
            if distance <= self.native.minimum_distance_from_collisions:
                h[index] = restoring_upper_bound(distance, self.native.minimum_distance_from_collisions,
                    self.native.gain, dt, self.native.bound_relaxation)
                self.restoring_rows += 1
        self.calls += 1
        return type(constraint)(G=constraint.G, h=h)


def worker(args):
    import mujoco as mj
    import probe_walking_contact_floor as floor_engine

    original_floor, original_save = floor_engine.floor_limit, engine.save_json
    fixed = [engine.identity(__file__), engine.identity(Path(__file__).with_name("test_walking_floor_recovery.py"))]
    limits = []

    def recovering_floor(mink, model, dt):
        configure_floor(model, float(model.geom_pos[model.geom("floor").id, 2]), mj)
        limit = RecoveringFloorLimit(original_floor(mink, model, dt), mj)
        limits.append(limit)
        return limit

    def save_effective(path, value):
        if any(engine.identity(row["path"]) != row for row in fixed):
            raise ValueError("The P15 recovery adapter or its controls changed")
        value = dict(value)
        value["effective_floor_adapter"] = {
            "protocol": "P15_SIGNED_FLOOR_DISTANCE_RECOVERY",
            "source": fixed[0], "controls": fixed[1],
            "runtime_floor_hook_verified": floor_engine.floor_limit is recovering_floor,
            "floor_limits_created": len(limits), "inequality_calls": sum(limit.calls for limit in limits),
            "restoring_rows": sum(limit.restoring_rows for limit in limits),
            "clearance_m": .0005, "displacement_recovery_fraction": .85,
            "floor_sameframe": "mjSAMEFRAME_NONE after runtime geom_pos change",
            "effective_floor_position_checked_at_each_qp": True,
            "time_scaling": "gain=0.85*dt, h=gain*(distance-clearance)/dt for both signs",
            "root_offset_and_targets": "Identical to P14; initialization still uses its static 32-step fit",
            "limits": "Compared with P14, both the effective floor coordinate and the restoring bound are corrected. Linear recovery can be infeasible under fixed root, position and speed limits. Never relax those limits or admit a failed or penetrated result."}
        if "frozen_inputs" in value:
            value["frozen_inputs"] = [*value["frozen_inputs"], *fixed]
        original_save(path, value)

    floor_engine.floor_limit, engine.save_json = recovering_floor, save_effective
    try:
        engine.worker(args)
    finally:
        floor_engine.floor_limit, engine.save_json = original_floor, original_save


def main(args):
    args.root, args.output = args.root.resolve(strict=True), args.output.resolve()
    if args.worker:
        worker(args)
        return 0
    if args.output.exists() or not args.output.is_relative_to(args.root / "data/supermimic_for_q1"):
        raise ValueError("Use a new P15 diagnostic directory")
    budget = engine.budget_for(args)
    budget.memory()
    args.output.mkdir()
    os.nice(10)
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--root", str(args.root),
        "--output", str(args.output), "--expected-oom-kill", str(args.expected_oom_kill), "--worker"]
    result = engine.guarded_run(command, args.root, budget, args.output / "guard.json",
        args.root / "logs" / (args.output.name + ".log"), check_interval_s=.25, grace_s=5.,
        stop_requested=lambda: stopped, progress_path=args.output / "probe", maximum_idle_s=900.)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "WORKER_COMPLETED" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    sys.exit(main(parser.parse_args()))
