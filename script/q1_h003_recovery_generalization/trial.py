"""Process-isolated closed-loop trials with a modified plant and unchanged tracker."""

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from protocol import TRAIN, TEST, TRAIN_SEED, TEST_SEED, TRAIN_IDS
from late_protocol import LATE_TRAIN, LATE_TEST, LATE_SEED
import bench
import relay
from run_suite import EXPERT, TRACKS, BUNDLE, verify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["train", "test", "parity", "train_late", "test_late"])
    ap.add_argument("condition")
    ap.add_argument("arm")
    ap.add_argument("kind", choices=["static", "relay"])
    args = ap.parse_args()
    condition = {
        "train": TRAIN,
        "parity": TRAIN,
        "test": TEST,
        "train_late": LATE_TRAIN,
        "test_late": LATE_TEST,
    }[args.split][args.condition]
    out = ROOT / "runs" / args.split / args.arm / args.condition / args.kind
    bench.HERE = out.parent / (args.kind + "_scene")
    original_scene = bench.scene

    def scene(directory):
        original_scene(directory)
        path = directory / "fall_hard.xml"
        tree = ET.parse(path)
        floor = tree.find(".//geom[@name='floor']")
        floor.set("friction", f"{condition.friction} 0.005 0.0001")
        if condition.payload_kg:
            # Compact ballast placed at the pelvis inertial center. Geometry and
            # collision surfaces stay fixed; point load adds no local inertia.
            robot_path = (directory / "supermimic_abs.xml").resolve()
            assert Path(tree.find("include").get("file")).resolve() == robot_path
            robot = ET.parse(robot_path)
            inertial = robot.find(".//body[@name='pelvis']/inertial")
            inertial.set("mass", str(float(inertial.get("mass")) + condition.payload_kg))
            robot.write(robot_path)
        tree.write(path)
        model = bench.mujoco.MjModel.from_xml_path(str(path))
        floor_id = model.geom("floor").id
        assert model.geom_friction[floor_id, 0] == condition.friction
        assert all(
            p < model.geom_priority[floor_id]
            for i, p in enumerate(model.geom_priority)
            if i != floor_id
        )
        assert abs(model.body("pelvis").mass[0] - 1.82817 - condition.payload_kg) < 1e-5
        return model

    bench.scene = relay.scene = scene
    real_step = bench.mujoco.mj_step
    last_data = queued = None
    actual_oob = 0

    def step(model, data, nstep=1):
        nonlocal last_data, queued, actual_oob
        if condition.delay_ticks and nstep == 4:
            issued = data.ctrl.copy()
            if last_data is not data:
                queued = data.qpos[7:].copy()
            data.ctrl[:] = queued
            queued = issued
            last_data = data
        real_step(model, data, nstep=nstep)
        # Actual actuator force on the final microstep, distinct from the
        # bench estimate using the issued target at the start of the tick.
        force = data.actuator_force
        actual_oob += int(
            np.count_nonzero(
                (force < model.actuator_forcerange[:, 0] - 1e-4)
                | (force > model.actuator_forcerange[:, 1] + 1e-4)
            )
        )

    bench.mujoco.mj_step = step
    verify(BUNDLE)
    assert (
        hashlib.sha256(EXPERT.read_bytes()).hexdigest()
        == "0231328d60f07297b38a4079e01167cf9fec65fdb2c89302144ee65e0011eb92"
    )
    expert = EXPERT if args.arm == "published" else ROOT / "models" / (args.arm + ".onnx")
    seed = TEST_SEED if args.split == "test" else TRAIN_SEED if args.split == "train" else 20261119
    if args.split == "test_late":
        seed = LATE_SEED
    if args.split == "train_late":
        seed = TRAIN_SEED
    if args.kind == "static":
        if args.split in ("train", "train_late"):
            original_poses = bench.poses
            bench.poses = lambda ctrl, s: (
                v for i, v in enumerate(original_poses(ctrl, s)) if i in TRAIN_IDS
            )
        bench.evaluate(out, seed, expert, public_runtime=True)
    else:
        rng = np.random.default_rng(seed)
        cases = []
        for track, kick, _ in relay.CASES:
            angle = rng.uniform(-0.1, 0.1)
            mag = rng.uniform(0.92, 1.08)
            c, s = np.cos(angle), np.sin(angle)
            x, y = kick
            cases.append(
                (
                    track,
                    [float(mag * (c * x - s * y)), float(mag * (s * x + c * y))],
                    int(rng.integers(80, 121)),
                )
            )
        relay.run(expert, out, True, TRACKS, cases=cases)
        (out / "cases.json").write_text(json.dumps(cases, indent=2))
    (out / "condition.json").write_text(
        json.dumps(
            dict(
                **vars(condition),
                seed=seed,
                model_sha256=hashlib.sha256(expert.read_bytes()).hexdigest(),
                last_microstep_force_oob=actual_oob,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
