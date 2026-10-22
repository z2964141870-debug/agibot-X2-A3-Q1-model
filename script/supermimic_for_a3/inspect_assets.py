#!/usr/bin/env python3
"""Inspect the existing A3 model and motion inputs without modifying them."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess

import mujoco
import numpy as np


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    vendor = args.root / "script/vendor/sonic_for_a3"
    xml = vendor / "gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_passive_foot_twostage_fit_optimized.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    actuated = set(model.actuator_trnid[:, 0].tolist())
    joints = []
    for i in range(model.njnt):
        joints.append({"name": model.joint(i).name, "type": int(model.jnt_type[i]),
                       "qpos": int(model.jnt_qposadr[i]), "dof": int(model.jnt_dofadr[i]),
                       "limited": bool(model.jnt_limited[i]), "range": model.jnt_range[i].tolist(),
                       "actuated": i in actuated})
    actuators = [{"name": model.actuator(i).name,
                  "joint": model.joint(int(model.actuator_trnid[i, 0])).name,
                  "gain": model.actuator_gainprm[i].tolist(),
                  "bias": model.actuator_biasprm[i].tolist(),
                  "force_range": model.actuator_forcerange[i].tolist(),
                  "ctrl_range": model.actuator_ctrlrange[i].tolist()}
                 for i in range(model.nu)]
    roots = [vendor / "a3_data", args.root / "data/datasets", args.root / "data/training"]
    motions = []
    for base in roots:
        if not base.exists():
            continue
        for file in sorted(base.rglob("*.pkl")):
            if len(motions) >= 80:
                break
            entry = identity(file)
            try:
                with file.open("rb") as stream:
                    value = pickle.load(stream)
                if isinstance(value, dict):
                    entry["fields"] = {str(k): {"type": type(v).__name__,
                        "shape": list(v.shape) if hasattr(v, "shape") else None,
                        "value": v if isinstance(v, (int, float, str)) else None}
                        for k, v in value.items()}
                else:
                    entry["type"] = type(value).__name__
            except Exception as exc:
                entry["error"] = str(exc)
            motions.append(entry)
    geoms = [{"name": model.geom(i).name, "body": model.body(int(model.geom_bodyid[i])).name,
              "type": int(model.geom_type[i]), "contype": int(model.geom_contype[i]),
              "conaffinity": int(model.geom_conaffinity[i])}
             for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]
    receipt = {"status": "INSPECTED_NOT_TRAINED", "mujoco": mujoco.__version__,
               "model": identity(xml), "nq": model.nq, "nv": model.nv, "nu": model.nu,
               "joints": joints, "actuators": actuators,
               "bodies": [model.body(i).name for i in range(model.nbody)],
               "collision_geoms": geoms, "motions": motions,
               "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
               "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu", "--format=csv"], text=True)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"nq": model.nq, "nv": model.nv, "nu": model.nu,
                      "unactuated": [j for j in joints if not j["actuated"]],
                      "motions": len(motions), "first_motions": motions[:3]}, indent=2))


if __name__ == "__main__":
    main()
