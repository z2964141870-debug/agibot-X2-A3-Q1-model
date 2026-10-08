#!/usr/bin/env python3
"""Compile Q1 MJCFs and check finite initial physics without robot control."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def inspect(path):
    result = {"asset": str(path), "sha256_record": "see resource manifest", "policy_test": False}
    try:
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        initial_finite = all(np.isfinite(a).all() for a in
                             (data.qpos, data.qvel, data.qacc, data.sensordata))
        # Passive steps validate basic engine operation, not standing or tracking.
        for _ in range(100):
            mujoco.mj_step(model, data)
        final_finite = all(np.isfinite(a).all() for a in
                           (data.qpos, data.qvel, data.qacc, data.sensordata))
        warnings = {mujoco.mjtWarning(i).name: int(w.number)
                    for i, w in enumerate(data.warning) if w.number}
        result.update({
            "compiled": True,
            "nq": model.nq,
            "nv": model.nv,
            "nu": model.nu,
            "nsensor": model.nsensor,
            "nsensordata": model.nsensordata,
            "nmesh": model.nmesh,
            "mass_kg": float(model.body_mass.sum()),
            "timestep_s": model.opt.timestep,
            "passive_steps": 100,
            "initial_finite": bool(initial_finite),
            "final_finite": bool(final_finite),
            "warnings": warnings,
            "basic_engine_check": bool(initial_finite and final_finite and not warnings),
        })
    except Exception as error:
        result.update({"compiled": False, "error": str(error), "basic_engine_check": False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = sorted(args.asset_dir.glob("*.xml"))
    if not paths:
        parser.error("No MJCF XML files found")
    results = [inspect(p) for p in paths]
    payload = {"mujoco_version": mujoco.__version__, "scope": "compile_and_passive_steps", "assets": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if all(r["basic_engine_check"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
