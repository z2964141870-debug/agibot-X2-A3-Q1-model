#!/usr/bin/env python3
"""Inspect effective foot-floor friction without integrating Q1 physics."""

import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np

from supermimic.q1_assets import Q1Assets, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assets = Q1Assets(args.root)
    model = copy.copy(assets.model)
    floor = model.geom("floor").id
    feet = set(int(value) for value in assets.foot_geoms)
    pose = model.qpos0.copy()
    pose[assets.qadr] = assets.q0
    pose[2] -= .01
    rows = []
    for coefficient in (.3, 1., 1.2):
        model.geom_friction[assets.foot_geoms, 0] = coefficient
        data = mujoco.MjData(model)
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        contacts = []
        for contact in data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            if floor not in pair or not pair.intersection(feet):
                continue
            contacts.append({"geom_ids": [int(contact.geom1), int(contact.geom2)],
                             "distance_m": float(contact.dist), "dimension": int(contact.dim),
                             "effective_friction": contact.friction.tolist()})
        if not contacts or data.time != 0.:
            raise ValueError("Contact fixture did not produce foot-floor contacts at time zero")
        rows.append({"requested_foot_friction": coefficient, "contacts": contacts,
                     "unique_effective_tangential_friction": sorted(set(
                         float(contact["effective_friction"][0]) for contact in contacts))})
    effective = [row["unique_effective_tangential_friction"] for row in rows]
    report = {"status": "EFFECTIVE_CONTACT_PARAMETER_AUDIT_COMPLETED",
              "rows": rows, "requested_friction_changes_contacts": effective[0] != effective[-1],
              "floor": {"geom_id": floor, "priority": int(model.geom_priority[floor]),
                        "friction": model.geom_friction[floor].tolist()},
              "feet": [{"geom_id": index, "priority": int(model.geom_priority[index]),
                        "body": model.body(int(model.geom_bodyid[index])).name} for index in sorted(feet)],
              "explicit_contact_pairs": model.npair,
              "global_contact_override": bool(model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_OVERRIDE),
              "fixture": "Q1 default pose with registered neutral hinge positions; root lowered 1cm for contact inspection; same pose in all cases",
              "pose": pose.tolist(), "assets": assets.manifest(),
              "simulator_source": identity(Path(__file__).parent / "supermimic/q1_sim.py"),
              "source": identity(__file__), "new_physics_steps": 0, "new_optimizer_updates": 0,
              "storage_status": "LOCAL_ONLY"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
