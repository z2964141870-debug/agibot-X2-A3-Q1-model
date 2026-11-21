"""Replay with explicit forearm/wrist vs leg pairs and check overlaps in saved motion."""

import argparse
import itertools
import json
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np

from capture import CatchRecovery, replay

mj = replay.mujoco
original_scene = replay.bench.scene


def add_limb_contacts(directory):
    directory = directory.resolve()
    nominal = original_scene(directory)
    names = [nominal.geom(i).name or "" for i in range(nominal.ngeom)]
    arms = [
        s for s in names if "_collision_" in s and ("_elbow_" in s or "_wrist_" in s)
    ]
    legs = [
        s
        for s in names
        if "_collision_" in s and any(t in s for t in ("_hip_", "_knee_", "_ankle_"))
    ]
    pairs = list(itertools.product(arms, legs))
    assert len(pairs) == 96
    source = directory / "fall_hard.xml"
    xml = ET.parse(source)
    contact = ET.SubElement(xml.getroot(), "contact")
    for a, b in pairs:
        ET.SubElement(contact, "pair", geom1=a, geom2=b)
    source.rename(directory / "original_ground.xml")
    xml.write(source)
    model = mj.MjModel.from_xml_path(str(source))
    assert model.npair == len(pairs)
    return model


def audit(root, scene, output):
    model = add_limb_contacts(scene)
    d = mj.MjData(model)
    mj.mj_resetDataKeyframe(model, d, model.key("stand").id)
    mj.mj_forward(model, d)
    pairs = {tuple(sorted(pair)) for pair in zip(model.pair_geom1, model.pair_geom2)}

    def overlaps():
        return [c for c in d.contact if tuple(sorted(c.geom)) in pairs and c.dist < 0.0]

    assert not overlaps(), "explicit pair intersects in nominal standing pose"
    trials = []
    for path in sorted(root.glob("trial_*.npz")):
        with np.load(path) as f:
            owned = f["trace"][:, 4].astype(bool)
            poses = f["qpos"]
        n = 0
        worst = {"mm": 0.0}
        for tick in np.flatnonzero(owned):
            d.qpos[:] = poses[tick]
            mj.mj_forward(model, d)
            hits = overlaps()
            if hits:
                n += 1
                c = min(hits, key=lambda item: item.dist)
                if -1000 * c.dist > worst["mm"]:
                    worst = {
                        "mm": float(-1000 * c.dist),
                        "end_tick": int(tick),
                        "geoms": [model.geom(int(g)).name for g in c.geom],
                    }
        trials.append(
            dict(
                trial=path.stem,
                owned_frames=int(owned.sum()),
                frames_intersecting=n,
                worst=worst,
            )
        )
    output.write_text(
        json.dumps(dict(pairs=len(pairs), nominal_overlaps=0, trials=trials), indent=2)
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("group", choices=["audit", "static", "push"])
    p.add_argument("source", type=Path, help="saved trials for audit, ONNX otherwise")
    p.add_argument("out", type=Path)
    p.add_argument("--seed", type=int, default=77331)
    p.add_argument("--catch", action="store_true")
    a = p.parse_args()
    if a.group == "audit":
        audit(a.source, a.out.with_suffix(".scene"), a.out)
    else:
        replay.bench.scene = add_limb_contacts
        if a.catch:
            replay.Q1AutonomousRecovery = CatchRecovery
        replay.simulate(
            a.group, a.seed, a.out.resolve(), a.source.resolve(), 500, 11, True, 2
        )
