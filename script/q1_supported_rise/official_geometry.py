"""Check arm/leg collision proxies at measured hardware joint angles."""

import hashlib
import json
from pathlib import Path
import numpy as np
from limbs import add_limb_contacts, mj


def evaluate(model, record):
    d = mj.MjData(model)
    d.qpos[2] = 2.0  # Internal distances are invariant to rigid root pose.
    pairs = {tuple(sorted(x)) for x in zip(model.pair_geom1, model.pair_geom2)}
    transitions = json.loads(record.with_suffix(".json").read_text())["transitions"]
    start, stop = [
        next(t["t_rel_s"] for t in transitions if t["desc"] == desc)
        for desc in ("STAND_UP", "BIPED_WALK_RUN")
    ]
    with np.load(record) as log:
        t = (log["joint_state_stamp"] - log["event_ns"].item()) * 1e-9
        ids = np.flatnonzero((t >= start) & (t < stop))
        order = [
            log["joint_names"].tolist().index(model.joint(i).name) for i in range(1, 23)
        ]
        q = log["q"][ids][:, order]
    worst = {"mm": 0.0}
    counts = {}
    any_hit = 0
    for i, joints in zip(ids, q):
        d.qpos[7:] = joints
        mj.mj_forward(model, d)
        hit = False
        seen = set()
        for c in d.contact:
            if tuple(sorted(c.geom)) in pairs and c.dist < 0.0:
                hit = True
                names = tuple(sorted(model.geom(int(g)).name for g in c.geom))
                seen.add(" / ".join(names))
                if -1000 * c.dist > worst["mm"]:
                    worst = dict(
                        mm=float(-1000 * c.dist), geoms=names, t_rel_s=float(t[i])
                    )
        any_hit += hit
        for pair in seen:
            counts[pair] = counts.get(pair, 0) + 1
    return dict(
        record=record.name,
        sha256=hashlib.sha256(record.read_bytes()).hexdigest(),
        frames=len(ids),
        start_s=start,
        stop_s=stop,
        hit_frames=any_hit,
        worst=worst,
        pair_frames=counts,
    )


if __name__ == "__main__":
    import sys

    directory = Path(sys.argv[1])
    out = Path(sys.argv[2])
    model = add_limb_contacts(out.with_suffix(".scene"))
    files = (
        "q1_official_standup_from_mcap.npz",
        "q1_official_standup_faceup_20261120.npz",
    )
    out.write_text(
        json.dumps([evaluate(model, directory / f) for f in files], indent=2)
    )
