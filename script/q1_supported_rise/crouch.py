"""Test the autonomous standing phase from hardware-derived crouches at rest."""

import argparse
import json
from pathlib import Path
import numpy as np

from limbs import CatchRecovery, add_limb_contacts, mj, replay


def starts(model, records, audit, seed):
    rng = np.random.default_rng(seed)
    for name in (
        "q1_official_standup_from_mcap",
        "q1_official_standup_faceup_20261120",
    ):
        with np.load(records / (name + ".npz")) as log:
            order = [
                log["joint_names"].tolist().index(model.joint(i).name)
                for i in range(1, 23)
            ]
            stamp = log["event_ns"].item()
            jt = (log["joint_state_stamp"] - stamp) * 1e-9
            it = (log["imu_stamp"] - stamp) * 1e-9
            for t in (6.55, 6.8, 7.05, 7.30):
                joints = np.array([np.interp(t, jt, log["q"][:, j]) for j in order])
                quat = log["orientation_xyzw"][np.argmin(abs(it - t)), [3, 0, 1, 2]]
                for replica in range(3):
                    d = mj.MjData(model)
                    d.qpos[3:7] = quat / np.linalg.norm(quat)
                    d.qpos[7:] = joints + rng.uniform(-0.025, 0.025, 22)
                    d.qpos[2] = -0.2
                    mj.mj_forward(model, d)
                    floor = model.geom("floor").id
                    distances = [c.dist for c in d.contact if floor in c.geom]
                    assert distances
                    d.qpos[2] -= min(distances)
                    mj.mj_forward(model, d)
                    audit.append(
                        dict(
                            record=name,
                            t=t,
                            replica=replica,
                            height=float(d.qpos[2]),
                            cosine=float(1.0 - 2 * np.sum(d.qpos[4:6] ** 2)),
                        )
                    )
                    yield name, d.qpos.copy()


def evaluate(records, expert, out, gate_class, seed):
    out = out.resolve()
    scene = add_limb_contacts(out.with_name(out.name + "_initial"))
    audit = []
    replay.bench.poses = lambda ctrl, seed: starts(scene, records, audit, seed)
    replay.bench.scene = add_limb_contacts

    class Owned(gate_class):
        def reset(self):
            super().reset()
            self.owner, self.entered = "recovery", 0.0
            self.rise.expert.gate.mode = "recover"

    replay.Q1AutonomousRecovery = Owned
    replay.simulate("static", seed, out, expert.resolve(), 500, 11, True, 2)
    results = json.loads((out / "results.json").read_text())
    for entry, r in zip(audit, results, strict=True):
        entry["stood_and_handed_back"] = bool(
            len(r["releases"]) == 1
            and not r["wrong_release"]
            and not r["takeovers"]
            and r["phases"][0]["post_1s"]
        )
        entry["solo_calm_s"] = r["solo_calm_s"]
    (out / "standing_phase.json").write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("records", type=Path)
    p.add_argument("expert", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--seed", type=int, default=77331)
    a = p.parse_args()
    evaluate(a.records, a.expert, a.out, CatchRecovery, a.seed)
