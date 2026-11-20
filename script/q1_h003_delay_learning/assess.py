"""Pair trajectories, verify bypass and summarize fixed and transferred delay performance."""

import json
from pathlib import Path
import numpy as np
from protocol import ROOT, EVALUATION, SCHEDULE
from transfer_protocol import EXTRA_SCHEDULE
from run import read, sha


def one(split, condition, arms):
    records = {}
    for arm in arms:
        folder = ROOT / "runs" / split / condition / arm
        base = ROOT / "runs" / split / condition / "published"
        part = {}
        for kind in ("static", "relay"):
            name = "trials.json" if kind == "static" else "relay.json"
            rows = read(folder / kind / name)
            base_rows = read(base / kind / name)
            info = read(folder / kind / "receipt.json")
            seed, ns, nr = (SCHEDULE | EXTRA_SCHEDULE)[split]
            assert info["seed"] == seed and len(rows) == len(info["episodes"]) == (
                ns if kind == "static" else nr
            )
            if kind == "relay":
                assert read(folder / kind / "cases.json") == read(base / kind / "cases.json")
            same_ticks = 0
            before_push = 0
            for i, row in enumerate(rows):
                f = f"trial_{i:02}.npz" if kind == "static" else f"relay_{i}.npz"
                with np.load(folder / kind / f) as d, np.load(base / kind / f) as b:
                    assert np.isfinite(d["trace"]).all() and np.isfinite(d["qpos"]).all()
                    if kind == "static":
                        assert np.array_equal(d["initial"], b["initial"])
                    else:
                        if np.any(d["trace"][: int(d["kicks"][0]), 5] > 0):
                            before_push += 1
                        positive = np.flatnonzero(b["trace"][:, 5] > 0)
                        first = int(positive[0]) if len(positive) else len(b["trace"])
                        assert np.array_equal(d["qpos"][:first], b["qpos"][:first])
                    if condition == "normal" and not arm.startswith("unconditional_"):
                        assert all(np.array_equal(d[k], b[k]) for k in ("trace", "qpos"))
                        same_ticks += len(d["trace"])
            part[kind] = dict(
                cases=len(rows),
                success=sum(x["success"] for x in rows),
                gate_engaged=sum(x["active"] > 0 for x in info["episodes"]),
                handoff=sum(x["handoff"] for x in info["episodes"]),
                release=sum(x["released"] for x in info["episodes"]),
                gains=[
                    i
                    for i, (a, b) in enumerate(zip(rows, base_rows))
                    if a["success"] and not b["success"]
                ],
                losses=[
                    i
                    for i, (a, b) in enumerate(zip(rows, base_rows))
                    if b["success"] and not a["success"]
                ],
                pd_oob=sum(x["pd_oob"] for x in rows),
                nominal_exact_ticks=same_ticks,
                fallen=sum(x.get("fallen", False) for x in rows),
                before_push=before_push,
                refall=sum(x.get("refall", False) for x in rows),
                checks=info.get("delivered_joint_checks", 0),
                demand_over=info.get("delivered_demand_over", 0),
                sampled_force_over=info.get("final_microstep_force_oob", 0),
                max_demand_ratio=info.get("max_demand_ratio", 0.0),
            )
        records[arm] = part
    return records


def main():
    chosen = [
        read(ROOT / f)["selected"]
        for f in ("linear_selection.json", "selection.json", "motion_selection.json")
    ]
    arms = ["published", *chosen]
    data = {
        split: {s: one(split, s, arms) for s in cond}
        for split, cond in [
            ("test0", EVALUATION),
            ("test1", EVALUATION),
            ("test2", ["half", "lag", "long"]),
        ]
    }
    audit = {
        s: one("audit", s, ["published"])["published"]
        for s in ("normal", "lag", "recovery_only", "outside_recovery")
    }
    # Prior nominal and 20 ms physical records on the exact diagnostic seed.
    old = ROOT.parent / "pace_generalization/runs/test_late/published"
    for a, b in (("normal", "nominal"), ("lag", "delay")):
        for kind in ("static", "relay"):
            files = sorted((ROOT / "runs/audit" / a / "published" / kind).glob("*.npz"))
            for p in files:
                with np.load(p) as x, np.load(old / b / kind / p.name) as y:
                    assert all(np.array_equal(x[k], y[k]) for k in ("qpos", "trace"))
    table = {}
    for s in EVALUATION:
        table[s] = {
            a: [
                sum(data[d][s][a][k]["success"] for d in ("test0", "test1"))
                for k in ("static", "relay")
            ]
            for a in arms
        }
    always = {
        s: one(s, "normal", ["published", "unconditional_net", "unconditional_motion"])
        for s in ("test0", "test1")
    }
    always_counts = {
        a: [sum(always[s][a][k]["success"] for s in always) for k in ("static", "relay")]
        for a in always["test0"]
    }
    lag_by_pose = {}
    for arm in arms:
        rows = []
        for split in ("test0", "test1"):
            rows += read(ROOT / "runs" / split / "lag" / arm / "static/trials.json")
        lag_by_pose[arm] = {
            pose: sum(r["success"] for r in rows if r["pose"] == pose)
            for pose in ("prone", "supine", "side")
        }
    result = dict(
        arms=arms,
        audit=audit,
        primary=table,
        per_split=data,
        archive_parity=58,
        always_on_nominal=always_counts,
        lag_static_by_pose=lag_by_pose,
    )
    (ROOT / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(table), flush=True)
    print(
        {
            s: {a: [v["static"]["success"], v["relay"]["success"]] for a, v in x.items()}
            for s, x in data["test2"].items()
        }
    )


if __name__ == "__main__":
    main()
