"""Aggregate exclusive-recovery tests; simulator truth is only read for scoring."""

import json
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent


def gather(root):
    totals = {
        "releases": 0,
        "wrong_release": 0,
        "truth_previous_half_second": 0,
        "pd_oob": 0,
        "owner_conflicts": 0,
    }
    groups = {}
    episodes = []
    for f in sorted(root.glob("*/*/results.json")):
        group = f.parent.name
        for r in json.loads(f.read_text()):
            with np.load(f.parent / f"trial_{r['trial']:02d}.npz") as d:
                tr = d["trace"]
                if group == "flat":
                    with np.load(D.parent / f"flat/published/relay_{r['trial']}.npz") as archive:
                        np.testing.assert_array_equal(d["qpos"], archive["qpos"])
            counters = groups.setdefault(
                group,
                dict(
                    n=0,
                    confirmed=0,
                    solo_1s=0,
                    released=0,
                    static_pass=0,
                    post_1s=0,
                    terminal_solo_1s=0,
                    fallen=0,
                    avoided=0,
                    double_clean=0,
                    released_pose={},
                    release_s=[],
                ),
            )
            counters["n"] += 1
            counters["confirmed"] += int(tr[:, 8].any() or bool(r["releases"]))
            counters["solo_1s"] += int(r["solo_calm_s"] >= 1)
            counters["terminal_solo_1s"] += int(np.all((tr[-50:, 5] == 1) & (tr[-50:, 4] == 1)))
            counters["released"] += int(bool(r["releases"]))
            first = r["phases"][0]
            counters["fallen"] += int(first["fell"])
            counters["avoided"] += int(
                not first["fell"]
                and not first["captures"]
                and np.all(tr[-50:, 0] >= 0.35)
                and np.all(tr[-50:, 1] > 0.85)
            )
            counters["post_1s"] += int(first["fell"] and first["post_1s"])
            if group == "double":
                counters["double_clean"] += int(
                    len(r["kicks"]) == 2
                    and all(
                        (
                            p["fell"]
                            and len(p["captures"]) == len(p["releases"]) == 1
                            and p["post_1s"]
                        )
                        or (not p["fell"] and not p["captures"] and not p["releases"])
                        for p in r["phases"]
                    )
                    and np.all(tr[-50:, 0] >= 0.35)
                    and np.all(tr[-50:, 1] > 0.85)
                )
            first_release = r["releases"][:1]
            if first_release:
                counters["release_s"].append((first_release[0] - (r["kicks"] or [0])[0]) * 0.02)
            good = first["post_1s"] and r["solo_calm_s"] >= 1 and not r["wrong_release"]
            if group == "static" and good:
                counters["static_pass"] += 1
                pose = r["pose"]
                counters["released_pose"][pose] = counters["released_pose"].get(pose, 0) + 1
            totals["releases"] += len(r["releases"])
            totals["pd_oob"] += r["pd_oob"]
            totals["wrong_release"] += r["wrong_release"]
            totals["truth_previous_half_second"] += sum(
                int(np.all(tr[t - 25 : t + 1, 5])) for t in r["releases"]
            )
            totals["owner_conflicts"] += int(np.count_nonzero(tr[:, 6] != (1 - tr[:, 4])))
            episodes.append(dict(seed=int(f.parent.parent.name), group=group, **r))
    for v in groups.values():
        times = v.pop("release_s")
        v["median_release_s"] = round(float(np.median(times)), 3) if times else None
    return dict(groups=groups, totals=totals, episodes=episodes)


if __name__ == "__main__":
    result = {suite: gather(D / "runs" / suite) for suite in ("dev", "test")}
    assert {k: v["n"] for k, v in result["test"]["groups"].items()} == dict(
        static=48, push=20, double=10, offline=6, flat=4
    )
    (D / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result["test"].items() if k != "episodes"}, indent=2))
