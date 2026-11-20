"""Compute paired static/relay transitions and verify the raw test evidence."""

import hashlib
import json
from pathlib import Path

import numpy as np
from protocol import TEST

ROOT = Path(__file__).resolve().parent
ARMS = ("published", "selected", "bias_only")


def read(path):
    return json.loads(path.read_text())


def transitions(base, arm):
    return dict(
        gains=[i for i, (a, b) in enumerate(zip(base, arm)) if b["success"] and not a["success"]],
        losses=[i for i, (a, b) in enumerate(zip(base, arm)) if a["success"] and not b["success"]],
    )


def summarize(split, conditions, arms, selection_file):
    out = {}
    for condition, environment in conditions.items():
        all_rows = {}
        for arm in arms:
            folder = ROOT / "runs" / split / arm / condition
            base = ROOT / "runs" / split / "published" / condition
            static = read(folder / "static/trials.json")
            relay = read(folder / "relay/relay.json")
            assert len(static) == 24 and len(relay) == 5
            assert read(base / "relay/cases.json") == read(folder / "relay/cases.json")
            effort = 0
            prepush = []
            for kind in ("static", "relay"):
                meta = read(folder / kind / "condition.json")
                assert all(meta[k] == v for k, v in vars(environment).items())
                effort += meta["last_microstep_force_oob"]
                for i in range(len(static) if kind == "static" else len(relay)):
                    name = f"trial_{i:02}.npz" if kind == "static" else f"relay_{i}.npz"
                    with np.load(folder / kind / name) as d, np.load(base / kind / name) as b:
                        assert np.isfinite(d["qpos"]).all() and np.isfinite(d["trace"]).all()
                        if kind == "static":
                            assert np.array_equal(d["initial"], b["initial"])
                        else:
                            kick = int(d["kicks"][0])
                            assert int(b["kicks"][0]) == kick
                            if np.any(b["trace"][:kick, 5]):
                                prepush.append(i)
                            first = np.flatnonzero(b["trace"][:, 5] > 0)
                            stop = int(first[0]) if len(first) else len(b["trace"])
                            assert np.array_equal(d["qpos"][:stop], b["qpos"][:stop])
            all_rows[arm] = dict(
                static=sum(x["success"] for x in static),
                static_by_pose={
                    p: sum(x["success"] for x in static if x["pose"] == p)
                    for p in ("prone", "supine", "side")
                },
                static_ever_1s=sum(x["first_stable_s"] is not None for x in static),
                relay=sum(x["success"] for x in relay),
                fallen=sum(x["fallen"] for x in relay),
                base_prekick_takeover=prepush,
                issued_effort_oob=sum(x["pd_oob"] for x in static + relay),
                final_microstep_effort_oob=effort,
                static_pairs=transitions(read(base / "static/trials.json"), static),
                relay_pairs=transitions(read(base / "relay/relay.json"), relay),
            )
        out[condition] = all_rows
    totals = {}
    for arm in arms:
        totals[arm] = {
            k: sum(out[s][arm][k] for s in conditions)
            for k in ("static", "relay", "issued_effort_oob", "final_microstep_effort_oob")
        }
    baseline = totals["published"]
    selected = totals[arms[1]]
    release = dict(
        nominal_pairs_retained=not (
            out["nominal"][arms[1]]["static_pairs"]["losses"]
            or out["nominal"][arms[1]]["relay_pairs"]["losses"]
        ),
        static_total_gain=selected["static"] > baseline["static"],
        relay_total_gain=selected["relay"] > baseline["relay"],
        effort_limits_pass=not (
            selected["issued_effort_oob"] or selected["final_microstep_effort_oob"]
        ),
    )
    return dict(
        selection=read(ROOT / selection_file)["selected"],
        conditions=out,
        totals=totals,
        release_checks=release,
        decision="eligible_in_sim" if all(release.values()) else "hold",
    )


if __name__ == "__main__":
    import sys
    from late_protocol import LATE_TEST

    late = "--late" in sys.argv
    result = (
        summarize(
            "test_late",
            LATE_TEST,
            ("published", "late_selected", "late_bias"),
            "selection_late.json",
        )
        if late
        else summarize("test", TEST, ARMS, "selection.json")
    )
    (ROOT / ("results_late.json" if late else "results.json")).write_text(
        json.dumps(result, indent=2)
    )
    for s, rows in result["conditions"].items():
        print(
            s, [(p, x["static"], x["relay"], x["base_prekick_takeover"]) for p, x in rows.items()]
        )
    print(result["totals"], result["release_checks"])
