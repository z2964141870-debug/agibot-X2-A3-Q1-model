"""Paired end-to-end selection and independent evaluation."""

import json, sys
from collections import Counter
import numpy as np
from design import *


def read(path):
    return json.loads(path.read_text())


def put(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False))


def rows(split, scene, arm, kind):
    path = destination(split, scene, arm, kind)
    if kind == "static" and not path.exists():
        effective = next((a for a in arm.split("+") if a in GAINS), "published")
        path = destination(split, scene, effective, kind)
    data = read(path / ("trials.json" if kind == "static" else "relay.json"))
    if kind == "relay":
        for r in data:
            start = r["release"]
            r["outcome5"] = (
                "recovered"
                if r["pass5"]
                else "avoided"
                if r["avoided"]
                else "no_ground_fall"
                if not r["down"]
                else "no_release"
                if start is None
                else "refall"
                if r["refell"]
                else "incomplete_cycle"
            )
            if start is None:
                continue
            with np.load(path / f"relay_{r['trial']}.npz") as f:
                xy = f["qpos"][start : start + 250, :2]
                t = f["trace"][start : start + 250]
                r["width5_cm"] = float(t[:, 7].mean()) if len(t) == 250 else None
                if r["outcome5"] == "incomplete_cycle":
                    if len(t) < 250:
                        r["outcome5"] = "short_window"
                    elif not np.all((t[:, 0] >= 0.35) & (t[:, 1] > 0.85) & (t[:, 5] == 0)):
                        r["outcome5"] = "posture"
                    elif f["body_floor"][start + 1 : start + 250].any():
                        r["outcome5"] = "body_contact"
                    elif t[:, 7].mean() > 22:
                        r["outcome5"] = "wide"
                    elif np.linalg.norm(xy[-1] - xy[0]) <= 0.25:
                        r["outcome5"] = "low_progress"
                ref = f["reference"][start : start + 250]
            for label, lo, hi in (("first", 0, 50), ("late", 150, 250)):
                if len(xy) < hi:
                    continue
                # World-frame progress along the commanded trunk heading, including turns.
                w, x, y, z = ref[lo : hi - 1, 3:7].T
                yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                vx, vy = (np.diff(xy[lo:hi], axis=0) * 50).T
                r[label + "_forward"] = float((vx * np.cos(yaw) + vy * np.sin(yaw)).mean())
                r[label + "_lateral_rms"] = float(
                    np.sqrt(np.mean((-vx * np.sin(yaw) + vy * np.cos(yaw)) ** 2))
                )
    return data


def stats(rr):
    return {
        key: sum(bool(r.get(key)) for r in rr)
        for key in ("down", "avoided", "success", "phase3", "pass5", "refell")
    }


def paired(base, other):
    assert len(base) == len(other)
    out = {}
    for key in (
        "first_forward",
        "late_forward",
        "first_lateral_rms",
        "late_lateral_rms",
        "first_speed",
        "late_speed",
        "support_roll",
        "three_roll",
        "late_roll",
        "release_s",
    ):
        pairs = [
            (a[key], b[key])
            for a, b in zip(base, other)
            if a.get(key) is not None
            and b.get(key) is not None
            and a["down"]
            and b["down"]
            and not a["refell"]
            and not b["refell"]
        ]
        if pairs:
            v, w = np.asarray(pairs).T
            out[key] = dict(
                n=len(v),
                base=float(np.median(v)),
                arm=float(np.median(w)),
                delta=float(np.median(w - v)),
                ratio=float(np.median(w / np.maximum(v, 1e-9))),
            )
        else:
            out[key] = dict(n=0)
    out["gain"] = sum(b["pass5"] and not a["pass5"] for a, b in zip(base, other))
    out["loss"] = sum(a["pass5"] and not b["pass5"] for a, b in zip(base, other))
    return out


def select():
    base = rows("dev", "daily", "published", "relay")
    pp = {}
    for a in PACES:
        r = rows("dev", "daily", a, "relay")
        pp[a] = dict(counts=stats(r), paired=paired(base, r))

    def rank(a):
        d = pp[a]["paired"]
        safe = d["first_speed"].get("ratio", 0) >= 0.8 and d["late_speed"].get("ratio", 0) >= 0.95
        return (
            safe,
            pp[a]["counts"]["pass5"],
            -pp[a]["counts"]["refell"],
            -d["support_roll"].get("ratio", 999),
            PACES[a][0],
            -PACES[a][1],
        )

    # Keep one active intervention to quantify it on holdout, even if development favors baseline.
    pace = max((a for a in PACES if a != "published"), key=rank)
    gg = {}
    for g in GAINS:
        parts = []
        for scene in ("half", "lag", "long", "trainvar"):
            s = rows("dev", scene, g, "static")
            r = rows("dev", scene, g, "relay")
            parts.append([sum(x["success"] for x in s), sum(x["pass5"] for x in r)])
        gg[g] = parts
    gain = max(GAINS, key=lambda g: sum(s / 9 + r / 10 for s, r in gg[g]))
    put(D / "selection.json", dict(pace=pace, gain=gain, pace_dev=pp, gain_dev=gg))


def parity():
    folder = destination("parity", "daily", "published", "relay")
    old = OLD / "runs/test0/normal/published/relay"
    report = []
    for i, (a, b) in enumerate(zip(read(old / "relay.json"), read(folder / "relay.json"))):
        with np.load(old / f"relay_{i}.npz") as x, np.load(folder / f"relay_{i}.npz") as y:
            n = len(x["qpos"])
            assert n == b["legacy_end"] and a["success"] == b["success"]
            assert all(
                np.array_equal(x[k], y[k][: len(x[k])])
                for k in ("qpos", "trace", "reference", "loads", "ankle_offset", "body_floor")
            )
            report.append(n)
    assert len(report) == 10
    put(D / "parity.json", dict(episodes=10, ticks=sum(report), lengths=report))


def final():
    sel = read(D / "selection.json")
    a, b = sel["pace"], sel["gain"]
    arms = ["published", a, b, a + "+" + b]
    daily = {}

    def dailyrows(arm, kind):
        return sum((rows(s, "daily", arm, kind) for s in ("hold0", "hold1")), [])

    base = dailyrows("published", "relay")
    for arm in arms:
        r = dailyrows(arm, "relay")
        rr = dailyrows(arm, "double")
        stat = dailyrows(arm, "static")
        flat = destination("hold0", "daily", arm, "flat")
        frames = 0
        for i in range(2):
            with (
                np.load(flat / f"relay_{i}.npz") as x,
                np.load(D.parent / f"flat/published/relay_{i}.npz") as y,
            ):
                assert all(
                    np.array_equal(x[k], y[k]) for k in ("qpos", "trace", "reference", "loads")
                )
                frames += len(x["trace"])
        p = paired(base, r)
        daily[arm] = dict(
            static=sum(x["success"] for x in stat),
            static_n=len(stat),
            counts=stats(r),
            n=len(r),
            outcome5=dict(Counter(v["outcome5"] for v in r)),
            static_by_pose=dict(Counter(v["pose"] for v in stat if v["success"])),
            paired=p,
            repeat=stats(rr),
            repeat_n=len(rr),
            both_pushes=sum(len(v["phases5"]) == 2 for v in rr),
            flat_equal_ticks=frames,
        )
    stress = {}
    for scene in ("half", "lag", "long", "variable", "slick", "payload", "combined"):
        stress[scene] = {}
        for arm in arms + ["linear", "fixed", "motion"]:
            st = rows("hold0", scene, arm, "static")
            r = rows("hold0", scene, arm, "relay")
            stress[scene][arm] = dict(
                static=sum(x["success"] for x in st), static_n=len(st), counts=stats(r), n=len(r)
            )
    for arm in arms:
        d = daily[arm]
        ref = daily["published"]
        p = d["paired"]
        gates = dict(
            static=d["static"] >= ref["static"],
            walk5=d["counts"]["pass5"] >= ref["counts"]["pass5"],
            refalls=d["counts"]["refell"] <= ref["counts"]["refell"],
            repeat=d["repeat"]["pass5"] >= ref["repeat"]["pass5"],
            speed1=p["first_forward"].get("ratio", 0) >= 0.8,
            speed_late=p["late_forward"].get("ratio", 0) >= 0.95,
            time=p["release_s"].get("delta", 999) <= 0.5,
            roll3=p["three_roll"].get("ratio", 999) <= 1.0,
            roll_late=p["late_roll"].get("ratio", 999) <= 1.0,
            benefit=(
                d["counts"]["pass5"] - ref["counts"]["pass5"] >= 0.05 * len(base)
                or p["support_roll"].get("ratio", 999) <= 0.90
            ),
        )
        d["gates"] = gates
    accepted = [arm for arm in arms[1:] if all(daily[arm]["gates"].values())]
    accepted.sort(key=lambda arm: (-daily[arm]["counts"]["pass5"], "+" in arm))
    receipts = [
        f
        for f in (D / "runs").glob("hold*/*/*/*/receipt.json")
        if f.parent.name in ("static", "relay", "double", "flat")
    ]
    rec = [read(f) for f in receipts]
    pd = sum(
        sum(int(r["pd_oob"]) for r in read(f))
        for f in (D / "runs").glob("hold*/*/*/*/*ls.json")
        if f.name == "trials.json"
    )
    pd += sum(
        sum(int(r["pd_oob"]) for r in read(f))
        for f in (D / "runs").glob("hold*/*/*/*/relay.json")
        if f.parent.name in ("relay", "double", "flat")
    )
    assert not pd and not sum(v["force_oob"] for v in rec)
    put(
        D / "results.json",
        dict(
            daily=daily,
            stress=stress,
            selected=accepted[0] if accepted else "published",
            physical_jobs=len(rec),
            issued_oob=pd,
            actual_force_oob=sum(v["force_oob"] for v in rec),
            delivered_demand_oob=sum(v["applied_demand_oob"] for v in rec),
        ),
    )


if __name__ == "__main__":
    {"select": select, "parity": parity, "final": final}[sys.argv[1]]()
