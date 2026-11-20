"""Check transport causality, held-out schedules, pace prefixes and standalone inference."""

import sys, subprocess
import numpy as np
from design import *
from assess import read, put
from transport import Transport


def queue_contracts():
    checked = 0
    for mode in ("0", "10", "20", "30", "unseen", "train_var"):
        t = Transport(mode, 191)
        ref = []
        applied = -1
        missing = 0
        for seq in range(180):
            cmd = np.full(22, seq, dtype=float)
            features, valid, age = t.observation(cmd)
            if not valid:
                missing += 1
            if valid:
                assert 0 <= age <= 1.5 and t.ack[0] < t.micro and t.ack[1] < seq
            delay = t.submit(cmd, np.full(22, -1.0))
            ref.append((4 * seq + delay, seq))
            for micro in range(4 * seq, 4 * seq + 4):
                due = max([sid for deadline, sid in ref if deadline <= micro], default=-1)
                got = t.next()
                assert due >= applied and (got == due).all()
                applied = due
                checked += 1
        if mode in ("unseen", "train_var"):
            assert missing > 1
    return checked


def match(path1, path2, pattern, keys):
    total = 0
    for a in sorted(path1.glob(pattern)):
        with np.load(a) as x, np.load(path2 / a.name) as y:
            for k in keys:
                np.testing.assert_array_equal(x[k], y[k])
            total += len(x["qpos"])
    assert total
    return total


def main():
    selection = read(D / "selection.json")
    b = selection["gain"]
    a = selection["pace"]
    result = {"queue_microsteps": queue_contracts(), "independent": {}}
    for scene, kind in (("variable", "relay"), ("lag", "static")):
        job = ("hold0", scene, b, kind)
        dst = D / "entry_replay" / "hold0" / scene / b / kind
        if not (dst / "receipt.json").exists():
            subprocess.run([sys.executable, str(D / "worker.py"), *job, "independent"], check=True)
        keys = ("qpos", "trace", "loads", "reference") if kind == "relay" else ("qpos", "trace")
        result["independent"][scene] = match(
            destination(*job), dst, ("relay_*" if kind == "relay" else "trial_*") + ".npz", keys
        )
    prefix = flat = confirmed = 0
    for split in ("hold0", "hold1"):
        for left, right in (("published", b), (a, a + "+" + b)):
            root = destination(split, "daily", left, "relay")
            other = destination(split, "daily", right, "relay")
            confirmed += match(root, other, "relay_*.npz", ("qpos", "trace", "loads", "reference"))
        for kind in ("relay", "double"):
            base = destination(split, "daily", "published", kind)
            other = destination(split, "daily", a, kind)
            for i, row in enumerate(read(base / "relay.json")):
                first = next((r for p in row["phases5"] for r in p["releases"]), None)
                with np.load(base / f"relay_{i}.npz") as x, np.load(other / f"relay_{i}.npz") as y:
                    n = len(x["trace"]) if first is None else first
                    for key in ("qpos", "reference", "trace", "loads"):
                        np.testing.assert_array_equal(x[key][:n], y[key][:n])
                    prefix += n
        for arm in ("published", a, b, a + "+" + b):
            flat += match(
                destination(split, "daily", arm, "flat"),
                D.parent / "flat/published",
                "relay_*.npz",
                ("qpos", "trace", "loads", "reference"),
            )
    result.update(
        pace_prefix_ticks=prefix, flat_ticks_over_all_runs=flat, no_delay_relay_ticks=confirmed
    )
    put(D / "verification.json", result)


if __name__ == "__main__":
    main()
