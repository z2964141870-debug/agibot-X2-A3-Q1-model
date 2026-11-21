"""Check matched inputs and count strict recovery outcomes from the saved CPU trials."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root):
    report = {"groups": {}, "raw_sha256": {}}
    for path in sorted(root.glob("*/*/results.json")):
        group = path.parent.name
        arm = path.parent.parent.name
        records = json.loads(path.read_text())
        base = root / "native" / group
        peers = json.loads((base / "results.json").read_text())
        assert len(records) == len(peers) == {"static": 24, "push": 10}[group]
        assert [r["kicks"] for r in records] == [r["kicks"] for r in peers]
        for i, row in enumerate(records):
            name = f"trial_{i:02d}.npz"
            raw = path.parent / name
            with np.load(raw) as data, np.load(base / name) as ref:
                np.testing.assert_array_equal(data["initial"], ref["initial"])
                assert np.isfinite(data["qpos"]).all() and np.isfinite(data["peaks"]).all()
                tr = data["trace"]
                assert np.all(tr[:, 6] == (1 - tr[:, 4]))
                if arm not in ("native", "direct2"):
                    assert row["max_command_step"] <= .200001
            report["raw_sha256"][str(raw.relative_to(root))] = digest(raw)
        bucket = report["groups"].setdefault(arm, {})
        bucket[group] = dict(trials=len(records), falls=sum(r["phases"][0]["fell"] for r in records),
                             successes=sum(r["success"] for r in records),
                             pre_impact_falls=sum(r["pre_impact_fall"] for r in records),
                             ever_released=sum(bool(r["releases"]) for r in records),
                             ever_solo_calm_1s=sum(r["solo_calm_s"] >= 1. for r in records),
                             peak_qd=max(r.get("peak_qd", 0.) for r in records),
                             peak_leg_qd=max(r.get("peak_leg_qd", 0.) for r in records),
                             max_penetration_mm=max(r.get("penetration_mm", 0.) for r in records),
                             results_sha256=digest(path), manifest_sha256=digest(path.parent/"manifest.json"))
    report["legacy"] = json.loads((root / "legacy_5ms_grouped/summary.json").read_text())
    report["official_mc"] = json.loads((root / "official.json").read_text())
    report["artifacts"] = {p.name: {"sha256": digest(p), "bytes": p.stat().st_size} for p in
                           [root / "Q1_prone_matched_1x.mp4", root / "official.json",
                            root / "replay.py", root / "replay_controls.py", root / "legacy_replay.py"]}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("x") as handle:
        json.dump(summarize(args.root.resolve()), handle, indent=2)
