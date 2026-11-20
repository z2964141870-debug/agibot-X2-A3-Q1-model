"""Score the first sustained single support and pair physically identical prefixes."""

import argparse
import json
from pathlib import Path

import numpy as np
from score_gait import gait
from bench import quat_wxyz_to_mat


def first_support(data, first):
    loads = data["loads"][first : first + 150]
    trace = data["trace"][first : first + 150]
    if len(loads) < 150:
        return None
    for k in range(len(loads) - 3):
        for side in (0, 1):
            window = loads[k : k + 4]
            if np.all(window[:, side] > 20) and np.all(window[:, 1 - side] < 5):
                segment = data["qpos"][first + k : first + k + 15, 3:7]
                if len(segment) < 15:
                    return None
                R = quat_wxyz_to_mat(segment)
                roll = np.rad2deg(np.unwrap(np.arctan2(R[:, 2, 1], R[:, 2, 2])))
                return {
                    "onset_s": k * 0.02,
                    "side": ["left", "right"][side],
                    "roll_range_deg": float(np.ptp(roll)),
                    "roll_peak_deg": float(np.max(np.abs(roll))),
                    "width_cm": float(trace[k : k + 4, 7].mean()),
                }
    return None


def evaluate(folder):
    case_list = json.loads((folder / "cases.json").read_text())
    rows = json.loads((folder / "relay.json").read_text())
    assert len(rows) == len(case_list) and len(rows) == len(list(folder.glob("relay_*.npz")))
    for i, row in enumerate(rows):
        with np.load(folder / f"relay_{i}.npz") as f:
            d = {
                k: f[k]
                for k in (
                    "trace",
                    "qpos",
                    "reference",
                    "loads",
                    "reanchors",
                    "kicks",
                    "ankle_offset",
                )
            }
        assert all(np.isfinite(x).all() for x in d.values())
        row["reanchors"] = d["reanchors"].tolist()
        row["peak_offset_rad"] = float(np.max(d["ankle_offset"]))
        if not row["reanchors"]:
            row["release"] = None
            continue
        g = d["trace"][:, 5]
        hits = np.flatnonzero((np.arange(len(g)) >= row["reanchors"][0]) & (g == 0))
        row["release"] = int(hits[0]) if len(hits) else None
        if row["release"] is not None:
            row["gait"] = gait(d, row["release"], 150)
            row["support"] = first_support(d, row["release"])
    return rows


def compare(root):
    names = sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "cases.json").is_file())
    rows = {name: evaluate(root / name) for name in names}
    base = rows["published"]
    stats = {}
    for name, arm in rows.items():
        assert (root / name / "cases.json").read_bytes() == (
            root / "published/cases.json"
        ).read_bytes()
        for i, (a, b) in enumerate(zip(base, arm)):
            with (
                np.load(root / name / f"relay_{i}.npz") as v,
                np.load(root / "published" / f"relay_{i}.npz") as w,
            ):
                cut = a["reanchors"][0] if a["reanchors"] else len(w["trace"])
                for key in ("trace", "qpos", "loads"):
                    assert np.array_equal(w[key][:cut], v[key][:cut]), (name, i, key, cut)
        ids = [i for i in range(len(base)) if arm[i]["success"] and base[i]["success"]]
        support = [
            i for i in ids if arm[i]["support"] is not None and base[i]["support"] is not None
        ]
        measures = {}
        for key in (
            "forward_speed_std",
            "forward_mean_m_s",
            "roll_range_deg",
            "lateral_rms_m_s",
            "forward_reference_rmse",
        ):
            measures[key] = (
                [float(np.mean([x[i]["gait"][key] for i in ids])) for x in (base, arm)]
                if ids
                else []
            )
        for key in ("roll_range_deg", "roll_peak_deg", "onset_s"):
            measures["support_" + key] = (
                [float(np.mean([x[i]["support"][key] for i in support])) for x in (base, arm)]
                if support
                else []
            )
        stats[name] = {
            "cases": len(arm),
            "success": sum(x["success"] for x in arm),
            "fallen": sum(x["fallen"] for x in arm),
            "paired_walks": ids,
            "paired_supports": support,
            "new_failures": [
                i for i in range(len(base)) if base[i]["success"] and not arm[i]["success"]
            ],
            "new_success": [
                i for i in range(len(base)) if not base[i]["success"] and arm[i]["success"]
            ],
            "refalls": sum(x["refall"] for x in arm),
            "oob": sum(x["pd_oob"] for x in arm),
            "paired": measures,
        }
    return {"profiles": stats, "trials": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    a = ap.parse_args()
    (a.root / "assessment.json").write_text(json.dumps(compare(a.root), indent=2, allow_nan=False))
