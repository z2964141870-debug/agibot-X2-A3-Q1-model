"""Verify archived controls, nominal bypass, repeat pushes, and paired gait criteria."""

import json
from pathlib import Path

import numpy as np

PROFILES = ("published", "fast", "damped_moderate")
FIELDS = ("qpos", "reference", "trace", "loads")


def read(path):
    return json.loads(path.read_text())


def identical(first, second, fields):
    with np.load(first) as a, np.load(second) as b:
        for field in fields:
            if not np.array_equal(a[field], b[field]):
                raise ValueError(f"{first} differs from {second} in {field}")
        return len(a["trace"])


def review(root):
    static = root / "static"
    old = root / "previous_study"
    for i in range(24):
        name = f"trial_{i:02}.npz"
        identical(static / name, root / "formal_03/pinned_e1" / name, ("qpos", "trace", "initial"))
    static_rows = read(static / "trials.json")
    assert static_rows == read(root / "formal_03/pinned_e1/trials.json")

    for i in range(25):
        path = "relay01" if i < 5 else "blind01"
        file = f"relay_{i if i < 5 else i - 5}.npz"
        for profile, archive in (("published", "original"), ("fast", "ramp15")):
            source = old / path / archive / file
            if not source.exists() and profile == "fast":
                # The fork runner saves no candidate when there is no reanchor.
                with np.load(root / f"dev/{profile}/relay_{i}.npz") as candidate:
                    assert len(candidate["reanchors"]) == 0
                source = old / path / "original" / file
            identical(root / f"dev/{profile}/relay_{i}.npz", source, FIELDS)

    report = {
        "static_cases": len(static_rows),
        "static_pass": sum(x["success"] for x in static_rows),
        "archive_dynamic_matches_per_control": 25,
        "profiles": {},
    }
    summary = read(root / "confirm/assessment.json")
    common = [i for i in range(30) if all(summary["trials"][p][i]["success"] for p in PROFILES)]
    report["common_walk_ids"] = common
    for profile in PROFILES:
        flat_ticks = 0
        for i in range(2):
            file = f"relay_{i}.npz"
            dest = root / "flat" / profile / file
            flat_ticks += identical(dest, old / "nominal01/original" / file, FIELDS)
            with np.load(dest) as d:
                assert not np.any(d["trace"][:, 5]) and not np.any(d["ankle_offset"])
        if profile != "fast":
            for i in range(5):
                file = f"relay_{i}.npz"
                identical(
                    root / "direct" / profile / file,
                    root / "dev" / profile / file,
                    FIELDS + ("ankle_offset",),
                )
        repeat_rows = read(root / "repeat_checked" / profile / "relay.json")
        assert len(repeat_rows) == 5
        vals = summary["profiles"][profile]
        metrics = {}
        for item, field in (
            ("gait", "forward_speed_std"),
            ("gait", "forward_mean_m_s"),
            ("gait", "roll_range_deg"),
            ("gait", "lateral_rms_m_s"),
            ("support", "roll_peak_deg"),
        ):
            metrics[field] = float(
                np.mean([summary["trials"][profile][i][item][field] for i in common])
            )
        candidate = summary["trials"][profile]
        refalls = [i for i, t in enumerate(candidate) if t["refall"]]
        base_refalls = {i for i, t in enumerate(summary["trials"]["published"]) if t["refall"]}
        p = vals["paired"]
        bounds = all(x["peak_offset_rad"] <= 0.03000001 for x in candidate)
        criteria = dict(
            preserve_passes=not vals["new_failures"],
            no_new_takeovers_after_release=not set(refalls) - base_refalls,
            safe_pd=vals["oob"] == 0,
            ankle_within_bound=bounds,
            reduce_forward_variation=p["forward_speed_std"][1] < p["forward_speed_std"][0],
            first_step_peak_no_worse=p["support_roll_peak_deg"][1] <= p["support_roll_peak_deg"][0],
            three_second_roll_no_worse=p["roll_range_deg"][1] <= p["roll_range_deg"][0],
            lateral_rms_no_worse=p["lateral_rms_m_s"][1] <= p["lateral_rms_m_s"][0],
            repeated_pushes_all_pass=all(x["success"] for x in repeat_rows),
        )
        report["profiles"][profile] = dict(
            criteria=criteria,
            decision=(
                "baseline"
                if profile == "published"
                else "eligible_in_sim"
                if all(criteria.values())
                else "hold"
            ),
            common_metrics=metrics,
            refall_ids=refalls,
            nominal_ticks_exact=flat_ticks,
            cached_reference_direct_replays=5 if profile != "fast" else 0,
            repeat=repeat_rows,
        )
    return report


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path, default=Path.cwd(), nargs="?")
    a = p.parse_args()
    (a.root / "regression.json").write_text(json.dumps(review(a.root), indent=2, allow_nan=False))
