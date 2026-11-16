"""Locate the stage increases responsible for gaps between archived FGP outputs."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

from .review_cano_solver_tail import matcher, read_jsonl


def stage_match(path, stage):
    return matcher([row for row in read_jsonl(path) if row["stage"] == stage])


def review(path):
    host = path / "profile.jsonl"
    child = path / "cpu_child/profile.jsonl"
    calibration = stage_match(host, "imu_calibration")
    cano = stage_match(host, "cano_session")
    native = stage_match(child, "native_solver")
    frames = []
    for row in read_jsonl(path / "fgp.jsonl"):
        if not row["output"]:
            continue
        due, started, denoised, cpu, finished, ack = [row[key] for key in (
            "capture_ns", "started_ns", "denoised_ns", "cpu_started_ns", "inferred_ns", "acknowledged_ns")]
        cal = calibration(started, denoised, "calibration")
        cuda = cano(started, denoised, "CANO")
        solver = native(cpu, finished, "native solve")
        cal_ns, cuda_ns, solve_ns = [span["ended_ns"] - span["began_ns"] for span in (cal, cuda, solver)]
        stages = dict(start_lateness=started - due, imu_calibration=cal_ns, cano_host=cuda_ns,
                      gpu_host_other=denoised - started - cal_ns - cuda_ns,
                      cpu_queue=cpu - denoised, native_physics=solve_ns, cpu_other=finished - cpu - solve_ns)
        if sum(stages.values()) != finished - due or min(stages.values()) < 0:
            raise ValueError("stage decomposition failed")
        coarse = dict(start_lateness=started-due, gpu_host=denoised-started, cpu_queue=cpu-denoised,
                      cpu_compute=finished-cpu)
        frames.append(dict(row=row, stages=stages, coarse=coarse, due=due, finished=finished, ack=ack))
    gaps = []
    for a, b in zip(frames, frames[1:]):
        if b["row"]["frame"] != a["row"]["frame"] + 1:
            raise ValueError("comparison requires consecutive source frames")
        gap = b["finished"] - a["finished"]
        increments = {k: v - a["stages"][k] for k, v in b["stages"].items()}
        coarse = {k: v - a["coarse"][k] for k, v in b["coarse"].items()}
        if gap != b["due"] - a["due"] + sum(increments.values()):
            raise ValueError("gap decomposition failed")
        gaps.append(dict(frame=b["row"]["frame"], output_gap_ms=gap / 1e6,
                         ack_gap_ms=(b["ack"] - a["ack"]) / 1e6,
                         later_frame_credit_reserve_ms=b["row"]["reserve_ns"] / 1e6,
                         start_lateness_ms=b["stages"]["start_lateness"] / 1e6,
                         detail_largest=max(increments, key=increments.get),
                         coarse_largest=max(coarse, key=coarse.get),
                         changes_ms={key: value / 1e6 for key, value in increments.items()}))
    for gap, following in zip(gaps, gaps[1:]):
        gap["next_gap_ms"] = following["output_gap_ms"]
    slow = [g for g in gaps if g["output_gap_ms"] >= 40.0]
    return dict(intervals=len(gaps), at_least_40_ms=len(slow),
                at_least_40_ms_followed_by_under_30_ms=sum(g.get("next_gap_ms", 30) < 30 for g in slow),
                under_30_ms=sum(g["output_gap_ms"] < 30 for g in gaps),
                slow_largest_stage_change=dict(Counter(g["coarse_largest"] for g in slow)),
                slow_largest_detailed_change=dict(Counter(g["detail_largest"] for g in slow)),
                slow_reserve_ms_median=median(g["later_frame_credit_reserve_ms"] for g in slow),
                all_reserve_ms_median=median(g["later_frame_credit_reserve_ms"] for g in gaps),
                worst=sorted(gaps, key=lambda g: g["output_gap_ms"], reverse=True)[:3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rounds", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps({path.name: review(path) for path in args.rounds}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
