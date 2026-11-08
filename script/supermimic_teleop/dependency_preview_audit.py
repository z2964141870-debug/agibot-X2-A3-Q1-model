"""Metadata-only raw-support timing audit; no poses, policies or physics are read."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import resource
import sys
import time

from dependency_preview import HORIZONS
from forecast_data import identity, source_support
from stream_contract import DT_NS


SPLIT_SHA256 = "f40c1d5a856a08d89f1a8c23dfe6d990fc7439232b9da9a658d9860456353145"


def phase_counts(frames):
    if type(frames) is not int or frames < 11:
        raise ValueError("Require at least one complete ten-target reference window")
    anchors = frames - 10
    return {phase: max(0, (anchors + 4 - phase) // 5) for phase in range(5)}


def prefix_at_phase(phase, horizon):
    if type(phase) is not int or not 0 <= phase < 5 or horizon not in HORIZONS:
        raise ValueError("Require a raw-support cycle phase and known horizon")
    control = (phase + horizon) * DT_NS
    count = 0
    for target in range(phase + 1, phase + horizon + 1):
        if source_support(target)["dependency_end_ns"] > control:
            break
        count += 1
    return count


def run(split_plan, output):
    started = time.perf_counter()
    split_plan, output = Path(split_plan).resolve(strict=True), Path(output)
    before = identity(split_plan)
    if before["sha256"] != SPLIT_SHA256 or output.exists():
        raise ValueError("Require the pinned whole-source split and a fresh output")
    plan = json.loads(split_plan.read_text())
    clips = plan["clips"]
    if (len(clips) != 393 or len({row["relative"] for row in clips}) != 393
            or plan["fps"] != 50 or plan["future_frames"] != 10
            or plan["preprocessing"]["type"] != "offline_120_to_50_linear_slerp"):
        raise ValueError("Unexpected frozen source or resampling metadata")
    groups = {}
    for split in ("train", "validation", "test"):
        rows = [row for row in clips if row["split"] == split]
        counts = Counter()
        for row in rows:
            counts.update(phase_counts(row["frames"]))
        groups[split] = dict(clips=len(rows), anchors=sum(counts.values()), phases=dict(counts))
    if [groups[key]["clips"] for key in ("train", "validation", "test")] != [167, 98, 128]:
        raise ValueError("Source split changed")
    total = Counter()
    for group in groups.values():
        total.update(group["phases"])
    phase_rows, summary = [], []
    for horizon in HORIZONS:
        histogram, delayed = Counter(), 0
        for phase in range(5):
            count = prefix_at_phase(phase, horizon)
            wait = source_support(phase + horizon)["right_support_ns"]
            phase_rows.append(dict(horizon_frames=horizon, anchor_phase_mod5=phase,
                scheduled_wait_ns=horizon * DT_NS, full_prefix_raw_extra_wait_ns=wait,
                ready_prefix_frames=count, generated_tail_frames=10-count,
                corpus_anchor_count=total[phase]))
            histogram[count] += total[phase]
            delayed += total[phase] if wait else 0
        summary.append(dict(horizon_frames=horizon, scheduled_wait_ns=horizon * DT_NS,
            full_requested_prefix_exceeds_cutoff_anchors=delayed,
            total_anchors=sum(total.values()), ready_prefix_histogram=dict(histogram),
            maximum_full_prefix_extra_wait_ns=max(row["full_prefix_raw_extra_wait_ns"]
                for row in phase_rows if row["horizon_frames"] == horizon)))
    elapsed = time.perf_counter()-started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
    if elapsed >= 30 or peak_bytes >= 256 * 1024**2 or identity(split_plan) != before:
        raise RuntimeError("Metadata audit exceeded its bound or its source changed")
    code = [Path(__file__), Path(__file__).with_name("dependency_preview.py"),
            Path(__file__).with_name("forecast_data.py"),
            Path(__file__).with_name("student_data.py"),
            Path(__file__).with_name("student_inputs.py"),
            Path(__file__).with_name("causal_tail.py"),
            Path(__file__).with_name("stream_contract.py")]
    result = dict(stage="FROZEN_METADATA_RESAMPLING_TIMING_AUDIT",
        status="SAVED_RESAMPLING_TIMING_AUDIT_PASSED", split_plan=before,
        source_inputs_unchanged=True, code={path.name: identity(path) for path in code},
        groups=groups, horizons=summary, phase_rows=phase_rows,
        anchor_range="frame0 through frames-11; all ten labels remain inside the clip",
        source_body_read=False, human_poses_loaded=0, model_calls=0, physics_steps=0,
        optimizer_updates=0, server_gpu_seconds=0,
        cost_ledger_last_confirmed_seconds=5036, cost_ledger_modified=False,
        test_outcomes_opened=False, source_metadata_only=True,
        sensor_clock_or_upstream_causality_verified=False,
        measured_end_to_end_latency_ns=None, prior84_quality_reinterpreted=False,
        receiver_gpu_action_parity="PENDING", receiver_physical_evaluation="PENDING",
        elapsed_seconds=elapsed, peak_rss_bytes=peak_bytes,
        limitations=["Pinned stored interpolation arithmetic, not measured capture or hardware delay",
            "Legacy84 did not record raw support; no claim that it read a future physical human pose",
            "Generated tails remain predictions; no new action, stability or naturalness result",
            "Old test clips are exploratory; source metadata do not open held-out event outcomes"],
        original_shared_training_resumed=False, user_accepted_checkpoint=None)
    output.mkdir(parents=True, exist_ok=False)
    with (output/"phases.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(phase_rows[0]))
        writer.writeheader()
        writer.writerows(phase_rows)
    with (output/"summary.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(summary=identity(output/"summary.json"),
        csv=identity(output/"phases.csv"), elapsed_seconds=elapsed, peak_rss_bytes=peak_bytes)))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--split-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.split_plan, args.output)
