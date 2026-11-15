"""Inspect saved output jitter and a fixed-delay feasibility bound; no model calls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from script.x2_joint_latency.timing import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing diagnostics")
    source = args.source / "fgp.jsonl"
    inputs = [source, args.analysis, args.review]
    before = {str(p): digest(p) for p in inputs}
    summary = json.loads(args.analysis.read_text())
    review = json.loads(args.review.read_text())
    if (summary["status"] != "PIPELINE_SAVED_ANALYSIS_PASSED"
            or review["status"] != "ALLOCATION_R2_SAVED_REVIEW_PASSED"
            or not review["source_integrity_equal"]
            or not review["all_role_kernel_masks_verified"]):
        raise ValueError("Require completed saved analysis and source/kernel review")
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows = [row for row in rows if row["output"]]
    if len(rows) != summary["outputs"] or summary["skipped_input_frames"]:
        raise ValueError("This diagnostic requires the full consecutive-output cohort")
    frame = np.asarray([row["frame"] for row in rows], dtype=np.int64)
    capture = np.asarray([row["capture_ns"] for row in rows], dtype=np.int64)
    ack = np.asarray([row["acknowledged_ns"] for row in rows], dtype=np.int64)
    if not (np.all(np.diff(frame) == 1) and np.all(np.diff(capture) > 0)
            and np.all(np.diff(ack) > 0) and np.all(ack >= capture)):
        raise ValueError("Nonconsecutive or invalid saved clocks")
    age_ns, interval_ns = ack-capture, np.diff(ack)
    if not np.array_equal(interval_ns, np.diff(capture)+np.diff(age_ns)):
        raise ValueError("Source/output age accounting failed")
    interval, age = interval_ns/1e6, age_ns/1e6
    long = interval > 1000/30
    stage_keys = ("gpu_stage_ms", "cpu_stage_ms", "cpu_queue_ms",
                  "sender_queue_ms", "human_rpc_ms", "bridge_ms")
    by_gap = {}
    for key in stage_keys:
        values = np.asarray([row[key] for row in rows])[1:]
        by_gap[key] = dict(long_gap=statistics(values[long]),
                           other_gap=statistics(values[~long]))
    feasibility = []
    for delay in (50., 60., 65., 70., 75.):
        deadline_ns = capture+int(delay*1e6)
        late = ack > deadline_ns
        feasibility.append(dict(fixed_delay_ms=delay, outputs=len(rows),
            ready_by_deadline=int(np.sum(~late)), missed=int(np.sum(late)),
            additional_wait_on_ready_ms=statistics((deadline_ns[~late]-ack[~late])/1e6)))
    result = dict(status="SAVED_OUTPUT_JITTER_REVIEW_PASSED", outputs=len(rows),
        intervals=statistics(interval, 1000/30), source_interval_ms=statistics(np.diff(capture)/1e6),
        software_age_ms=statistics(age), age_delta_ms=statistics(np.diff(age)),
        exact_nanosecond_age_identity=True, long_gap_count=int(long.sum()),
        long_gap_followed_by_short=int(np.sum(long[:-1] & ~long[1:])),
        comparison_scope="Current output stages grouped by preceding ACK gap; descriptive association, not causal attribution",
        stages_by_gap=by_gap, fixed_delay_feasibility=feasibility,
        fixed_delay_limit="Only saved original-ready versus ideal source-clock deadlines. No timer, interpolation, deployment, new pose or feedback tested. Missing deadlines require an explicit fallback; added waiting cannot remove the 180ms preview",
        new_model_calls=0, hardware_commands=0, training_updates=0, source_sha256=before)
    if before != {str(p): digest(p) for p in inputs}:
        raise ValueError("Saved sources changed during review")
    args.output.mkdir(parents=True)
    (args.output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    with (args.output/"frames.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "output_gap_ms", "source_age_ms", *stage_keys])
        for index, row in enumerate(rows):
            writer.writerow([row["frame"], None if index == 0 else interval[index-1],
                             age[index], *(row[key] for key in stage_keys)])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    time = (ack-ack[0])/1e9
    axes[0, 0].plot(time[1:], interval, lw=.6, color="#267c67")
    axes[0, 0].axhline(1000/30, ls="--", color="#ad3c37", lw=1)
    axes[0, 0].set(title="ACK intervals (899 gaps)", xlabel="Seconds", ylabel="ms")
    axes[0, 1].hist(interval, bins=35, color="#737982")
    axes[0, 1].axvline(1000/30, ls="--", color="#ad3c37", lw=1)
    axes[0, 1].set(title="Output interval distribution", xlabel="ms", ylabel="Count")
    axes[0, 2].plot(time, age, color="#267c67", lw=.8)
    axes[0, 2].set(title="Source to ACK age (no preview)", xlabel="Seconds", ylabel="ms")
    for key, label in (("gpu_stage_ms", "GPU host stage"), ("cpu_stage_ms", "CPU stage"),
                       ("human_rpc_ms", "Human RPC")):
        axes[1, 0].plot(time, [row[key] for row in rows], label=label, lw=.6)
    axes[1, 0].set(title="Concurrent stage wall times (do not sum)", xlabel="Seconds", ylabel="ms")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].scatter(np.diff(age), interval, s=3, color="#737982", alpha=.5)
    axes[1, 1].set(title="Clock identity, not a causal model", xlabel="Age change from previous output (ms)", ylabel="ACK gap (ms)")
    axes[1, 2].bar([str(int(x["fixed_delay_ms"])) for x in feasibility],
                   [x["missed"] for x in feasibility], color="#b16535")
    axes[1, 2].set(title="Fixed source-delay readiness (offline only)", xlabel="Source to release delay (ms)", ylabel="Missed / 900")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle("R2 saved jitter | no new inference | uniform release has an added-wait tradeoff", fontsize=13)
    fig.savefig(args.output/"six_panel.png", dpi=160)
    fig.savefig(args.output/"six_panel.pdf")
    plt.close(fig)
    print(json.dumps({k: result[k] for k in ("status", "outputs", "long_gap_count", "long_gap_followed_by_short", "fixed_delay_feasibility")}))


if __name__ == "__main__":
    main()
