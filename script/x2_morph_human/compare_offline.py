"""Compare saved CPU replay outputs across hosts; no robot or policy imports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_run(root):
    summary = json.loads((root / "summary.json").read_text())
    rows = [json.loads(line) for line in (root / "targets.jsonl").read_text().splitlines() if line.strip()]
    holds_path = root / "hold_events.json"
    holds = json.loads(holds_path.read_text()) if holds_path.exists() else []
    if summary["robot_state_source"] != "SYNTHETIC_FIXED_DEFAULT_STATE" or summary["simulation_metrics"] is not None:
        raise ValueError("comparison requires offline fixed-state replays")
    if summary["hardware_commands"] != 0 or summary["publish_allowed"] is not False:
        raise ValueError("unexpected hardware/publishing declaration")
    if len(rows) != summary["frames"] or len(holds) != summary["midstream_hold_ticks"]:
        raise ValueError("saved output counts differ from summary")
    q = np.asarray([r["q_target_rad"] for r in rows], dtype=np.float64)
    if q.shape != (len(rows), 31) or not len(rows) or not np.isfinite(q).all():
        raise ValueError("invalid joint targets")
    for row in rows:
        if not row["ready"] or row["publish_allowed"] is not False:
            raise ValueError("unexpected target readiness/publishing")
    return summary, rows, holds, q


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--mac", required=True, type=Path)
    p.add_argument("--orin", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--atol-rad", type=float, default=2e-5)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()
    if not np.isfinite(args.atol_rad) or args.atol_rad <= 0:
        p.error("atol-rad must be finite and positive")
    left, a, ah, aq = read_run(args.mac)
    right, b, bh, bq = read_run(args.orin)
    for key in ("source", "policy_sha256", "frames", "startup_wait_packets", "midstream_hold_ticks", "max_interp_gap_ms", "human_points", "clock_50hz", "preview_span_ms"):
        if left[key] != right[key]:
            raise ValueError(f"different comparison condition: {key}")
    if ah != bh:
        raise ValueError("hold schedules differ")
    for x, y in zip(a, b):
        for key in ("state_sequence", "tick_index", "tick_ns", "joint_names", "timing"):
            if x[key] != y[key]:
                raise ValueError(f"saved frame contract differs: {key}")
    delta = np.abs(aq - bq)
    ok = bool(np.max(delta) <= args.atol_rad)
    summary = dict(status="OFFLINE_PARITY_PASSED" if ok else "OFFLINE_PARITY_FAILED",
                   compared_frames=len(a), joint_count=31, max_abs_target_error_rad=float(delta.max()),
                   tolerance_rad=args.atol_rad, identical_timing_and_hold_schedule=True,
                   startup_wait_ticks=left["startup_wait_packets"], hold_ticks=len(ah),
                   source_sha256=left["source"]["sha256"], policy_sha256=left["policy_sha256"],
                   mac_runtime_ms=left["full_runtime_ms"], orin_runtime_ms=right["full_runtime_ms"],
                   preview_span_ms=left["preview_span_ms"], hardware_end_to_end_latency_ms=None,
                   hardware_commands=0, fixed_synthetic_robot_feedback=True)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    np.savetxt(args.output / "frame_comparison.csv",
               np.column_stack(([r["tick_index"] for r in a], [r["full_runtime_ms"] for r in a],
                                [r["full_runtime_ms"] for r in b], delta.max(axis=1))),
               delimiter=",", header="tick,mac_runtime_ms,orin_runtime_ms,max_target_error_rad", comments="")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        ticks = np.asarray([r["tick_index"] for r in a])
        ams = np.asarray([r["full_runtime_ms"] for r in a])
        bms = np.asarray([r["full_runtime_ms"] for r in b])
        for ax, values, title, color in ((axes[0, 0], ams, "Mac CPU processing", "#087f8c"),
                                          (axes[0, 1], bms, "Orin CPU processing", "#bc4b45")):
            ax.plot(ticks, values, color=color, lw=.8)
            ax.axhline(20, color="#555555", ls="--", lw=1)
            ax.set(xlabel="Replay tick", ylabel="ms", title=title)
        ax = axes[0, 2]
        qtiles = [50, 95, 99]
        x = np.arange(3)
        ax.bar(x-.18, np.percentile(ams, qtiles), .36, label="Mac", color="#087f8c")
        ax.bar(x+.18, np.percentile(bms, qtiles), .36, label="Orin", color="#bc4b45")
        ax.set(xticks=x, xticklabels=["p50", "p95", "p99"], ylabel="ms", title="Processing quantiles")
        ax.legend()
        ax = axes[1, 0]
        ready = np.ones(int(ticks[-1]) + 1)
        ready[:int(ticks[0])] = 0
        for h in ah:
            ready[int(h["tick_index"])] = 0
        ax.step(np.arange(len(ready)), ready, where="post", color="#667545")
        ax.set(xlabel="Replay tick", ylabel="Ready (both hosts)", title="Matched readiness and holds", yticks=[0, 1], ylim=(-.1, 1.1))
        axes[1, 1].plot(ticks, delta.max(axis=1), color="#73526d", lw=.8)
        axes[1, 1].set(xlabel="Replay tick", ylabel="rad", title="Maximum target difference per frame")
        axes[1, 2].bar(np.arange(31), delta.max(axis=0), color="#087f8c")
        axes[1, 2].set(xlabel="Native joint index", ylabel="rad", title="Maximum target difference per joint")
        fig.suptitle("X2 Orin basic preflight: saved human input and fixed synthetic robot feedback", fontsize=14)
        fig.savefig(args.output / "orin_offline_parity_6panel.png", dpi=150)
        fig.savefig(args.output / "orin_offline_parity_6panel.pdf")
        plt.close(fig)
    print(json.dumps(summary, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
