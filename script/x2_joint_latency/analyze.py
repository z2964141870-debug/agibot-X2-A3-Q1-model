"""Read-only joined timing analysis; never invokes either model or hardware."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np

from script.x2_fgp_realtime.assets import file_digest
from .timing import statistics


def load_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Preserve previous analysis")
    receipt = json.loads((a.input/"run_receipt.json").read_text())
    fgp_summary = json.loads((a.input/"fgp_summary.json").read_text())
    native_status = json.loads((a.input/"native/status.json").read_text())
    if (receipt["status"] != "JOINT_RECORDING_COMPLETED" or not 1 <= fgp_summary["outputs"] <= 900
            or any(native_status[k] for k in ("hardware_commands", "sdk_publish_calls"))):
        raise ValueError("Joint completion or observe invariant failed")
    fgp = load_lines(a.input/"fgp.jsonl")
    outputs = [r for r in fgp if r["output"]]
    samples = [r for r in load_lines(a.input/"native/native.jsonl") if r.get("kind") == "policy_sample"]
    ready = [r for r in samples if r["reply"].get("ready")]
    accepted = [r for r in samples if r["accepted"]]
    if len(outputs) != fgp_summary["outputs"] or not ready:
        raise ValueError("Incomplete FGP or no real policy result")
    for r in samples:
        if r["reply"].get("publish_allowed") is not False:
            raise ValueError("Not an advisory reply")
    for r in ready:
        if r["reply"]["joint_test_timing"]["preview_span_ms"] != 180:
            raise ValueError("Preview span changed")
    first_due = fgp[0]["capture_ns"]
    rows = []
    for r in ready:
        t = r["reply"]["joint_test_timing"]
        rows.append(dict(sequence=r["state"]["sequence"], accepted=r["accepted"],
            seconds=(t["reply_ready_ns"]-first_due)/1e9,
            rpc_ms=r["rpc_ms"], runtime_ms=r["reply"]["full_runtime_ms"],
            target_age_lower_ms=t["oldest_reference_age_ms"],
            target_age_upper_ms=t["oldest_reference_age_ms"]+r["rpc_ms"],
            newest_support_age_ms=t["newest_support_age_ms"],
            newest_slot_age_ms=t["newest_slot_age_ms"]))
    resources = receipt["resources"]
    summary = dict(status="JOINT_SAVED_ANALYSIS_PASSED", recording_wall_s=receipt["wall_s"],
        input_frames=len(fgp), fgp_outputs=len(outputs), native_requests=len(samples),
        scheduling=fgp_summary["queue_discipline"],
        dropped_input_frames=fgp_summary["dropped_input_frames"],
        training_rate_matches_processed_rate=fgp_summary.get("training_rate_matches_processed_rate", True),
        native_ready=len(ready), native_accepted=len(accepted),
        native_rejection_reasons=dict(Counter(r["rejection_reason"] for r in samples if not r["accepted"])),
        rpc_all_ms=statistics([r["rpc_ms"] for r in samples], 20.),
        rpc_ready_ms=statistics([r["rpc_ms"] for r in ready], 20.),
        rpc_accepted_ms=statistics([r["rpc_ms"] for r in accepted], 20.),
        policy_runtime_ms=statistics([r["reply"]["full_runtime_ms"] for r in ready], 20.),
        fgp_ms=statistics([r["fgp_ms"] for r in outputs], 1000/30),
        fgp_queue_ms=statistics([r["queue_ms"] for r in outputs]),
        bridge_ms=statistics([r["bridge_ms"] for r in outputs]),
        human_rpc_ms=statistics([r["human_rpc_ms"] for r in outputs]),
        source_to_human_ack_ms=statistics([r["software_input_age_at_ack_ms"] for r in outputs]),
        target_age_lower_ms=statistics([r["target_age_lower_ms"] for r in rows]),
        target_age_upper_ms=statistics([r["target_age_upper_ms"] for r in rows]),
        newest_support_age_ms=statistics([r["newest_support_age_ms"] for r in rows]),
        target_age_accepted_lower_ms=statistics([r["target_age_lower_ms"] for r in rows if r["accepted"]]),
        fgp_output_wall_rate_hz=(len(outputs)-1)*1e9/(outputs[-1]["acknowledged_ns"]-outputs[0]["acknowledged_ns"]),
        fgp_output_interval_ms=statistics(np.diff([r["acknowledged_ns"] for r in outputs])/1e6, 1000/30),
        ready_policy_wall_rate_hz=(len(rows)-1)/(rows[-1]["seconds"]-rows[0]["seconds"]),
        ready_policy_interval_ms=statistics(np.diff([r["seconds"] for r in rows])*1000, 20.),
        accepted_policy_intervals_ms=statistics(np.diff([r["seconds"] for r in rows if r["accepted"]])*1000, 20.),
        first_policy_output_from_first_source_ms=(ready[0]["reply"]["joint_test_timing"]["reply_ready_ns"]-first_due)/1e6,
        preview_span_ms=180, full_synthetic_clock_is_hardware_latency=False,
        fgp_first_last_queue_ms=[outputs[0]["queue_ms"], outputs[-1]["queue_ms"]],
        fgp_first_last_source_age_ms=[outputs[0]["software_input_age_at_ack_ms"], outputs[-1]["software_input_age_at_ack_ms"]],
        min_available_bytes=min(r["available_bytes"] for r in resources),
        max_temperature_c=max(r["max_temperature_c"] for r in resources),
        boot_id=receipt["boot_id"], native_status=native_status,
        human_input=receipt["human_input"], robot_feedback=receipt["robot_feedback"],
        source_clock="Local replay release, not original sensor sampling clock",
        backend_age_bound="Service reply-ready age <= backend receive age <= service age + full native RPC; conservative upper bound double counts part of computation",
        hardware_end_to_end_ms=None, ble_radio_ms=None, mechanical_response_ms=None,
        hardware_commands=0, mc_mutations=0, new_model_calls=0,
        limitation="About 30s ideal-30Hz recorded IMU, live read-only robot feedback. No BLE jitter or physical phase measurement. Latest scheduling skips raw samples when overloaded; the unchanged FGP recurrent/physics rate is still 30Hz, so pose quality under skipped samples is unverified.")
    a.output.mkdir(parents=True, exist_ok=False)
    with (a.output/"policy_timing.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (a.output/"fgp_timing.csv").open("x", newline="") as stream:
        keys = ["frame", "capture_ns", "queue_ms", "fgp_ms", "bridge_ms", "human_rpc_ms", "software_input_age_at_ack_ms"]
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader(); writer.writerows(outputs)
    summary["source_files"] = [dict(path=str(path), bytes=path.stat().st_size, sha256=file_digest(path))
        for path in sorted(a.input.rglob('*')) if path.is_file()]
    (a.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    seconds = [(r["capture_ns"]-first_due)/1e9 for r in outputs]
    axes[0, 0].plot(seconds, [r["fgp_ms"] for r in outputs], lw=.8)
    axes[0, 0].axhline(1000/30, color="#a33c32", ls="--")
    axes[0, 0].set(title="FGP full pipeline (GPU + CPU)", ylabel="ms")
    axes[0, 1].plot(seconds, [r["queue_ms"] for r in outputs], label="Input wait / sample age", lw=.9)
    axes[0, 1].plot(seconds, [r["software_input_age_at_ack_ms"] for r in outputs], label="HumanFrame acknowledged", lw=.9)
    axes[0, 1].set(title="Source age and backlog", ylabel="ms"); axes[0, 1].legend()
    axes[0, 2].plot([r["seconds"] for r in rows], [r["rpc_ms"] for r in rows], lw=.8)
    axes[0, 2].axhline(20, color="#a33c32", ls="--")
    axes[0, 2].set(title="Morph real-feedback RPC (all ready)", ylabel="ms")
    x = [r["seconds"] for r in rows]
    axes[1, 0].fill_between(x, [r["target_age_lower_ms"] for r in rows],
                         [r["target_age_upper_ms"] for r in rows], color="#22866b", alpha=.5,
                         label="Backend receive conservative bounds")
    axes[1, 0].axhline(180, color="#555555", ls="--", label="Preview span only")
    axes[1, 0].set(title="Oldest human reference to target", ylabel="ms"); axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(x, [r["newest_support_age_ms"] for r in rows], lw=.8)
    axes[1, 1].set(title="Newest raw support age at target", ylabel="ms")
    reasons = Counter("accepted" if r["accepted"] else r["rejection_reason"] for r in samples)
    axes[1, 2].barh(list(reasons), list(reasons.values()), color=["#22866b" if k == "accepted" else "#a9473b" for k in reasons])
    axes[1, 2].set(title="Native observe acceptance (no publishing)", xlabel="Requests")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for ax in list(axes[0])+list(axes[1, :2]):
        ax.set_xlabel("Replay wall time (s)")
    for ax in axes[0, :2]:
        ax.set_xlabel("Source release schedule (s)")
    fig.suptitle("X2 SoC1 joint replay: target FGP 30Hz + Morph 50Hz | real feedback | motor commands 0", fontsize=13)
    fig.savefig(a.output/"six_panel.png", dpi=160)
    fig.savefig(a.output/"six_panel.pdf")
    plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k not in ("source_files", "native_status")}))


if __name__ == "__main__":
    main()
