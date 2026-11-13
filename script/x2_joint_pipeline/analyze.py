"""Read saved pipeline recordings and plot a comparison; never runs models."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from script.x2_fgp_realtime.assets import file_digest
from script.x2_joint_latency.analyze import load_lines
from script.x2_joint_latency.timing import statistics


def interval_overlap(a, b):
    a, b = sorted(a), sorted(b)
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1])-max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyze(source, baseline, output):
    if output.exists():
        raise ValueError("Preserve prior analysis")
    run = json.loads((source/"run_receipt.json").read_text())
    fgp = json.loads((source/"fgp_summary.json").read_text())
    config = json.loads((source/"pipeline_config.json").read_text())
    native = json.loads((source/"native/status.json").read_text())
    if (run["status"] != "JOINT_RECORDING_COMPLETED" or any(c["returncode"] for c in run["children"])
            or fgp["status"] != "FGP_REPLAY_COMPLETED" or fgp["in_flight_limit"] != 2
            or fgp["max_in_flight"] > 2 or fgp["dropped_after_denoising"]
            or fgp["submitted"] != fgp["completed"] or fgp["outputs"] != fgp["completed"]
            or native["hardware_commands"] or native["sdk_publish_calls"]
            or config["parity_receipt"]["status"] != "PIPELINE_PREFIX_EQUIVALENCE_PASSED"):
        raise ValueError("Incomplete pipeline or changed observe/ordering contract")
    rows = load_lines(source/"fgp.jsonl")
    outputs = [r for r in rows if r["output"]]
    policy = [r for r in load_lines(source/"native/native.jsonl") if r.get("kind") == "policy_sample"]
    ready = [r for r in policy if r["reply"].get("ready")]
    accepted = [r for r in policy if r["accepted"]]
    if len(outputs) != fgp["outputs"] or not ready:
        raise ValueError("Incomplete frames or no ready policy requests")
    for r in policy:
        if r["reply"].get("publish_allowed") is not False:
            raise ValueError("Non-advisory policy result")
    for r in ready:
        if r["reply"]["joint_test_timing"]["preview_span_ms"] != 180:
            raise ValueError("Preview changed")
    keys = ("gpu_stage_ms", "cpu_stage_ms", "cpu_queue_ms", "sender_queue_ms", "fgp_ms", "queue_ms",
            "bridge_ms", "human_rpc_ms", "software_input_age_at_ack_ms")
    stages = {key: statistics([r[key] for r in outputs]) for key in keys}
    age = [r["reply"]["joint_test_timing"]["oldest_reference_age_ms"] for r in ready]
    upper = [x+r["rpc_ms"] for x, r in zip(age, ready)]
    overlap = interval_overlap([(r["started_ns"], r["denoised_ns"]) for r in outputs],
                               [(r["cpu_started_ns"], r["inferred_ns"]) for r in outputs])
    reference = json.loads(baseline.read_text())
    summary = dict(status="PIPELINE_SAVED_ANALYSIS_PASSED", baseline=reference,
        wall_s=run["wall_s"], boot_id=run["boot_id"], input_frames=len(rows), outputs=len(outputs),
        skipped_input_frames=fgp["dropped_input_frames"], gpu_calls=fgp["gpu_calls"],
        human_hz=(len(outputs)-1)*1e9/(outputs[-1]["acknowledged_ns"]-outputs[0]["acknowledged_ns"]),
        output_intervals_ms=statistics(np.diff([r["acknowledged_ns"] for r in outputs])/1e6, 1000/30),
        native_requests=len(policy), native_ready=len(ready), native_accepted=len(accepted),
        refusals=dict(Counter(r["rejection_reason"] for r in policy if not r["accepted"])),
        rpc_ready_ms=statistics([r["rpc_ms"] for r in ready], 20.),
        rpc_all_ms=statistics([r["rpc_ms"] for r in policy], 20.),
        reference_age_lower_ms=statistics(age), reference_age_upper_ms=statistics(upper),
        stages_ms=stages, host_gpu_path_cpu_overlap_ms=overlap/1e6,
        overlap_limit="Host stage interval intersection; not isolated device-kernel telemetry",
        in_flight_maximum=fgp["max_in_flight"], dropped_after_denoising=0,
        min_available_bytes=min(r["available_bytes"] for r in run["resources"]),
        max_temperature_c=max(r["max_temperature_c"] for r in run["resources"]),
        preview_span_ms=180, hardware_commands=0, training_updates=0,
        physical_end_to_end_ms=None, accuracy_ground_truth_evaluated=False,
        human_input=run["human_input"], robot_feedback=run["robot_feedback"],
        limitation="Short ideal30Hz recorded IMU with live read-only feedback. Sequential comparison; skipped histories can differ. Not live BLE or mechanical latency or sustained stability certification.")
    output.mkdir(parents=True)
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    with (output/"frames.csv").open("x", newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["frame", *keys],extrasaction="ignore")
        writer.writeheader();writer.writerows(outputs)
    sources=[source/"run_receipt.json",source/"fgp_summary.json",source/"fgp.jsonl",
             source/"fgp_outputs.npz",source/"native/status.json",source/"native/native.jsonl",baseline]
    (output/"sources.json").write_text(json.dumps([dict(path=str(p),bytes=p.stat().st_size,sha256=file_digest(p))
                                                  for p in sources],indent=2)+"\n")
    return summary


def plot(saved, output):
    if output.exists():
        raise ValueError("Preserve prior figures")
    s=json.loads(saved.read_text());b=s["baseline"]
    if s["status"] != "PIPELINE_SAVED_ANALYSIS_PASSED":
        raise ValueError("Analysis gate not passed")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout="constrained")
    pairs=[("New human information", "Hz", b["fgp_output_wall_rate_hz"],s["human_hz"],30),
           ("Source to HumanFrame ACK p95", "ms",b["source_to_human_ack_ms"]["p95"],s["stages_ms"]["software_input_age_at_ack_ms"]["p95"],None),
           ("All-ready Morph RPC p95", "ms",b["rpc_ready_ms"]["p95"],s["rpc_ready_ms"]["p95"],20),
           ("Oldest reference age p95 upper", "ms",b["target_age_upper_ms"]["p95"],s["reference_age_upper_ms"]["p95"],180),
           ("Skipped raw input", "frames",b["dropped_input_frames"],s["skipped_input_frames"],None),
           ("Morph RPC deadline refusals", "requests",b["rpc_ready_ms"]["over_budget"],s["rpc_ready_ms"]["over_budget"],None)]
    for ax,(title,unit,old,new,limit) in zip(axes.flat,pairs):
        ax.bar(["Serial best","Pipeline R1"],[old,new],color=["#686f79","#197b68"])
        ax.set(title=title,ylabel=unit);ax.grid(axis="y",alpha=.2)
        if limit is not None:ax.axhline(limit,color="#ad3c37",ls="--")
    fig.suptitle("X2 SoC1 bounded pipeline | recorded IMU + live feedback | motor commands 0",fontsize=13)
    output.mkdir(parents=True)
    fig.savefig(output/"six_panel.png",dpi=160);fig.savefig(output/"six_panel.pdf");plt.close(fig)
    with (output/"comparison.csv").open("x",newline="") as f:
        w=csv.writer(f);w.writerow(["metric","unit","serial_best","pipeline_r1"])
        w.writerows(row[:4] for row in pairs)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--input",type=Path);p.add_argument("--baseline",type=Path)
    p.add_argument("--saved",type=Path);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.saved:
        if a.input or a.baseline:p.error("Use either saved plotting or input analysis")
        plot(a.saved,a.output)
    else:
        if not a.input or not a.baseline:p.error("Require input and baseline")
        s=analyze(a.input,a.baseline,a.output)
        print(json.dumps({k:v for k,v in s.items() if k!="baseline"}))


if __name__ == "__main__":
    main()
