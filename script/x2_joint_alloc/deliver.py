"""Plot verified saved R2 data; no robot connection or model execution."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--r1", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing figures")
    current, r1, review = load(args.analysis), load(args.r1), load(args.review)
    if (current["status"] != "PIPELINE_SAVED_ANALYSIS_PASSED"
            or r1["status"] != current["status"] or current["baseline"] != r1["baseline"]
            or review["status"] != "ALLOCATION_R2_SAVED_REVIEW_PASSED"
            or not review["source_integrity_equal"] or not review["all_role_kernel_masks_verified"]
            or current["hardware_commands"] or current["preview_span_ms"] != 180):
        raise ValueError("Require complete R2/source/kernel-mask review and identical historical baseline")
    monitor = load(args.monitor/"summary.json")
    rows = [json.loads(line) for line in (args.monitor/"samples.jsonl").read_text().splitlines()]
    if monitor["status"] != "TARGET_EXITED" or not rows or monitor["boot_id"] != current["boot_id"]:
        raise ValueError("Require the completed same-boot monitor")
    baseline = current["baseline"]
    metrics = [
        ("New human information (mean)","Hz",baseline["fgp_output_wall_rate_hz"],r1["human_hz"],current["human_hz"],30),
        ("Output interval p95","ms",baseline["fgp_output_interval_ms"]["p95"],
         r1["output_intervals_ms"]["p95"],current["output_intervals_ms"]["p95"],1000/30),
        ("All-ready Morph RPC p95","ms",baseline["rpc_ready_ms"]["p95"],r1["rpc_ready_ms"]["p95"],current["rpc_ready_ms"]["p95"],20),
        ("Ready RPC over 20 ms","%",100*baseline["rpc_ready_ms"]["over_budget"]/baseline["native_ready"],
         100*r1["rpc_ready_ms"]["over_budget"]/r1["native_ready"],100*current["rpc_ready_ms"]["over_budget"]/current["native_ready"],None),
        ("Source to HumanFrame ACK p95","ms",baseline["source_to_human_ack_ms"]["p95"],
         r1["stages_ms"]["software_input_age_at_ack_ms"]["p95"],current["stages_ms"]["software_input_age_at_ack_ms"]["p95"],None),
        ("Oldest reference p95 upper (includes preview)","ms",baseline["target_age_upper_ms"]["p95"],
         r1["reference_age_upper_ms"]["p95"],current["reference_age_upper_ms"]["p95"],180)]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    args.output.mkdir(parents=True)
    fig, axes = plt.subplots(2,3,figsize=(15,8),layout="constrained")
    for ax, (title,unit,serial,previous,value,target) in zip(axes.flat,metrics):
        values = [serial,previous,value]
        for index, (number,color) in enumerate(zip(values,("#737982","#b16535","#267c67"))):
            if number is not None:
                ax.bar(index,number,color=color)
                ax.annotate(format(number,".2f"),(index,number),ha="center",va="bottom",fontsize=9)
        ax.set_xticks([0,1,2],["Serial","Pipeline R1","Affinity R2"])
        ax.set(title=title,ylabel=unit)
        ax.grid(axis="y",alpha=.2)
        if target is not None:
            ax.axhline(target,color="#ad3c37",ls="--",lw=1)
    fig.suptitle("X2 saved short tests | historical controls on earlier boot | no motor commands",fontsize=13)
    fig.savefig(args.output/"six_panel.png",dpi=160)
    fig.savefig(args.output/"six_panel.pdf")
    plt.close(fig)
    with (args.output/"comparison.csv").open("x",newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric","unit","historical_serial","historical_pipeline_r1","affinity_r2","reference_line"])
        writer.writerows(metrics)
    start = rows[0]["monotonic_ns"]
    times = [(row["monotonic_ns"]-start)/1e9 for row in rows]
    tegra = [t for row in rows for t in row["tegrastats"]]
    fig, axes = plt.subplots(2,3,figsize=(15,8),layout="constrained")
    ax = axes[0,0]
    for cpu in sorted(k for k in rows[0]["cpu_busy_percent"] if k != "cpu"):
        ax.plot(times,[r["cpu_busy_percent"].get(cpu,float("nan")) for r in rows],label=cpu,lw=1)
    ax.set(title="Whole-host CPU by core",ylabel="%",ylim=(0,105))
    ax.legend(ncol=4,fontsize=8)
    valid = [t for t in tegra if t["gpu_busy_percent"] is not None]
    axes[0,1].plot([(t["received_monotonic_ns"]-start)/1e9 for t in valid],
                   [t["gpu_busy_percent"] for t in valid],color="#267c67")
    axes[0,1].set(title="Whole-device GPU (not per-process)",ylabel="%",ylim=(0,105))
    pids = sorted({p["pid"] for row in rows for p in row["owned_processes"]})
    for pid in pids:
        values = [next((p["cpu_percent_one_core"] for p in row["owned_processes"] if p["pid"]==pid),float("nan")) for row in rows]
        axes[0,2].plot(times,values,label=str(pid),lw=1)
    axes[0,2].set(title="Owned processes (100% = one core)",ylabel="%")
    axes[0,2].legend(fontsize=8,ncol=2)
    axes[1,0].plot(times,[max((v for t in r["tegrastats"] for v in t["temperature_c"].values()),default=float("nan")) for r in rows],color="#b16535")
    axes[1,0].set(title="Maximum reported sensor temperature",ylabel="C")
    axes[1,1].plot(times,[r["memory"]["MemAvailable_bytes"]/1024**3 for r in rows],color="#267c67")
    axes[1,1].set(title="Available shared RAM",ylabel="GiB")
    axes[1,2].plot(times,[r["sampler_read_wall_ms"] for r in rows],color="#737982")
    axes[1,2].set(title="Sampler read wall time (includes scheduling)",ylabel="ms")
    for ax in axes.flat:
        ax.set_xlabel("Seconds from first CPU sample")
        ax.grid(alpha=.2)
    fig.suptitle("R2 synchronized 1 Hz host counters | not individual RPC causal attribution",fontsize=13)
    fig.savefig(args.output/"resources_six_panel.png",dpi=160)
    fig.savefig(args.output/"resources_six_panel.pdf")
    plt.close(fig)
    inputs = [args.analysis,args.r1,args.review,args.monitor/"summary.json",args.monitor/"samples.jsonl"]
    (args.output/"sources.json").write_text(json.dumps([dict(path=str(p),bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs],indent=2)+"\n")
    print(json.dumps(dict(status="SAVED_R2_FIGURES_COMPLETED",outputs=current["outputs"],model_calls=0,
                         hardware_commands=0,output=str(args.output))))


if __name__ == "__main__":
    main()
