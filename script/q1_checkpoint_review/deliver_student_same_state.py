"""Draw prefix-only same-state residuals and instrumented f0 software timing."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

VARIANTS = ("coverage_f0", "original_f0", "teacher_h4", "teacher_h10")
LABELS = ("New f0 / H4", "New f0 / H10", "Old f0 / H4", "Old f0 / H10")
COLORS = ("#197d85", "#924462", "#197d85", "#924462")


def read(path):
    return json.loads(path.read_text())


def deliver(source, output):
    if not source.is_absolute() or not output.is_absolute() or output.exists() or output.is_relative_to(source):
        raise ValueError("Require saved originals and a new figure directory")
    result, comparison, audit = read(source/"source_result.json"),read(source/"summary.json"),read(source/"audit.json")
    raw=(source/"source_result.json").read_bytes()
    original=comparison.get("original_result",{})
    if (comparison.get("stage") != "SAME_STATE_SAVED_ANALYSIS" or comparison.get("status") != "PASSED"
            or comparison.get("source_integrity") != "PASSED" or comparison.get("original_campaign_status") != "FAILED"
            or result.get("status") != "FAILED" or len(result.get("phases",[])) != 8
            or comparison.get("reused_policy_calls") != 936 or comparison.get("new_policy_calls") != 0
            or comparison.get("new_physical_steps") != 0 or comparison.get("new_optimizer_updates") != 0
            or result.get("cost_settlement",{}).get("state") != "SETTLED"
            or comparison.get("privileged_future_for_offline_reference_only") is not True
            or audit.get("status") != "SAME_STATE_SAVED_ANALYSIS_PASSED" or audit.get("source_inputs_unchanged") is not True
            or len(raw) != original.get("bytes") or hashlib.sha256(raw).hexdigest() != original.get("sha256")):
        raise ValueError("Require independent saved936 analysis while preserving the original FAILED root")
    files = [source/"source_result.json",source/"summary.json",source/"audit.json"]
    if len(comparison.get("phase_guards",[])) != 8 or any(
            g.get("status") != "WORKER_COMPLETED" or g.get("cleanup",{}).get("worker_reaped") is not True
            or g["cleanup"].get("worker_return_code") != 0 for g in comparison["phase_guards"]):
        raise ValueError("Incomplete or unreaped actual936 phase")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    table = []
    for case in comparison["comparisons"]:
        clip = case["clip"]
        summaries = comparison["software"][clip]
        if any(s["calls"] != case["prefix_calls"] or s.get("status") != "SAME_STATE_CALLS_PASSED" for s in summaries.values()):
            raise ValueError("Different recorded prefixes in the comparison")
        rows = case["residuals"]
        def lookup(student, teacher, window):
            found = [r for r in rows if (r["student"],r["privileged_reference"],r["window"]) == (student,teacher,window)]
            if len(found) != 1:raise ValueError("Missing or duplicate residual")
            return found[0]
        ordered = [lookup(s,t,"all_prefix") for s in VARIANTS[:2] for t in VARIANTS[2:]]
        fig, axes = plt.subplots(2,3,figsize=(16,9.4),constrained_layout=True)
        x = np.arange(4)
        axes[0,0].bar(x,[r["rmse_by_group"]["present"] for r in ordered],color=COLORS)
        axes[0,0].set(title="Same-state action RMSE: all present joints",xticks=x,xticklabels=LABELS)
        for shift,key,label,color in ((-.17,"upper","Upper","#197d85"),(.17,"other_present","Other present","#b36b31")):
            axes[0,1].bar(x+shift,[r["rmse_by_group"][key] for r in ordered],width=.32,label=label,color=color)
        axes[0,1].set(title="Upper / other present action residual",xticks=x,xticklabels=LABELS)
        axes[0,1].legend(fontsize=8)
        for student,color in zip(VARIANTS[:2],("#197d85","#924462")):
            for teacher,style in (("teacher_h4","-"),("teacher_h10","--")):
                points=[lookup(student,teacher,w)["rmse_by_group"]["present"] for w in ("first_25","last_25")]
                axes[0,2].plot([0,1],points,style,marker="o",label=student+" / "+teacher,color=color)
        axes[0,2].set(title="Early / final recorded25 calls",xticks=[0,1],xticklabels=["First25","Last25"])
        axes[0,2].legend(fontsize=7)
        for student,color in zip(VARIANTS[:2],("#197d85","#924462")):
            row=lookup(student,"teacher_h4","all_prefix")
            axes[1,0].plot(np.arange(29),row["per_joint_rmse"],label=student,color=color)
        axes[1,0].set(title="Per canonical joint: residual to H4",xlabel="Canonical joint index")
        axes[1,0].legend(fontsize=8)
        stages=("cpu_native_state","cpu_causal_receiver","cpu_current_history_features",
                "input_transfer_and_history_normalization_host","student_forward_host","output_transfer_or_explicit_hold_host")
        short=("CPU state","Receiver","Features","Input/norm","Forward host","Output host")
        for shift,student,color in ((-.17,"coverage_f0","#197d85"),(.17,"original_f0","#924462")):
            timing=summaries[student]["timing"]["without_first_two"]["host_stages"]
            axes[1,1].bar(np.arange(6)+shift,[timing[k]["p50_ms"] for k in stages],width=.32,label=student,color=color)
        axes[1,1].set(title="Host partitions, without first2: median ms",xticks=np.arange(6),xticklabels=short)
        axes[1,1].legend(fontsize=8)
        for shift,student,color in ((-.17,"coverage_f0","#197d85"),(.17,"original_f0","#924462")):
            t=summaries[student]["timing"]
            axes[1,2].bar(np.arange(3)+shift,[t["without_first_two"]["worker_action"][k]
                for k in ("p50_ms","p95_ms","maximum_ms")],width=.32,label=student,color=color)
        axes[1,2].set(title="Worker software ms, without first2",xticks=np.arange(3),xticklabels=("Median","p95","Max"))
        axes[1,2].legend(fontsize=8)
        for ax in axes.flat:
            ax.grid(axis="y",alpha=.2)
            ax.tick_params(axis="x",labelsize=8)
        for ax in (axes[0,0],axes[0,1],axes[1,1]):ax.tick_params(axis="x",rotation=22)
        note="failure call included" if case["failure_call_included"] else "failure call beyond128 cap"
        fig.suptitle(f"{clip}: {case['prefix_calls']} recorded states; {note}\n"
            "Canonical action deltas, not tracking error. H4/H10 are privileged offline references. No new physics/training; no hardware latency claim.",fontsize=10)
        for suffix in ("png","pdf"):fig.savefig(output/(clip+"."+suffix),dpi=150)
        plt.close(fig)
        for row in rows:
            table.append(dict(clip=clip,student=row["student"],reference=row["privileged_reference"],window=row["window"],
                samples=row["samples"],**row["rmse_by_group"]))
    with (output/"same_state_residuals.csv").open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    inputs=[]
    for p in files:
        b=p.read_bytes();inputs.append(dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
    (output/"sources.json").write_text(json.dumps(dict(inputs=inputs,physics=0,training=0,
        hardware_end_to_end_latency_ms=None,natural_teleoperation_qualified=False),indent=2)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args();deliver(a.source,a.output)
