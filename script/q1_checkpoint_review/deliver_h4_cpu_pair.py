"""Plot the audited R1 H4 CPU comparison; never load a policy or run physics."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent/"supermimic_teleop"),str(HERE.parent/"q1_five_stage"),
               str(HERE.parent/"primebot_q1")]
from common import identity,require,save
from delivery import read
from event_profile import statistics
from h4_cpu_pair import CASES, compare_pair, timing


def deliver(source,output):
    require(source.is_absolute() and output.is_absolute() and not output.exists()
        and not output.is_relative_to(source),"Use saved originals and a fresh figure directory")
    manifest = read(source/"originals.json")
    for row in manifest["files"]:
        p = source/row["name"]
        raw = p.read_bytes()
        require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"],"Original identity changed")
    result = read(source/"result.json")
    require(result.get("status") == "CPU_H4_PAIRED_ACTIONS_PASSED" and result.get("source_integrity") == "PASSED"
        and result.get("new_policy_calls") == 512 and len(result.get("phases",[])) == 4
        and result.get("new_physical_steps") == result.get("new_training_updates") == 0
        and result.get("cost_settlement",{}).get("state") == "SETTLED",
        "Require completed512 actions, source gate and settled cleanup")
    groups,table,counts,inputs = {},[],[],[identity(source/"result.json"),identity(source/"originals.json")]
    for event,converter in CASES:
        name = event+"_"+converter
        path = source/(name+".frames.jsonl")
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        value = read(source/(name+".summary.json"))
        guard = read(source/(name+".guard.json"))
        require(value["calls"] == 128 and value["status"] == "SAVED_ACTION_PARITY_PASSED"
            and guard["status"] == "WORKER_COMPLETED" and guard["cleanup"]["worker_reaped"] is True
            and guard["cleanup"]["worker_return_code"] == 0 and timing(rows) == value["timing"],
            "Saved cases, timing or cleanup differ")
        groups[(event,converter)] = rows,value
        ready = sum(r["stream"]["input_ready"] is True for r in rows)
        require(all(type(r["stream"]["input_ready"]) is bool for r in rows),"Missing explicit readiness")
        counts.append(dict(event=event,cpu_converter=converter,requests=len(rows),
            ready_cuda_forwards=ready,explicit_fallbacks=len(rows)-ready))
        inputs += [identity(source/(name+suffix)) for suffix in (".frames.jsonl",".summary.json",".guard.json")]
        saved_metrics = dict(value["timing"])
        for flag,label in ((True,"ready_without_first_two"),(False,"fallback_without_first_two")):
            selected = [r for r in rows[2:] if r["stream"]["input_ready"] is flag]
            if selected:
                saved_metrics[label] = {field:statistics([r["stream"][field] for r in selected])
                    for field in ("gpu_policy_wall_compute_ns","worker_action_wall_ns",
                                  "isolated_request_worker_wall_ns","local_pipe_rpc_ns")}
        for scope,metrics in saved_metrics.items():
            if not isinstance(metrics,dict):
                continue
            for metric,stats in metrics.items():
                table.append(dict(event=event,cpu_converter=converter,scope=scope,metric=metric,**stats))
    for old in result["comparisons"]:
        event = old["event"]
        a,b = [groups[(event,c)] for c in ("original","kinematic")]
        require(a[1]["ready"]["canonical_present"] == b[1]["ready"]["canonical_present"],"Presence changed")
        found = dict(event=event,**compare_pair(a[0],b[0],a[1]["ready"]["canonical_present"]),
            original_timing=a[1]["timing"],kinematic_timing=b[1]["timing"])
        require(found == old,"Original comparison disagrees with saved frames")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True,exist_ok=False)
    fig,axes = plt.subplots(2,3,figsize=(16,9.5),constrained_layout=True)
    colors = {"original":"#a64e64","kinematic":"#16847a"}
    summaries = []
    for event,converter in CASES:
        rows,value = groups[(event,converter)]
        color = colors[converter]
        label = event+" "+converter
        wall = np.asarray([r["stream"]["isolated_request_worker_wall_ns"] for r in rows])/1e6
        rpc = np.asarray([r["stream"]["local_pipe_rpc_ns"] for r in rows])/1e6
        errors = np.asarray([r["saved_action_maximum_difference"] for r in rows])
        ax = axes[0,0 if event == "nominal" else 1]
        ax.plot(range(128),wall,label=converter,color=color,linewidth=1)
        axes[0,2].plot(range(128),errors,label=label,color=color,
            linestyle="-" if event == "nominal" else "--",linewidth=1)
        stable = np.sort(wall[2:])
        axes[1,0].plot(stable,np.arange(1,127)/126,label=label,color=color,
            linestyle="-" if event == "nominal" else "--")
        summaries.append((label,wall[2:],rpc[2:]))
    for ax,event in zip(axes[0,:2],("nominal","dropout")):
        ax.set(title=event+": complete request worker, all128 calls",xlabel="Frame",ylabel="ms",yscale="log")
        ax.axhline(20,color="#555555",linestyle="--",linewidth=.7)
    axes[0,2].set(title="Difference from actual saved R1 H4 actions",xlabel="Frame",ylabel="Maximum absolute action difference")
    axes[0,2].axhline(2e-5,color="#555555",linestyle="--",linewidth=.7)
    axes[0,2].ticklabel_format(axis="y",style="sci",scilimits=(-2,2))
    axes[1,0].set(title="Complete worker CDF, excluding first2",xlabel="ms",ylabel="Fraction")
    x = np.arange(len(summaries))
    for offset,field,q,label in ((-.27,1,50,"Worker p50"),(-.09,1,95,"Worker p95"),
                                (.09,2,50,"RPC p50"),(.27,2,95,"RPC p95")):
        axes[1,1].bar(x+offset,[np.percentile(r[field],q) for r in summaries],width=.17,label=label)
    axes[1,1].axhline(20,color="#555555",linestyle="--",linewidth=.7)
    axes[1,1].set(title="Software period, excluding first2",ylabel="ms",xticks=x,
        xticklabels=[r[0].replace(" ","\n") for r in summaries])
    for offset,field,label in ((-.18,1,"Worker"),(.18,2,"Local pipe RPC")):
        axes[1,2].bar(x+offset,[np.sum(r[field]>20) for r in summaries],width=.34,label=label)
    axes[1,2].set(title="Calls above20ms /126, excluding first2",ylabel="Calls",xticks=x,
        xticklabels=[r[0].replace(" ","\n") for r in summaries])
    for ax in axes.flat:
        ax.grid(axis="y",alpha=.2)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)
    forward_calls = sum(r["ready_cuda_forwards"] for r in counts)
    fallback_calls = sum(r["explicit_fallbacks"] for r in counts)
    fig.suptitle(f"R1 H4 CPU_PAIR01:512 requests / {forward_calls} ready forwards / {fallback_calls} fallbacks; serial ABBA\n"
        "80ms requested future preview remains. Software only; no new physics, training or hardware end-to-end claim.",fontsize=10)
    for suffix in ("png","pdf"):
        fig.savefig(output/("r1_h4_cpu_pair."+suffix),dpi=150)
    plt.close(fig)
    with (output/"r1_h4_cpu_pair.csv").open("x",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    require(all(identity(r["path"]) == r for r in inputs),"Originals changed while plotting")
    save(output/"audit.json",dict(status="PASSED",inputs=inputs,source_files_unchanged=True,
        figures_png=1,figures_pdf=1,new_policy_calls=0,new_physical_steps=0,new_training_updates=0,
        saved_request_counts=counts,saved_requests=sum(r["requests"] for r in counts),
        saved_ready_cuda_forwards=forward_calls,saved_explicit_fallbacks=fallback_calls,
        hardware_end_to_end_latency_ms=None,natural_teleoperation_qualified=False,user_accepted_checkpoint=None))
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    print(deliver(args.source,args.output))
