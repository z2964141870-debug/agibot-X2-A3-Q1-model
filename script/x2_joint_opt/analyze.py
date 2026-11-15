"""Read this experiment's saved tapes; no models, ROS, or hardware imports."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import PLANS, validate_prefix
from script.x2_joint_latency.timing import statistics


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(root):
    execution=json.loads((root/"execution.json").read_text())
    if execution["status"] != "HOST_OPT_COMPARISON_COMPLETED" or not execution["source_unchanged"]:
        raise ValueError("Incomplete experiment; preserve partial results")
    results={}; files=[root/"execution.json",root/"dispatch.json",root/"admitted.json"]
    per_frame=[]
    for variant,plan in PLANS.items():
        joint=root/(variant+"_joint")
        prefix=root/(variant+"_prefix")
        opt=json.loads((joint/"opt_config.json").read_text())
        validate_prefix(prefix,opt["config"])
        if opt["plan"]!=plan or not opt["source_unchanged"]:
            raise ValueError("Wrong candidate source/plan")
        run=json.loads((joint/"run_receipt.json").read_text())
        info=json.loads((joint/"fgp_summary.json").read_text())
        native=json.loads((joint/"native/status.json").read_text())
        if (run["status"]!="JOINT_RECORDING_COMPLETED" or any(c["returncode"] for c in run["children"])
                or info["outputs"]!=info["completed"] or info["outputs"]!=info["gpu_calls"]
                or info["dropped_after_denoising"] or info["max_in_flight"]>2
                or native["hardware_commands"] or native["sdk_publish_calls"]):
            raise ValueError("Joint completion/ordering/observe invariant failed")
        rows=[r for r in lines(joint/"fgp.jsonl") if r["output"]]
        policy=[r for r in lines(joint/"native/native.jsonl") if r.get("kind")=="policy_sample"]
        ready=[r for r in policy if r["reply"].get("ready")]
        if len(rows)!=info["outputs"] or not ready:
            raise ValueError("Incomplete recording")
        bindings=lines(joint/"kernel_binding_fgp.jsonl")+lines(joint/"kernel_binding_policy.jsonl")
        bindings.append(json.loads((joint/"native_kernel_binding.json").read_text()))
        if Counter(r["role"] for r in bindings)!=Counter(plan.keys()):
            raise ValueError("Missing/duplicate role binding")
        if any(r["cpus"]!=plan[r["role"]] for r in bindings):
            raise ValueError("Actual CPU affinity mismatch")
        profile=lines(joint/"profile.jsonl")
        counts=Counter(r["stage"] for r in profile)
        for name in ("cano_session","cano_adapter","lip_forward","gp_forward","gp_tracker_inclusive"):
            if counts[name]!=len(rows):raise ValueError("Missing profile events: "+name)
        if counts["native_solver"]+counts["worker_solver_rpc"]!=len(rows):
            raise ValueError("Unaccounted physics calls")
        sections={}
        for name in sorted(counts):
            selected=[r for r in profile if r["stage"]==name]
            sections[name]={key:statistics([r[key] for r in selected]) for key in ("wall_ms","thread_cpu_ms")}
        for key in ("h2d_stream_span_ms","graph_stream_span_ms","d2h_stream_span_ms"):
            sections["cano_session"][key]=statistics([r[key] for r in profile if r["stage"]=="cano_session"])
        first,last=rows[0]["acknowledged_ns"],rows[-1]["acknowledged_ns"]
        monitor=root/(variant+"_monitor")
        mon=json.loads((monitor/"summary.json").read_text())
        if mon["status"]!="TARGET_EXITED" or mon["tegrastats_returncode"]!=-15:
            raise ValueError("Resource monitor did not finish cleanly")
        samples=[r for r in lines(monitor/"samples.jsonl") if first<=r["monotonic_ns"]<=last]
        gpu=[t["gpu_busy_percent"] for r in samples for t in r["tegrastats"]
             if first<=t["received_monotonic_ns"]<=last and t.get("gpu_busy_percent") is not None]
        ages=[r["reply"]["joint_test_timing"]["oldest_reference_age_ms"] for r in ready]
        if any(r["reply"].get("publish_allowed") is not False or r["reply"]["joint_test_timing"]["preview_span_ms"]!=180 for r in ready):
            raise ValueError("Preview or publish invariant changed")
        result=dict(outputs=len(rows),skipped=info["dropped_input_frames"],human_hz=(len(rows)-1)*1e9/(last-first),
            output_interval_ms=statistics(np.diff([r["acknowledged_ns"] for r in rows])/1e6,1000/30),
            source_to_ack_ms=statistics([r["software_input_age_at_ack_ms"] for r in rows]),
            rpc_ready_ms=statistics([r["rpc_ms"] for r in ready],20),
            deadline_exceed={str(b):sum(r["rpc_ms"]>b for r in ready) for b in (20,25,30,1000/30)},
            accepted=sum(bool(r["accepted"]) for r in policy),ready=len(ready),
            rejection_reasons=dict(Counter(r["rejection_reason"] for r in policy if not r["accepted"])),
            reference_age_lower_ms=statistics(ages),
            reference_age_upper_ms=statistics([a+r["rpc_ms"] for a,r in zip(ages,ready)]),
            fgp_stages_ms={k:statistics([r[k] for r in rows]) for k in
                ("gpu_stage_ms","cpu_stage_ms","cpu_queue_ms","sender_queue_ms","bridge_ms","human_rpc_ms","fgp_ms")},
            policy_stages_ms={k:statistics([r["reply"][k] for r in ready]) for k in
                ("preprocessing_ms","policy_and_features_ms","full_runtime_ms")},
            profile_counts=dict(counts),profile_stages=sections,kernel_bindings=bindings,
            steady_resources=dict(cpu_samples=len(samples),gpu_samples=len(gpu),
                cpu_mean_percent={k:float(np.mean([r["cpu_busy_percent"][k] for r in samples])) for k in samples[0]["cpu_busy_percent"]},
                gpu_busy=statistics(gpu),max_temperature_c=mon["max_temperature_c"],
                min_available_bytes=mon["min_available_bytes"],whole_device_not_process_attributed=True),
            guard_wall_s=run["wall_s"],hardware_commands=0)
        results[variant]=result
        for r in rows:per_frame.append(dict(variant=variant,frame=r["frame"],
            source_to_ack_ms=r["software_input_age_at_ack_ms"],gpu_host_ms=r["gpu_stage_ms"],
            cpu_ms=r["cpu_stage_ms"],human_rpc_ms=r["human_rpc_ms"]))
        for base,names in ((joint,("run_receipt.json","fgp_summary.json","native/status.json","opt_config.json",
                "fgp.jsonl","fgp_outputs.npz","profile.jsonl","native/native.jsonl","kernel_binding_fgp.jsonl","kernel_binding_policy.jsonl","native_kernel_binding.json")),
                (prefix,("guard.json","parity.json","fgp_outputs.npz","opt_config.json","kernel_binding_fgp.jsonl")),
                (monitor,("summary.json","samples.jsonl"))):
            files.extend(base/n for n in names)
    full_parity={"same_selected_frames":False,"max_absolute_difference":None,"tolerance":2e-5}
    with np.load(root/"r2_profile_joint/fgp_outputs.npz",allow_pickle=False) as left, np.load(root/"dedicated_hosts_joint/fgp_outputs.npz",allow_pickle=False) as right:
        full_parity["same_selected_frames"]=np.array_equal(left["frame"],right["frame"])
        if full_parity["same_selected_frames"]:
            if not all(np.isfinite(x[k]).all() for x in (left,right) for k in ("pose","translation")):
                raise ValueError("Nonfinite saved pose/root")
            full_parity["max_absolute_difference"]={k:float(np.max(np.abs(left[k]-right[k]))) for k in ("pose","translation")}
    full_parity["passed"]=bool(full_parity["same_selected_frames"] and max(full_parity["max_absolute_difference"].values())<=full_parity["tolerance"])
    hashes=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    return dict(status="HOST_OPT_SAVED_ANALYSIS_PASSED",variants=results,source_files=hashes,
        limitation="Short sequential, equally instrumented, recorded-IMU/live-read-only-feedback comparison. Not pure kernel or mechanical latency, live human quality, causal attribution, or adoption.",
        full_sequence_parity=full_parity,preview_ms=180,hardware_commands=0),per_frame


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():parser.error("Preserve existing analysis")
    result,rows=summarize(a.root)
    a.output.mkdir(parents=True)
    (a.output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    with (a.output/"frames.csv").open("x",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    if any(sha(Path(r["path"]))!=r["sha256"] for r in result["source_files"]):
        raise RuntimeError("Recording changed during saved analysis")
    print(json.dumps({k:{"hz":v["human_hz"],"rpc_p95_ms":v["rpc_ready_ms"]["p95"],
        "ack_p95_ms":v["source_to_ack_ms"]["p95"],"interval_p95_ms":v["output_interval_ms"]["p95"],
        "over20":v["rpc_ready_ms"]["over_budget"]} for k,v in result["variants"].items()}))


if __name__=="__main__":main()
