"""Analyze completed paused-preview artifacts without new inference or physics."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_teleop"))
from analyze import metrics
from tracking_response import analyze_signal


def identity(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def analyze(source, output):
    started = time.monotonic()
    result = json.loads((source / "result.json").read_text())
    authorization = json.loads((source / "authorization.json").read_text())
    assert result["status"] == "COMPLETED" and result["source_integrity"] == "PASSED"
    assert authorization["iteration"] == 32000 and authorization["policy_device"] == "cuda"
    paths = [source / name for name in ("result.json", "authorization.json", "comparison.json", "equivalence.json")]
    paths.extend((Path(__file__).resolve(), HERE / "analyze.py",
                  HERE.parent / "supermimic_teleop/tracking_response.py"))
    rows, phases = [], []
    for method, horizon, clip in authorization["cases"]:
        folder = source if method == "masked" else source / ("tail_"+method)
        folder = folder / f"h{horizon}" / clip
        saved = folder / "review/result.json"
        value = json.loads(saved.read_text())
        assert value["status"] == "PASSED" and value["iteration"] == 32000 and len(value["cases"]) == 1
        case = value["cases"][0]
        assert case["clip"] == clip and case["video"]["original_speed"] is True
        trace, telemetry = Path(case["trace"]), folder / "stream_timing.jsonl"
        upper = trace.with_name("upper_timing.csv")
        paths.extend((saved, trace, telemetry, upper))
        before = [identity(path) for path in (trace, telemetry, upper)]
        row = dict(method=method, horizon_frames=horizon, preview_budget_ms=horizon*20,
            clip=clip, **metrics(case), failure_frame=case["failure_frame"],
            completion_fraction=case["valid_steps"]/case["reference_steps"], video=case["video"])
        with np.load(trace, allow_pickle=False) as arrays:
            actions = arrays["canonical_action"]
            if len(actions) > 1:
                jumps = np.sqrt(np.mean(np.diff(actions, axis=0)**2, axis=1))
                row.update(action_delta_rms_p95=float(np.quantile(jumps, .95)),
                           action_delta_rms_max=float(jumps.max()))
        timings = [json.loads(line) for line in telemetry.read_text().splitlines()]
        assert len(timings) == case["attempted_steps"] and all(t["policy_device"] == "cuda" for t in timings)
        compute = [t["gpu_policy_wall_compute_ns"]/1e6 for t in timings]
        row["gpu_goal_call_wall_ms"] = (dict(p50=float(np.quantile(compute,.5)),
            p95=float(np.quantile(compute,.95)), maximum=max(compute)) if compute else None)
        with upper.open(newline="") as stream:
            reader = csv.DictReader(stream)
            names = [name.removeprefix("actual_") for name in reader.fieldnames if name.startswith("actual_")]
            values = list(reader)
        if case["complete"]:
            for name in names:
                signal = analyze_signal([float(v["reference_"+name]) for v in values],
                                        [float(v["actual_"+name]) for v in values])
                phases.append(dict(method=method, horizon_frames=horizon, clip=clip, joint=name, **signal))
        assert all(identity(Path(record["path"])) == record for record in before)
        rows.append(row)
    assert len(rows) == 27 and len({(r["method"],r["horizon_frames"],r["clip"]) for r in rows}) == 27
    paths = list(dict.fromkeys(paths))
    inputs = [identity(path) for path in paths]
    output.mkdir(parents=True, exist_ok=False)
    value = dict(stage="PAUSED32K_PREVIEW_SAVED_ARTIFACT_ANALYSIS", rows=rows, upper_phase=phases,
        inputs=inputs, evaluation_commit=authorization["evaluation_commit"],
        complete_cases=sum(row["complete"] for row in rows), total_cases=len(rows),
        method_order=["masked", "hold", "damped_cv"], physical_transitions=0, optimizer_updates=0,
        new_inference_calls=0, measured_end_to_end_latency_ms=None, user_accepted_checkpoint=None,
        interpretation="Nominal offline replay of three previously seen clips; phase correlation is descriptive, not measured robot delay",
        action_delta_note="RMS of 29 canonical action changes per20ms, including seven absent zero joints; not joint acceleration",
        compute_note="Synchronized goal packing, transfers and GPU forward; excludes observation preprocessing, RPC and robot execution",
        failed_case_policy="No full-clip error or phase estimates for failed episodes; completion fraction uses valid steps")
    (output / "summary.json").write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    flattened = [{k:v for k,v in row.items() if k not in ("video", "gpu_goal_call_wall_ms")} for row in rows]
    fields = list(dict.fromkeys(key for row in flattened for key in row))
    with (output / "per_case.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)
    plot(rows, output)
    assert all(identity(Path(record["path"])) == record for record in inputs)
    (output / "audit.json").write_text(json.dumps(dict(status="PASSED", source_files=len(inputs),
        source_inputs_unchanged=True, seconds=time.monotonic()-started, new_physics=0,
        new_inference=0, new_training=0), indent=2)+"\n")
    return value


def plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = (("completion_fraction","Valid completion (%)",100),
        ("upper_rmse_rad","Upper joint RMSE (rad)",1),
        ("slip_left_m_s","Loaded slip (mm/s); right dashed",1000),
        ("sole_left_m","Sole p95 (mm); right dashed",1000),
        ("alternations","Alternating lifts above10mm",1),
        ("tilt_rms_rad","Torso tilt RMS (deg)",180/np.pi))
    palette = dict(masked="#247c89", hold="#ba643a", damped_cv="#637a32")
    for clip in dict.fromkeys(row["clip"] for row in rows):
        figure, axes = plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
        for axis,(key,title,scale) in zip(axes.flat, panels):
            for method,color in palette.items():
                series = sorted((r for r in rows if r["clip"]==clip and r["method"]==method),
                                key=lambda r:r["preview_budget_ms"])
                axis.plot([r["preview_budget_ms"] for r in series],
                    [r.get(key,np.nan)*scale if r.get(key) is not None else np.nan for r in series],
                    "o-",color=color,label=method)
                right = {"slip_left_m_s":"slip_right_m_s","sole_left_m":"sole_right_m"}.get(key)
                if right:
                    axis.plot([r["preview_budget_ms"] for r in series],
                        [r.get(right,np.nan)*scale if r.get(right) is not None else np.nan for r in series],
                        "o--",color=color,linewidth=1)
            axis.set(title=title,xlabel="Required real preview (ms)",xticks=(20,40,80,100,200))
            axis.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8)
        figure.suptitle(clip+" | Q1 32k human/CUDA; failed episodes have no full-clip errors",fontsize=11)
        figure.savefig(output/(clip+".png"),dpi=150)
        figure.savefig(output/(clip+".pdf"))
        plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    result = analyze(args.source,args.output)
    print(json.dumps(dict(complete=result["complete_cases"],total=result["total_cases"],output=str(args.output))))
