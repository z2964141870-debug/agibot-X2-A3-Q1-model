"""Draw the real coverage review against saved H4 and original f0 controls."""

import argparse
import copy
import csv
from pathlib import Path
import shutil

import numpy as np

from student_coverage_summary import identity, join_comparisons, read, require, save
from student_summary import FIXED, full_group

VARIANTS = ("teacher", "student", "original_f0", "coverage_f0")
LABELS = ("Teacher H4", "R1 H4", "Original f0", "Coverage f0")
COLORS = ("#328b96", "#ab6546", "#77818b", "#637c45")


def saved_gate(value, audit):
    rows = value.get("rows", [])
    require(value.get("stage") == "COVERAGE01_MATCHED_SAVED_ANALYSIS"
        and value.get("total_comparison_cases") == len(rows) == 28
        and value.get("new_saved_cases") == 7 and value.get("reused_saved_controls") == 21
        and value.get("selected_update") == 500
        and value.get("new_inference_calls") == value.get("new_physical_transitions")
            == value.get("new_training_updates") == 0
        and value.get("measured_hardware_end_to_end_latency_ms") is None
        and value.get("natural_teleoperation_qualified") is False
        and audit.get("status") == "PASSED" and audit.get("source_inputs_unchanged") is True,
        "Require the actual completed saved28 coverage comparison")
    controls = [copy.deepcopy(r) for r in rows if r["variant"] in ("teacher", "student")]
    original = [dict(copy.deepcopy(r), variant="f0_student") for r in rows if r["variant"] == "original_f0"]
    current = [dict(copy.deepcopy(r), variant="f0_student") for r in rows if r["variant"] == "coverage_f0"]
    require(len(controls) == 14 and len(original) == len(current) == 7
        and value.get("complete_new_cases") == sum(r["complete"] for r in current),
        "Do not omit or substitute a failed task")
    join_comparisons(current, controls, original)
    groups = [dict(variant=v, **full_group([r for r in rows if r["event"] == "nominal"
        and r["variant"] == v])) for v in VARIANTS]
    require(value.get("nominal_groups") == groups, "Incomplete nominal groups cannot acquire full-set means")
    require(all(r.get("upper_rmse_rad") is None for r in rows if not r["complete"]),
        "Failed cases cannot acquire full-clip upper error")
    return rows


def deliver(source, output):
    require(source.is_absolute() and output.is_absolute() and not output.exists()
        and not output.is_relative_to(source), "Use a new independent coverage figure directory")
    inputs = [identity(source/n) for n in ("summary.json", "audit.json", "per_case.csv")]
    value, audit = read(source/"summary.json"), read(source/"audit.json")
    rows = saved_gate(value, audit)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    output.mkdir(parents=True, exist_ok=False)
    gait = (("completion_fraction", "Valid completion (%)", 100),
        ("upper_rmse_rad", "Full-clip upper RMSE (rad)", 1),
        ("slip_left_m_s", "Loaded slip L/R (mm/s)", 1000),
        ("sole_left_m", "Sole height p95 L/R (mm)", 1000),
        ("alternations", "Alternating lifts above10mm", 1),
        ("tilt_rms_rad", "Torso tilt RMS (deg)", 180/np.pi))
    events = (("upper_rmse_rad", "Full-clip upper RMSE (rad)", 1),
        ("slip_left_m_s", "Loaded slip L/R (mm/s)", 1000),
        ("sole_left_m", "Sole height p95 L/R (mm)", 1000),
        ("tilt_rms_rad", "Torso tilt RMS (deg)", 180/np.pi),
        ("window_upper", "Capture-clock first1000ms upper RMSE (rad)", 1),
        ("action_jump", "Maximum saved canonical action jump", 1))

    def measured(row, key):
        if key == "window_upper":
            return None if row["capture_window"] is None else row["capture_window"]["upper_rmse_rad"]
        if key == "action_jump":
            return row["software"]["action_delta_max_joint_max"]
        return row.get(key)

    def draw(selected, panels, title, stem, comparable):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
        for ax, (key, name, scale) in zip(axes.flat, panels):
            right = {"slip_left_m_s":"slip_right_m_s", "sole_left_m":"sole_right_m"}.get(key)
            values = [measured(r, key)*scale for r in comparable if measured(r, key) is not None]
            if right:
                values.extend(r[right]*scale for r in comparable if r.get(right) is not None)
            lower = min(values+[0.])*1.2
            limit = 110 if key == "completion_fraction" else max(values+[1e-9])*1.2
            for index, row in enumerate(selected):
                datum = measured(row, key)
                if datum is None:
                    ax.text(index, limit*.03, "FAILED / WINDOW INCOMPLETE", rotation=90,
                        ha="center", va="bottom", fontsize=7)
                    continue
                ax.bar(index, datum*scale, color=COLORS[index], width=.6)
                if right and row.get(right) is not None:
                    ax.plot(index, row[right]*scale, "_", color="#151515", markersize=13, markeredgewidth=2)
            ax.set(title=name, xticks=np.arange(4), xticklabels=LABELS, ylim=(lower, limit))
            if lower < 0:
                ax.axhline(0, color="#555555", linewidth=.7)
            ax.tick_params(axis="x", labelsize=8)
            ax.grid(axis="y", alpha=.2)
        completion = " | ".join(f"{label}: {row['completion_fraction']:.1%}" for label,row in zip(LABELS,selected))
        fig.suptitle(title+"\n"+completion, fontsize=10)
        for suffix in ("png", "pdf"):
            fig.savefig(output/(stem+"."+suffix), dpi=150)
        plt.close(fig)
        return output/(stem+".png")

    figures = []
    for clip in FIXED:
        selected = [next(r for r in rows if (r["event"],r["clip"],r["variant"])
            == ("nominal",clip,v)) for v in VARIANTS]
        figures.append(draw(selected,gait,clip+" | selected500 / saved controls; not hardware latency",
            "nominal_"+clip,[r for r in rows if r["event"] == "nominal"]))
    for event in ("stop", "reverse", "dropout", "late_burst"):
        selected = [next(r for r in rows if (r["event"],r["variant"]) == (event,v)) for v in VARIANTS]
        figures.append(draw(selected,events,event+" | synthetic capture-clock comparison / A022",
            "event_"+event,[r for r in rows if r["event"] != "nominal"]))
    overview = Image.new("RGB",(1800,2200),"white")
    pen = ImageDraw.Draw(overview)
    for index,path in enumerate(figures):
        with Image.open(path) as source_image:
            thumb = source_image.convert("RGB")
            thumb.thumbnail((900,510))
        x,y = (index%2)*900,(index//2)*550
        overview.paste(thumb,(x,y+24))
        pen.text((x+8,y+3),path.stem,fill="black")
    overview.save(output/"overview.png")
    timings = []
    for row in rows:
        for field in ("gpu_policy_wall_compute_ms","worker_action_wall_ms","local_pipe_rpc_ms"):
            for scope in ("all","without_first","without_first_two"):
                timings.append(dict(event=row["event"],clip=row["clip"],variant=row["variant"],
                    metric=field,scope=scope,**row["software"][field][scope]))
    with (output/"software_timing.csv").open("x",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in timings for k in r)))
        writer.writeheader();writer.writerows(timings)
    for name in ("summary.json","audit.json","per_case.csv"):
        shutil.copyfile(source/name,output/name)
    require(all(identity(r["path"]) == r for r in inputs), "Coverage source changed while drawing")
    save(output/"delivery_audit.json",dict(status="PASSED",inputs=inputs,figures=7,
        new_inference=0,new_physics=0,new_training=0,
        outputs=[identity(p) for p in sorted(output.iterdir()) if p.is_file()]))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    print(deliver(args.source,args.output))
