"""Draw seven six-panel comparisons only from the real saved f0 analysis."""

import argparse
import csv
from pathlib import Path
import shutil

import numpy as np

from student_causal_summary import FIXED, identity, match_controls, read, require, save


def deliver(source, output):
    inputs = [identity(source/name) for name in ("summary.json", "audit.json", "per_case.csv")]
    value, audit = read(source/"summary.json"), read(source/"audit.json")
    require(value.get("stage") == "R2_F0_MATCHED_SAVED_REVIEW_ANALYSIS"
        and value.get("total_comparison_cases") == len(value.get("rows", [])) == 21
        and value.get("new_saved_cases") == 7 and value.get("reused_saved_controls") == 14
        and audit.get("status") == "PASSED" and audit.get("source_inputs_unchanged") is True
        and value.get("new_inference_calls") == value.get("new_physical_transitions")
            == value.get("new_training_updates") == 0, "Require actual passed f0 saved analysis")
    rows = value["rows"]
    match_controls([r for r in rows if r["variant"] == "f0_student"],
                   [r for r in rows if r["variant"] != "f0_student"])
    require(output.is_absolute() and not output.exists(), "Use a new f0 delivery directory")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    output.mkdir(parents=True)
    variants = ("teacher", "student", "f0_student")
    labels = ("Teacher H4", "R1 H4", "R2 f0 + history")
    colors = ("#328b96", "#ab6546", "#637c45")
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
        ("window_upper", "Capture-clock first1000ms upper RMSE", 1),
        ("action_jump", "Maximum saved canonical action jump", 1))

    def measured(row, key):
        if key == "window_upper":
            return None if row["capture_window"] is None else row["capture_window"]["upper_rmse_rad"]
        if key == "action_jump":
            return row["software"]["action_delta_max_joint_max"]
        return row.get(key)

    def draw(selected, panels, title, stem, comparable):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for ax, (key, name, scale) in zip(axes.flat, panels):
            right = {"slip_left_m_s":"slip_right_m_s", "sole_left_m":"sole_right_m"}.get(key)
            limit = max([measured(r, key)*scale for r in comparable if measured(r, key) is not None]+[1e-9])
            if right:
                limit = max(limit, max([r[right]*scale for r in comparable if r.get(right) is not None]+[0]))
            limit = 110 if key == "completion_fraction" else limit*1.2
            for index, row in enumerate(selected):
                datum = measured(row, key)
                if datum is None:
                    ax.text(index, limit*.03, "FAILED / WINDOW INCOMPLETE", rotation=90,
                        ha="center", va="bottom", fontsize=7)
                    continue
                ax.bar(index, datum*scale, color=colors[index], width=.6)
                if right and row.get(right) is not None:
                    ax.plot(index, row[right]*scale, "_", color="#151515", markersize=13, markeredgewidth=2)
            ax.set(title=name, xticks=np.arange(3), xticklabels=labels, ylim=(0, limit))
            ax.tick_params(axis="x", labelsize=8)
            ax.grid(axis="y", alpha=.2)
        fig.suptitle(title, fontsize=11)
        for suffix in ("png", "pdf"):
            fig.savefig(output/(stem+"."+suffix), dpi=150)
        plt.close(fig)
        return output/(stem+".png")

    figures = []
    for clip in FIXED:
        selected = [next(r for r in rows if (r["event"], r["clip"], r["variant"])
            == ("nominal", clip, v)) for v in variants]
        figures.append(draw(selected, gait, clip+" | actual f0 vs saved H4 controls; not hardware latency",
            "nominal_"+clip, [r for r in rows if r["event"] == "nominal"]))
    for event in ("stop", "reverse", "dropout", "late_burst"):
        selected = [next(r for r in rows if (r["event"], r["variant"]) == (event, v)) for v in variants]
        figures.append(draw(selected, events, event+" | synthetic capture-clock comparison / A022",
            "event_"+event, [r for r in rows if r["event"] != "nominal"]))
    overview = Image.new("RGB", (1680, 2040), "white")
    pen = ImageDraw.Draw(overview)
    for index, path in enumerate(figures):
        with Image.open(path) as source_image:
            thumb = source_image.convert("RGB")
            thumb.thumbnail((840, 480))
        x, y = (index%2)*840, (index//2)*510
        overview.paste(thumb, (x, y+20))
        pen.text((x+8, y+2), path.stem, fill="black")
    overview.save(output/"overview.png")
    software_rows = []
    for row in rows:
        for field in ("gpu_policy_wall_compute_ms", "worker_action_wall_ms", "local_pipe_rpc_ms"):
            for scope in ("all", "without_first", "without_first_two"):
                software_rows.append(dict(event=row["event"], clip=row["clip"], variant=row["variant"],
                    metric=field, scope=scope, **row["software"][field][scope]))
    with (output/"software_timing.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in software_rows for k in r)))
        writer.writeheader()
        writer.writerows(software_rows)
    for name in ("summary.json", "audit.json", "per_case.csv"):
        shutil.copyfile(source/name, output/name)
    require(all(identity(r["path"]) == r for r in inputs), "Saved f0 analysis changed while plotting")
    save(output/"delivery_audit.json", dict(status="PASSED", inputs=inputs, figures=7,
        new_inference=0, new_physics=0, new_training=0,
        outputs=[identity(p) for p in sorted(output.iterdir()) if p.is_file()]))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(deliver(args.source, args.output))
