"""Create R1 comparison six-panel figures from the saved-artifact analysis only."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from student_summary import FIXED, identity, require, save


def deliver(source, output):
    summary_path, audit_path = source/"summary.json", source/"audit.json"
    inputs = [identity(p) for p in (summary_path, audit_path)]
    value, audit = (json.loads(p.read_text()) for p in (summary_path, audit_path))
    require(value.get("stage") == "R1_MATCHED_SAVED_REVIEW_ANALYSIS" and value.get("total_cases") == 23
        and audit.get("status") == "PASSED" and audit.get("source_inputs_unchanged") is True,
        "Use the real passed23-case saved analysis")
    require(not output.exists() and output.is_absolute(), "Use a new delivery root")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    output.mkdir(parents=True)
    rows = value["rows"]
    order = (("teacher", 10), ("teacher", 4), ("student", 4), ("teacher", 1), ("student", 1))
    labels = ("T H10", "T H4", "S H4", "T H1", "S H1")
    colors = ("#667078", "#328b96", "#ab6546", "#637c45", "#8e6397")
    figures = []
    gait = (("completion_fraction", "Valid completion (%)", 100),
        ("upper_rmse_rad", "Full-clip upper RMSE (rad)", 1),
        ("slip_left_m_s", "Loaded slip L/R (mm/s)", 1000),
        ("sole_left_m", "Sole height p95 L/R (mm)", 1000),
        ("alternations", "Alternating lifts above10mm", 1),
        ("tilt_rms_rad", "Torso tilt RMS (deg)", 180/np.pi))
    limits = {key:max([r[key]*scale for r in rows if r.get(key) is not None]+[1e-9])*1.2
              for key, _, scale in gait}
    floors = {key:min([r[key]*scale for r in rows if r.get(key) is not None]+[0.])*1.2
              for key, _, scale in gait}
    for left, right, scale in (("slip_left_m_s", "slip_right_m_s", 1000), ("sole_left_m", "sole_right_m", 1000)):
        limits[left] = max(limits[left], max([r[right]*scale for r in rows if r.get(right) is not None]+[0])*1.2)
        floors[left] = min(floors[left], min([r[right]*scale for r in rows if r.get(right) is not None]+[0])*1.2)
    limits["completion_fraction"] = 110
    for clip in FIXED:
        selected = [next(r for r in rows if (r["event"], r["variant"], r["horizon_frames"], r["clip"])
                         == ("nominal", v, h, clip)) for v, h in order]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for ax, (key, title, scale) in zip(axes.flat, gait):
            for index, row in enumerate(selected):
                val = row.get(key)
                if val is None:
                    ax.text(index, limits[key]*.03, "FAILED", rotation=90, ha="center", va="bottom", fontsize=8)
                    continue
                ax.bar(index, val*scale, color=colors[index], width=.65)
                right = {"slip_left_m_s":"slip_right_m_s", "sole_left_m":"sole_right_m"}.get(key)
                if right and row.get(right) is not None:
                    ax.plot(index, row[right]*scale, "_", color="#151515", markersize=13, markeredgewidth=2)
            ax.set(title=title, xticks=np.arange(5), xticklabels=labels, ylim=(floors[key], limits[key]))
            if floors[key] < 0:
                ax.axhline(0, color="#555555", linewidth=.7)
            ax.grid(axis="y", alpha=.2)
        fig.suptitle(clip+" | 32k teacher(T) vs R1 student(S); same dependency-aware receiver", fontsize=11)
        for suffix in ("png", "pdf"):
            fig.savefig(output/("nominal_"+clip+"."+suffix), dpi=150)
        plt.close(fig)
        figures.append(output/("nominal_"+clip+".png"))
    events = ("stop", "reverse", "dropout", "late_burst")
    for event in events:
        selected = [next(r for r in rows if (r["event"], r["variant"], r["horizon_frames"])
                         == (event, variant, 4)) for variant in ("teacher", "student")]
        panels = (("upper_rmse_rad", "Full-clip upper RMSE (rad)", 1),
            ("slip_left_m_s", "Loaded slip L/R (mm/s)", 1000),
            ("sole_left_m", "Sole p95 L/R (mm)", 1000),
            ("tilt_rms_rad", "Torso tilt RMS (deg)", 180/np.pi),
            ("window_upper", "Capture-clock first1000ms upper RMSE", 1),
            ("action_jump", "Maximum saved canonical action jump", 1))
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for ax, (key, title, scale) in zip(axes.flat, panels):
            values = []
            for index, row in enumerate(selected):
                val = (row["capture_window"]["upper_rmse_rad"] if key == "window_upper" else
                    row["software"]["action_delta_max_joint_max"] if key == "action_jump" else row.get(key))
                if val is not None:
                    values.append(val*scale)
                    ax.bar(index, val*scale, color=colors[index+1], width=.55)
                    right = {"slip_left_m_s":"slip_right_m_s", "sole_left_m":"sole_right_m"}.get(key)
                    if right and row.get(right) is not None:
                        values.append(row[right]*scale)
                        ax.plot(index, row[right]*scale, "_", color="#151515", markersize=13, markeredgewidth=2)
                else:
                    ax.text(index, .03, "FAILED / WINDOW INCOMPLETE", rotation=90,
                        ha="center", va="bottom", transform=ax.get_xaxis_transform(), fontsize=8)
            ax.set(title=title, xticks=(0, 1), xticklabels=("Teacher H4", "Student H4"))
            lower, upper = min(values+[0.]), max(values+[0.])
            margin = max(upper-lower, 1e-6)*.1
            ax.set_ylim(lower-margin if lower < 0 else 0, upper+margin)
            if lower < 0:
                ax.axhline(0, color="#555555", linewidth=.7)
            ax.grid(axis="y", alpha=.2)
        fig.suptitle(event+" | synthetic event / A022 / 80ms requested preview; not hardware response latency", fontsize=11)
        for suffix in ("png", "pdf"):
            fig.savefig(output/("event_"+event+"."+suffix), dpi=150)
        plt.close(fig)
        figures.append(output/("event_"+event+".png"))
    thumbs = []
    for path in figures:
        image = Image.open(path).convert("RGB")
        image.thumbnail((840, 480))
        thumbs.append(image)
    overview = Image.new("RGB", (1680, 4*510), "white")
    draw = ImageDraw.Draw(overview)
    for i, (thumb, path) in enumerate(zip(thumbs, figures)):
        x, y = (i%2)*840, (i//2)*510
        overview.paste(thumb, (x, y+20))
        draw.text((x+8, y+2), path.stem, fill="black")
    overview.save(output/"overview.png")
    for name in ("summary.json", "audit.json", "per_case.csv"):
        shutil.copyfile(source/name, output/name)
    require(all(identity(p["path"]) == p for p in inputs), "Saved analysis changed while plotting")
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
