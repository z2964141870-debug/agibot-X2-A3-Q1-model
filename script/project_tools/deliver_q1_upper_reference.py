"""Plot and render the saved UR01 physical comparison; no new simulation steps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_ppo"))
from resources import identity, save

NAMES = ("RF11", "reference", "residual015")
LABELS = ("RF11", "Reference arms", "Residual +/-0.15 rad")
COLORS = ("#5d6268", "#008675", "#ce4b37")
FIXED = ("walk_ff_stop_180_R_002__A095", "walk_ff_start_180_R_002__A071",
         "triumph_two_handed_001__A058")


def render(runtime, trial, rows, traces):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from scipy.spatial.transform import Rotation

    model = mujoco.MjModel.from_xml_path(str(runtime / "q1_assets/scene.xml"))
    model.vis.quality.offsamples = 0
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 2., 135., -15.
    renderer = mujoco.Renderer(model, height=400, width=320)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    output = trial / "comparison.mp4"
    frames_written, segments = 0, []
    try:
        with imageio.get_writer(output, fps=25, codec="libx264", quality=7,
                                macro_block_size=16) as writer:
            for name in FIXED:
                index = next(i for i, row in enumerate(rows) if row["name"] == name)
                ref = traces["RF11"]
                length = int(ref["reference_steps"][index]) + 1
                initial_ref = np.loadtxt(rows[index]["path"], delimiter=",", ndmin=2)[0]
                initial_ref = np.r_[initial_ref[:3], initial_ref[[6, 3, 4, 5]],
                                    initial_ref[7:]]
                reference = np.concatenate((initial_ref[None], np.concatenate((
                    ref["reference_p"][:length - 1, index], ref["reference_quat"][:length - 1, index],
                    ref["reference_q"][:length - 1, index]), axis=1)))
                # Use every second recorded 50 Hz state at 25 fps, without interpolation.
                first = frames_written
                for frame in range(0, length, 2):
                    panels = []
                    for column, label in enumerate((*LABELS, "Kinematic reference")):
                        terminated = False
                        if column < 3:
                            trace = traces[NAMES[column]]
                            count = int(trace["steps"][index])
                            state = trace["initial"][index] if frame == 0 else trace["states"][min(frame, count) - 1, index]
                            qpos = np.r_[state[:7], state[13:35]]
                            terminated = frame > count
                        else:
                            qpos = reference[frame]
                        data.qpos[:] = qpos
                        mujoco.mj_kinematics(model, data)
                        mujoco.mj_comPos(model, data)
                        mujoco.mj_camlight(model, data)
                        camera.lookat[:] = [data.qpos[0], data.qpos[1], .45]
                        renderer.update_scene(data, camera=camera)
                        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                        image = Image.fromarray(renderer.render().copy())
                        draw = ImageDraw.Draw(image)
                        draw.rectangle((0, 0, 320, 46), fill=(22, 25, 28))
                        draw.text((6, 3), label, font=font, fill="white")
                        draw.text((6, 23), "Saved repeat 0 | " + str(frame * .02)[:4] + " s", font=font, fill="white")
                        if terminated:
                            draw.rectangle((0, 368, 320, 400), fill=(135, 25, 25))
                            draw.text((6, 375), "TERMINATED - pose held", font=font, fill="white")
                        panels.append(np.asarray(image))
                    writer.append_data(np.concatenate(panels, axis=1))
                    frames_written += 1
                segments.append(dict(name=name, first_frame=first, last_frame=frames_written - 1,
                    conditions={n: dict(steps=int(traces[n]["steps"][index]),
                                        failed=bool(traces[n]["failed"][index])) for n in NAMES}))
    finally:
        renderer.close()
    return dict(video=identity(output), fps=25, frames=frames_written,
                seconds=frames_written / 25, segments=segments,
                protocol="Three fixed independent clips, real-time playback; recorded physics, not a new rollout. Four synchronized columns; each camera follows its own root. Failed trajectories held and labelled. Initial reference uses source frame0, then recorded 50Hz reference.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    if (args.trial / "delivery.json").exists():
        raise FileExistsError("Saved delivery already exists")
    value = json.loads((args.trial / "results.json").read_text())
    rows, results = value["rows"], value["results"]
    traces = {}
    bounds = {}
    for name in NAMES:
        maxima = []
        for repeat in range(3):
            with np.load(args.trial / f"{name}_repeat{repeat}.npz", allow_pickle=False) as saved:
                trace = {key: saved[key] for key in saved.files}
            if repeat == 0:
                traces[name] = trace
            mask = trace["alive"].astype(bool)
            residual = trace["targets"][..., 14:] - trace["reference_q"][..., 14:]
            maxima.append(float(np.max(np.abs(residual[mask]))))
        bounds[name] = maxima
    if max(bounds["reference"]) > 2e-6 or max(bounds["residual015"]) > .150002:
        raise AssertionError("Recorded physical PD targets violated the requested residual bounds")
    metrics = ("root_xy_rmse", "upper_rmse", "wrist_relative_rmse", "leg_rmse",
               "target_oscillation", "loaded_slip")
    records = []
    for repeat in range(3):
        for name in NAMES[1:]:
            comparison = results[name][repeat]["RF11_common_prefix_comparison"]
            for metric, pair in comparison["measurements"].items():
                records.append(dict(condition=name, repeat=repeat, metric=metric,
                                    **pair, ratio=comparison["ratios"][metric]))
    with (args.trial / "paired_metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from evaluation import local_wrists

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, metric, title in zip(axes[0], metrics[:3],
            ("Validation Root XY RMSE (m)", "Validation Upper RMSE (rad)", "Validation Local Wrist RMSE (m)")):
        # Paired prefix differs by candidate; show both members of each pair.
        for x, name in enumerate(NAMES[1:]):
            pairs = [r["RF11_common_prefix_comparison"]["measurements"][metric] for r in results[name]]
            for offset, key, color, label in ((-.16, "baseline", COLORS[0], "RF11 paired prefix"),
                                             (.16, "candidate", COLORS[x + 1], LABELS[x + 1])):
                vals = [p[key] for p in pairs]
                axis.scatter(np.full(3, x + offset), vals, color=color, s=25)
                axis.plot([x + offset - .08, x + offset + .08], [np.mean(vals)] * 2, color=color, lw=2)
        axis.set_xticks([0, 1], LABELS[1:])
        axis.set_ylim(bottom=0)
        axis.set_title(title, fontsize=11)
        axis.grid(alpha=.2)
    index = next(i for i, row in enumerate(rows) if row["name"] == FIXED[2])
    ref = traces["RF11"]
    total = int(ref["reference_steps"][index])
    wrists = local_wrists(ref["reference_wrists"][:total, index], ref["reference_p"][:total, index],
                          ref["reference_quat"][:total, index])
    for side, axis in enumerate(axes[1, :2]):
        axis.plot((np.arange(total) + 1) * .02, wrists[:, side, 2], color="black", ls="--", label="Reference")
        for name, label, color in zip(NAMES, LABELS, COLORS):
            trace = traces[name]
            count = int(trace["steps"][index])
            state = trace["states"][:count, index]
            actual = local_wrists(trace["wrists"][:count, index], state[:, :3], state[:, 3:7])
            axis.plot((np.arange(count) + 1) * .02, actual[:, side, 2], color=color, label=label)
            if bool(trace["failed"][index]):
                axis.scatter([count * .02], [actual[-1, side, 2]], marker="x", color=color, s=50)
        axis.set_title(("Left", "Right")[side] + " Wrist Height Relative to Pelvis (m)", fontsize=11)
        axis.set_xlabel("Seconds | fixed raised-hands clip, repeat 0")
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    axis = axes[1, 2]
    for x, name, color in zip(range(3), NAMES, COLORS):
        failures = [r["summary"]["validation"]["failures"] for r in results[name]]
        axis.scatter(np.full(3, x) + np.linspace(-.08, .08, 3), failures, color=color)
    axis.set_xticks(range(3), LABELS)
    axis.set_ylim(-.2, 3)
    axis.set_title("Validation Early Terminations / 30 (3 repeats)", fontsize=11)
    axis.grid(alpha=.2)
    figure.suptitle("UR01 | Frozen RF11 | 0 optimizer steps | 33 diagnostic environments x 3 repeats", fontsize=13)
    for extension in ("png", "pdf"):
        figure.savefig(args.trial / ("physical_comparison." + extension), dpi=160)
    plt.close(figure)
    media = render(args.runtime, args.trial, rows, traces)
    save(args.trial / "delivery.json", dict(status="COMPLETED", new_physics_steps=0,
        new_optimizer_steps=0, source=identity(args.trial / "results.json"),
        recorded_upper_target_max_abs_residual_rad=bounds, media=media,
        artifacts=[identity(args.trial / name) for name in
                   ("physical_comparison.png", "physical_comparison.pdf", "paired_metrics.csv")]))


if __name__ == "__main__":
    main()
