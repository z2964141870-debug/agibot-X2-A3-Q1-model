"""Six physical prediction panels from one completed, source-gated WM probe."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


METRICS = (
    ("joint_position_rad", "Joint position", "rad", 1.),
    ("body_position_m", "World body position", "mm", 1000.),
    ("joint_velocity_rad_s", "Joint velocity", "rad/s", 1.),
    ("body_orientation_rad", "Body orientation", "rad", 1.),
    ("body_linear_velocity_m_s", "World body linear velocity", "m/s", 1.),
    ("body_angular_velocity_rad_s", "World body angular velocity", "rad/s", 1.),
)


def deliver(source, output):
    paths = [source/name for name in ("result.json", "probe.guard.json", "probe_result.json")]
    before = {str(path):path.read_bytes() for path in paths}
    campaign, guard, probe = [json.loads(before[str(path)]) for path in paths]
    raw_probe = before[str(paths[-1])]
    if (campaign.get("status") != "WM_PROBE_COMPLETED" or campaign.get("source_integrity") != "PASSED"
            or campaign.get("cost_settlement", {}).get("state") != "SETTLED"
            or guard.get("status") != "WORKER_COMPLETED"
            or guard.get("cleanup", {}).get("worker_reaped") is not True
            or guard.get("cleanup", {}).get("worker_return_code") != 0
            or len(raw_probe) != campaign["probe_result"]["bytes"]
            or hashlib.sha256(raw_probe).hexdigest() != campaign["probe_result"]["sha256"]
            or probe.get("status") != "WM_PROBE_COMPLETED" or probe.get("source_integrity") != "PASSED"
            or probe.get("actual_optimizer_updates") != 0 or probe.get("new_physical_transitions") != 0
            or probe.get("test_sources_read") is not False or probe.get("sampled_transition_count") != 256
            or probe.get("cuda_wm_step_forward_calls") != 368):
        raise ValueError("Use the actual settled256-row diagnostic, including its failed scientific screen")
    if not output.is_absolute() or output.exists():
        raise FileExistsError("Use a fresh absolute diagnostic figure root")
    groups = probe["calibration"]["groups"]
    validation = [next(g for g in groups if g["origin"] == origin and g["split"] == "validation")
        for origin in ("original", "mixed")]
    if any(g["sources"] != 6 or g["source_weighting"] != "equal" for g in validation):
        raise ValueError("Both validation origins must retain all six equally weighted sources")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, (key, title, units, scale) in zip(axes.flat, METRICS):
        wm = [g["wm"][key]*scale for g in validation]
        hold = [g["identity"][key]*scale for g in validation]
        ax.bar([-.18, .82], wm, .34, label="Frozen WM (native BF16)", color="#247d78")
        ax.bar([.18, 1.18], hold, .34, label="Hold current physical state", color="#bb6949")
        ax.set_xticks([0, 1], ["Original states", "Mixed states"])
        ax.set(title=title, ylabel=units, ylim=(0, max(wm+hold)*1.28))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        for index, g in enumerate(validation):
            ax.text(index, max(wm[index], hold[index])*1.06,
                "WM/hold: %.3f" % g["ratios"][key], ha="center", fontsize=9)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Frozen32k Q1 WM | 256 saved transitions | validation: six equal-weight sources per origin\n"
        "One-step calibration passed; BF16 physical-gradient screen failed | no training or new physics",
        fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(output/("wm_physical_prediction_six_panel."+suffix), dpi=160)
    plt.close(fig)
    rows = [dict(origin=g["origin"], split=g["split"], metric=key, unit=units,
        wm=g["wm"][key]*scale, identity=g["identity"][key]*scale,
        ratio=g["ratios"][key], equally_weighted_sources=g["sources"])
        for g in groups for key, _, units, scale in METRICS]
    gradients = [dict(origin=g["origin"], split=g["split"], joint=row["joint"],
        autograd=row["autograd"], epsilon=fd["epsilon"], finite_difference=fd["finite_difference"],
        absolute_error=fd["absolute_error"], epsilon_passed=fd["passed"],
        joint_passed=row["finite_difference_passed"])
        for g in probe["gradients"] for row in g["present_joints"] for fd in row["finite_differences"]]
    for name, values in (("prediction_groups.csv", rows), ("physical_gradient_checks.csv", gradients)):
        with (output/name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    if any(Path(path).read_bytes() != raw for path, raw in before.items()):
        raise ValueError("Probe evidence changed while plotting")
    audit = dict(stage="SAVED_WM_PROBE_DIAGNOSTIC_SIX_PANEL", new_training=0, new_physics=0,
        new_cuda_calls=0, original_sampled_transitions=256, physical_gradient_screen=probe["physical_gradient_screen_passed"],
        calibration_screen=probe["calibration"]["calibration_screen_passed"], diagnostic_screen=probe["diagnostic_screen"],
        gradient_joint_checks=sum(len(g["present_joints"]) for g in probe["gradients"]),
        gradient_joint_checks_passed=sum(row["finite_difference_passed"]
            for g in probe["gradients"] for row in g["present_joints"]),
        sources={path:dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) for path, raw in before.items()},
        training_curve=False, hardware_latency_ms=None, natural_teleoperation_qualified=False,
        user_accepted_checkpoint=None)
    (output/"audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(deliver(args.source, args.output)))
