"""Plot the settled same-weight precision diagnostic, without new model calls."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from deliver_wm_probe import METRICS


def deliver(source, parent_probe, output):
    paths = [source/n for n in ("result.json", "precision.guard.json", "precision_result.json")]+[parent_probe]
    before = {str(p):p.read_bytes() for p in paths}
    campaign, guard, probe, native = [json.loads(before[str(p)]) for p in paths]
    raw = before[str(paths[2])]
    if (campaign.get("status") != "WM_PRECISION_COMPLETED" or campaign.get("source_integrity") != "PASSED"
            or campaign.get("cost_settlement", {}).get("state") != "SETTLED"
            or guard.get("status") != "WORKER_COMPLETED"
            or guard.get("cleanup", {}).get("worker_reaped") is not True
            or guard.get("cleanup", {}).get("worker_return_code") != 0
            or len(raw) != campaign["precision_result"]["bytes"]
            or hashlib.sha256(raw).hexdigest() != campaign["precision_result"]["sha256"]
            or hashlib.sha256(before[str(parent_probe)]).hexdigest() != probe["parent_result"]["sha256"]
            or len(before[str(parent_probe)]) != probe["parent_result"]["bytes"]
            or native.get("diagnostic_screen") != "FAILED"
            or probe.get("status") != "WM_PRECISION_COMPLETED" or probe.get("source_integrity") != "PASSED"
            or probe.get("sampled_transition_count") != 256 or probe.get("cuda_wm_step_forward_calls") != 376
            or probe.get("cuda_backward_calls") != 8 or probe.get("actual_optimizer_updates") != 0
            or probe.get("new_physical_transitions") != 0 or probe.get("test_sources_read") is not False
            or probe.get("selected_rows") != native.get("selected_rows")):
        raise ValueError("Require the real settled same-row precision comparison and original failed native probe")
    if not output.is_absolute() or output.exists():
        raise FileExistsError("Preserve existing figures and choose a fresh absolute output")
    validation = {}
    for label, value in probe["calibration"].items():
        validation[label] = [next(g for g in value["groups"] if g["origin"] == origin
            and g["split"] == "validation") for origin in ("original", "mixed")]
        if any(g["sources"] != 6 or g["source_weighting"] != "equal" for g in validation[label]):
            raise ValueError("Retain all six equally weighted validation sources per origin")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, (key, title, units, scale) in zip(axes.flat, METRICS):
        groups = validation["native"]
        hold = [g["identity"][key]*scale for g in groups]
        values = [[g["wm"][key]*scale for g in validation[label]] for label in ("native", "shadow")]
        for shift, data, label, color in ((-.25, values[0], "Native BF16", "#bb6949"),
                (0., values[1], "Identical weights FP32", "#247d78"),
                (.25, hold, "Hold current physical state", "#73787c")):
            ax.bar([shift, 1+shift], data, .23, label=label, color=color)
        ax.set_xticks([0, 1], ["Original states", "Mixed states"])
        ax.set(title=title, ylabel=units, ylim=(0, max(hold+values[0]+values[1])*1.25))
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    count = lambda gs:sum(r["finite_difference_passed"] for g in gs for r in g["present_joints"])
    native_passed, shadow_passed = count(native["gradients"]), count(probe["shadow_gradients"])
    fig.suptitle("Frozen32k Q1 WM | 256 identical saved transitions | six equal-weight validation clips per origin\n"
        "Physical-gradient checks: BF16 %d/88, FP32 %d/88 | no training or new physics" %
        (native_passed, shadow_passed), fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(output/("wm_precision_six_panel."+suffix), dpi=160)
    plt.close(fig)
    rows = [dict(precision=label, origin=g["origin"], split=g["split"], metric=key,
        unit=units, wm=g["wm"][key]*scale, identity=g["identity"][key]*scale,
        ratio=g["ratios"][key], equally_weighted_sources=g["sources"])
        for label, value in probe["calibration"].items() for g in value["groups"]
        for key, _, units, scale in METRICS]
    gradients = [dict(precision=label, origin=g["origin"], split=g["split"], joint=r["joint"],
        autograd=r["autograd"], epsilon=fd["epsilon"], finite_difference=fd["finite_difference"],
        absolute_error=fd["absolute_error"], epsilon_passed=fd["passed"],
        joint_passed=r["finite_difference_passed"])
        for label, gs in (("native_saved", native["gradients"]), ("shadow", probe["shadow_gradients"]))
        for g in gs for r in g["present_joints"] for fd in r["finite_differences"]]
    for name, values in (("prediction_groups.csv", rows), ("physical_gradient_checks.csv", gradients)):
        with (output/name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    if any(Path(p).read_bytes() != raw for p, raw in before.items()):
        raise ValueError("Diagnostic sources changed while plotting")
    audit = dict(stage="SAVED_WM_PRECISION_SIX_PANEL", diagnostic_screen=probe["diagnostic_screen"],
        native_original_diagnostic_screen=native["diagnostic_screen"], native_gradient_passed=native_passed,
        shadow_gradient_passed=shadow_passed, gradient_joint_checks=88, sampled_transitions=256,
        native_reproduction_passed=probe["native_reproduction_passed"],
        shadow_prediction_preservation_passed=probe["shadow_prediction_preservation"]["passed"],
        sources={p:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for p,raw in before.items()},
        new_cuda_calls=0, new_training=0, new_physics=0, training_curve=False,
        natural_teleoperation_qualified=False, hardware_latency_ms=None, user_accepted_checkpoint=None)
    (output/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source", "parent-probe", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(deliver(args.source,args.parent_probe,args.output)))
