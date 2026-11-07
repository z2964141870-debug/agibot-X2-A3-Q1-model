"""Read paired saved residual evidence and produce training six-panel figures."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

ARMS = ("zero_history", "causal_history")


def deliver(source, output):
    campaign = json.loads((source/"result.json").read_text())
    fit = json.loads((source/"fit_result.json").read_text())
    guard = json.loads((source/"fit.guard.json").read_text())
    if (campaign.get("status") != "PAIRED_RESIDUAL_TRAINED_PENDING_CLOSED_LOOP"
            or campaign.get("source_integrity") != "PASSED"
            or campaign.get("cost_settlement", {}).get("state") != "SETTLED"
            or guard.get("status") != "WORKER_COMPLETED"
            or guard.get("cleanup", {}).get("worker_reaped") is not True
            or guard["cleanup"].get("worker_return_code") != 0
            or fit.get("matched_batch_order_and_exposure") is not True
            or fit.get("human_real_future_frames") != 0
            or fit.get("new_optimizer_updates") != 6000):
        raise ValueError("Require actual complete paired fit, source integrity and owned cleanup")
    if output.exists():
        raise FileExistsError("Use one fresh residual figure output")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    traces = {arm:json.loads((source/arm/"training_trace.json").read_text()) for arm in ARMS}
    results = {r["arm"]:r for r in fit["arms"]}
    if (set(results) != set(ARMS)
            or results[ARMS[0]]["batch_sha256"] != results[ARMS[1]]["batch_sha256"]
            or results[ARMS[0]]["source_sample_counts"] != results[ARMS[1]]["source_sample_counts"]):
        raise ValueError("The pair differs in exposure or sampling order")
    output.mkdir(parents=True)
    statistics = []
    for arm in ARMS:
        rows, item = traces[arm], results[arm]
        if (not rows or rows[0]["update"] != 0 or rows[-1]["update"] != 3000
                or item.get("actual_adam_steps") != [3000]*4
                or any(a["update"] >= b["update"] for a, b in zip(rows, rows[1:]))):
            raise ValueError("Missing actual equal training updates or optimizer counters")
        eligible = [r for r in rows if r["fixed_anchor_non_regression"]]
        chosen = min(eligible, key=lambda r:r["validation_risk"])
        if chosen["update"] != item["selected_update"]:
            raise ValueError("Selected adapter is not the eligible validation minimum")
        steps = [r["update"] for r in rows]
        observed = [r for r in rows if r["loss"] is not None]
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        ax = axes[0, 0]
        for name, color in zip(ARMS, ("#4074a7", "#16816a")):
            ax.plot([r["update"] for r in traces[name]], [r["validation_risk"] for r in traces[name]],
                label=name, color=color)
        ax.axvline(item["selected_update"], color="#cc5a4b", linestyle="--", label="This arm selected")
        ax.set(title="Expanded validation action MSE", xlabel="Additional Adam updates", ylabel="Group-equal MSE")
        ax.legend(fontsize=8)
        ax = axes[0, 1]
        for key, label in (("fixed_anchor_r1_risk", "Original R1 states"),
                ("fixed_anchor_mixed_risk", "Mixed states")):
            line, = ax.plot(steps, [r[key] for r in rows], label=label)
            ax.axhline(rows[0][key], color=line.get_color(), linestyle=":", linewidth=.8)
        ax.set(title="Fixed validation anchors (non-regression required)", xlabel="Additional Adam updates", ylabel="MSE")
        ax.legend(fontsize=8)
        ax = axes[0, 2]
        ax.plot([r["update"] for r in observed], [r["loss"] for r in observed], color="#4074a7")
        ax.set(title="Recorded training batch MSE (every 500 updates)", xlabel="Additional Adam updates", ylabel="MSE")
        ax = axes[1, 0]
        ax.plot([r["update"] for r in observed], [r["gradient_norm"] for r in observed], color="#9b624a")
        ax.axhline(1., color="#555555", linestyle=":")
        ax.set(title="Adapter gradient before clipping", xlabel="Additional Adam updates", ylabel="L2 norm")
        def groups(values):
            clips = {}
            for value in values:
                clips.setdefault(value["group_id"], {}).setdefault(value["source_clip_id"], []).append(value["action_mse"])
            return {key:float(np.mean([np.mean(v) for v in group.values()])) for key, group in clips.items()}
        initial, selected = groups(rows[0]["validation"]), groups(chosen["validation"])
        if set(initial) != set(selected) or any(v <= 0 for v in initial.values()):
            raise ValueError("Validation source set changed or percentage denominator is zero")
        names = sorted(initial)
        changes = [100*(selected[k]/initial[k]-1) for k in names]
        ax = axes[1, 1]
        ax.bar(np.arange(len(names)), changes, color=["#16816a" if v < 0 else "#cc5a4b" for v in changes])
        ax.axhline(0., color="#555555", linewidth=.8)
        ax.set_xticks(np.arange(len(names)), [str(i+1) for i in range(len(names))])
        ax.set(title="Selected vs update0, every validation group", xlabel="Registered group index", ylabel="Action MSE change (%)")
        counts = {i:0 for i in range(3)}
        for key, count in item["source_sample_counts"].items():
            counts[int(key.split(":")[1])] += count
        ax = axes[1, 2]
        ax.bar(range(3), list(counts.values()), color=("#4074a7", "#16816a", "#9b624a"))
        ax.set_xticks(range(3), ("R1", "Aggregate", "Coverage"))
        ax.set(title="Actual sample exposure (identical between arms)", xlabel="Preserved state origin", ylabel="Sampled records")
        for ax in axes.flat:
            ax.grid(alpha=.2)
        charge = campaign["cost_settlement"]["charged_seconds"]
        fig.suptitle(f"Q1 f0 residual | {arm} | selected {item['selected_update']} | paired charge {charge}s | closed-loop pending", fontsize=13)
        for suffix in ("png", "pdf"):
            fig.savefig(output/(arm+"_six_panel."+suffix), dpi=160)
        plt.close(fig)
        with (output/(arm+"_curves.csv")).open("x", newline="") as stream:
            fields = ("update", "validation_risk", "fixed_anchor_r1_risk", "fixed_anchor_mixed_risk",
                "fixed_anchor_non_regression", "loss", "gradient_norm", "elapsed_seconds")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key:row[key] for key in fields})
        statistics.append(dict(arm=arm, selected_update=item["selected_update"],
            actual_updates=item["actual_updates"], actual_adam_steps=item["actual_adam_steps"],
            selected_adapter_is_zero=item["selected_adapter_is_zero"],
            initial_validation_risk=item["initial_validation_risk"],
            selected_validation_risk=item["selected_validation_risk"],
            validation_change_percent=100*(item["selected_validation_risk"]/item["initial_validation_risk"]-1),
            group_changes_percent=dict(zip(names, changes)), trainable_parameters=item["trainable_parameters"],
            sampled_records=sum(counts.values()), initial_and_selected_anchor_risks=[item["initial_fixed_anchor_risks"],item["selected_fixed_anchor_risks"]]))
    raw_files = sorted(p for p in source.rglob("*") if p.is_file())
    record = dict(stage="F0_PROPRIO_SAVED_TRAINING_FIGURES", status="PASSED", arms=statistics,
        source_inputs=[dict(relative=str(p.relative_to(source)), bytes=p.stat().st_size,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in raw_files],
        original_fit_gpu_seconds=campaign["cost_settlement"]["charged_seconds"],
        cumulative_gpu_charged_seconds=campaign["cumulative_gpu_charged_seconds"],
        raw_training_loss_every_update_available=False, loss_recording_interval_updates=500,
        fabricated_500_update_smoothing=False, reinforcement_episode_metrics_not_recorded=True,
        human_real_future_frames=0, new_inference_calls=0, new_optimizer_updates=0,
        new_physical_steps=0, natural_teleoperation_qualified=False, hardware_end_to_end_latency_ms=None,
        user_accepted_checkpoint=None)
    (output/"delivery.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    lines = ["# Q1 f0 Proprioceptive Residual Pair", "", "Paired training evidence; closed-loop review pending.",
        "", "Only saved f0/current-history data were used. Training curves are recorded every 500 updates; per-episode RL metrics and per-update raw losses are unavailable.", ""]
    for item in statistics:
        arm = item["arm"]
        lines += [f"## {arm}", "", f"3000 new updates, selected {item['selected_update']}; eligible validation MSE change {item['validation_change_percent']:.3f}%.",
            "", f"![Training six-panel]({arm}_six_panel.png)", "", f"[PDF]({arm}_six_panel.pdf) | [CSV]({arm}_curves.csv)", ""]
    lines += ["## Provenance", "", "[Delivery record](delivery.json)", ""]
    for p in raw_files:
        rel = str(p.relative_to(source))
        lines.append(f"- [{rel}](raw/{rel})")
    (output/"README.md").write_text("\n".join(lines)+"\n")
    archive_path = output.parent.parent/"deliveries/Q1_F0_PROPRIO_TRAINING_EVIDENCE_20261108.zip"
    archive_path.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file():
                archive.write(path, path.name)
        for path in raw_files:
            archive.write(path, "raw/"+str(path.relative_to(source)))
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("Evidence archive CRC or member identity failed")
        for path in raw_files:
            if archive.read("raw/"+str(path.relative_to(source))) != path.read_bytes():
                raise ValueError("Archived raw evidence differs")
    record["archive"] = dict(path=str(archive_path), bytes=archive_path.stat().st_size,
        sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest())
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(deliver(args.source, args.output), indent=2, allow_nan=False))
