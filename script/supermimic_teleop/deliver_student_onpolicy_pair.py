"""Plot and package actual equal-source training evidence, without model calls."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

ARMS = ("saved_only", "onpolicy_mix")


def identity(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def deliver(source, output, proof_path, archive):
    def read(name):
        return json.loads((source/name).read_text())
    campaign, fit, guard = read("result.json"), read("fit_result.json"), read("fit.guard.json")
    proof = json.loads(proof_path.read_text())
    if (campaign.get("status") != "ONPOLICY_PAIR_TRAINED_PENDING_CLOSED_LOOP"
            or campaign.get("source_integrity") != "PASSED"
            or campaign.get("cost_settlement", {}).get("state") != "SETTLED"
            or guard.get("status") != "WORKER_COMPLETED"
            or guard.get("cleanup", {}).get("worker_reaped") is not True
            or guard["cleanup"].get("worker_return_code") != 0
            or fit.get("matched_initialization_capacity_and_source_slots") is not True
            or fit.get("actual_optimizer_updates") != 6000 or fit.get("human_real_future_frames") != 0):
        raise ValueError("Require complete actual paired fitting, source gate, settlement and cleanup")
    if (proof.get("status") != "PASSED" or proof.get("result") != campaign
            or len(proof.get("exact_policy_proofs", [])) != 2
            or not all(r.get("initial_policy_equal") is True and r.get("selected_update") == 0
                for r in proof["exact_policy_proofs"])
            or proof.get("reused_existing_complete_cases") != 0 or proof.get("reused_existing_cases") != 7):
        raise ValueError("Require the actual CPU policy-equality proof before reusing prior outcomes")
    if output.exists():
        raise FileExistsError("Use a fresh paired figure root")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    traces = {arm:read(arm+"/training_trace.json") for arm in ARMS}
    runs = {r["variant"]:r for r in fit["arms"]}
    for key in ("initial_adapter_sha256", "initial_validation_risk", "initial_fixed_anchor_risks",
            "schedule_sha256", "source_sample_counts", "trainable_parameters"):
        if runs[ARMS[0]][key] != runs[ARMS[1]][key]:
            raise ValueError("Paired control mismatch: "+key)
    inputs = [identity(p) for p in sorted(source.rglob("*")) if p.is_file()]
    output.mkdir(parents=True)
    stats = []
    for arm in ARMS:
        rows, item = traces[arm], runs[arm]
        if ([r["update"] for r in rows] != list(range(0, 3001, 500)) or item["actual_adam_steps"] != [3000]*4):
            raise ValueError("Missing actual equal Adam exposure or validation records")
        allowed = [r for r in rows if r["fixed_anchor_non_regression"]]
        selected = min(allowed, key=lambda r:r["validation_risk"])
        if selected["update"] != item["selected_update"]:
            raise ValueError("Checkpoint selection differs from registered validation rule")
        x = [r["update"] for r in rows]
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        for name, color in zip(ARMS, ("#286b9a", "#19826b")):
            axes[0, 0].plot(x, [r["validation_risk"] for r in traces[name]], label=name, color=color)
        axes[0, 0].axvline(item["selected_update"], color="#b95148", linestyle="--")
        axes[0, 0].set(title="Paired validation (old/pure 50:50)", ylabel="Group-equal action MSE")
        axes[0, 0].legend(fontsize=8)
        for key, label in (("old_matched_validation_risk", "Matched old states"), ("pure_validation_risk", "Pure-student states")):
            axes[0, 1].plot(x, [r[key] for r in rows], label=label)
        axes[0, 1].set(title="Separate matched validation distributions", ylabel="Action MSE")
        axes[0, 1].legend(fontsize=8)
        for key, label in (("full_old_validation_risk", "All old"),
                ("fixed_anchor_r1_risk", "R1"), ("fixed_anchor_mixed_risk", "Mixed")):
            line, = axes[0, 2].plot(x, [r[key] for r in rows], label=label)
            axes[0, 2].axhline(rows[0][key], color=line.get_color(), linestyle=":", linewidth=.8)
        axes[0, 2].set(title="All three anchors must not regress", ylabel="Action MSE")
        axes[0, 2].legend(fontsize=8)
        observed = [r for r in rows if r["loss"] is not None]
        axes[1, 0].plot([r["update"] for r in observed], [r["loss"] for r in observed], color="#286b9a")
        axes[1, 0].set(title="Recorded batch loss (every 500 updates)", ylabel="Action MSE")
        axes[1, 1].plot([r["update"] for r in observed], [r["gradient_norm"] for r in observed], color="#9b6841")
        axes[1, 1].axhline(1., color="#555555", linestyle=":")
        axes[1, 1].set(title="Adapter gradient before clipping", ylabel="L2 norm")
        initial = {r["source_clip_id"]:r["action_mse"] for r in rows[0]["pure_validation"]}
        chosen = {r["source_clip_id"]:r["action_mse"] for r in selected["pure_validation"]}
        last = {r["source_clip_id"]:r["action_mse"] for r in rows[-1]["pure_validation"]}
        if set(initial) != set(chosen) or any(v <= 0. for v in initial.values()):
            raise ValueError("Pure-state source set or percentage denominator changed")
        names = sorted(initial)
        changes = [100*(chosen[k]/initial[k]-1) for k in names]
        last_changes = [100*(last[k]/initial[k]-1) for k in names]
        axes[1, 2].bar(range(len(names)), last_changes, label="Last (ineligible)",
            color=["#19826b" if v < 0 else "#b95148" for v in last_changes])
        axes[1, 2].plot(range(len(names)), changes, "o", color="#286b9a", label="Selected update0")
        axes[1, 2].axhline(0., color="#555555", linewidth=.8)
        axes[1, 2].set_xticks(range(len(names)), [str(k) for k in names])
        axes[1, 2].set(title="Pure-state validation by source", xlabel="Registered source index", ylabel="MSE change (%)")
        axes[1, 2].legend(fontsize=8)
        for ax in axes.flat:
            ax.grid(alpha=.2)
            if ax != axes[1, 2]:
                ax.set_xlabel("Additional Adam updates")
        charge = campaign["cost_settlement"]["charged_seconds"]
        fig.suptitle(f"Q1 f0 matched pair | {arm} | selected {item['selected_update']} | pair charge {charge}s | prior 0/7 reused", fontsize=13)
        for suffix in ("png", "pdf"):
            fig.savefig(output/(arm+"_six_panel."+suffix), dpi=160)
        plt.close(fig)
        fields = ("update", "validation_risk", "old_matched_validation_risk", "pure_validation_risk",
            "full_old_validation_risk", "fixed_anchor_r1_risk", "fixed_anchor_mixed_risk",
            "fixed_anchor_non_regression", "loss", "gradient_norm", "elapsed_seconds")
        with (output/(arm+"_curves.csv")).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            writer.writerows({k:r[k] for k in fields} for r in rows)
        stats.append(dict(arm=arm, selected_update=item["selected_update"],
            selected_adapter_unchanged=item["selected_adapter_unchanged"],
            validation_change_percent=100*(item["selected_validation_risk"]/item["initial_validation_risk"]-1),
            selected_pure_validation_risk=selected["pure_validation_risk"],
            pure_validation_change_percent=100*(selected["pure_validation_risk"]/rows[0]["pure_validation_risk"]-1),
            selected_fixed_anchor_risks=item["selected_fixed_anchor_risks"],
            actual_adam_steps=item["actual_adam_steps"], source_sample_counts=item["source_sample_counts"],
            selected_source_changes_percent=dict(zip(map(str, names), changes))))
    if any(identity(Path(r["path"])) != r for r in inputs):
        raise ValueError("Saved training evidence changed while plotting")
    result = dict(status="PASSED", stage="ONPOLICY_PAIR_SAVED_TRAINING_DELIVERY", arms=stats, source_inputs=inputs,
        original_paired_gpu_seconds=campaign["cost_settlement"]["charged_seconds"],
        cumulative_gpu_charged_seconds=campaign["cumulative_gpu_charged_seconds"],
        paired_source_slots_and_capacity_identical=True, teacher_recovery_ability_verified=False,
        loss_recorded_every_updates=500, raw_per_update_loss_available=False, reinforcement_episode_metrics_recorded=False,
        new_model_calls=0, new_optimizer_updates=0, new_physical_steps=0, new_rendered_frames=0,
        closed_loop_quality="EXACT_INITIAL_POLICY_PRIOR_0_OF_7_REUSED", policy_equality_proof=identity(proof_path),
        hardware_end_to_end_latency_ms=None, natural_teleoperation_qualified=False,
        original_shared_training_resumed=False, user_accepted_checkpoint=None)
    (output/"delivery.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    lines = ["# Q1 Pure-Student State Label Pair", "", "Actual equal-source training evidence. Selected policies exactly match the initial policy; the prior 0/7 complete outcome is reused after a CPU equality audit. No identical physical replay was repeated.",
        "", "Each arm received 3000 additional Adam updates. Half each batch shares the same old anchors; the other half uses source-matched old or pure-student states. Losses are recorded every 500 updates; episode rewards and raw per-update losses were not recorded.", ""]
    for row in stats:
        arm = row["arm"]
        lines += [f"## {arm}", "", f"Selected update {row['selected_update']}; eligible combined validation change {row['validation_change_percent']:.3f}%.",
            "", f"![Training six-panel]({arm}_six_panel.png)", "", f"[PDF]({arm}_six_panel.pdf) | [CSV]({arm}_curves.csv)", ""]
    lines += ["[Delivery audit](delivery.json)", "", "## Original Records", ""]
    raw_files = sorted(p for p in source.rglob("*") if p.is_file())
    lines += [f"- [{p.relative_to(source)}](raw/{p.relative_to(source)})" for p in raw_files]
    (output/"README.md").write_text("\n".join(lines)+"\n")
    archive.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as stream:
        for p in sorted(output.iterdir()):
            stream.write(p, p.name)
        for p in raw_files:
            stream.write(p, "raw/"+str(p.relative_to(source)))
    with zipfile.ZipFile(archive) as stream:
        if stream.testzip() is not None or len(stream.namelist()) != len(set(stream.namelist())):
            raise ValueError("Evidence CRC or member list failed")
        manifest = []
        for name in stream.namelist():
            b = stream.read(name); original = source/name[4:] if name.startswith("raw/") else output/name
            if b != original.read_bytes():
                raise ValueError("Archived evidence differs: "+name)
            manifest.append(dict(relative=name, bytes=len(b), sha256=hashlib.sha256(b).hexdigest()))
        links = re.findall(r"\]\(([^)]+)\)", stream.read("README.md").decode())
        if any(link not in stream.namelist() for link in links):
            raise ValueError("Broken README evidence link")
    result.update(archive=identity(archive), members=manifest, readme_links_checked=len(links))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--policy-proof", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(deliver(args.source, args.output, args.policy_proof, args.archive), indent=2, allow_nan=False))
