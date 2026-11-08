"""Compare saved restoration-gated forks on identical nonterminal support."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"supermimic_teleop"))
from tracking_response import analyze_signal

SERVER = Path("/opt/data/private/ZHY")
EXPERIMENT = "Q1_LOW_LATENCY_CKPT32000_TEACHER_RECOVERY01_RESUME1_20261108"
ARMS = ("student_f0", "privileged_teacher_h10")
COLORS = ("#b04e4e", "#237b68")


def read(path):
    return json.loads(path.read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(path):
    content = path.read_bytes()
    return dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


def saved_identity(packet, record):
    relative = Path(record["path"]).relative_to(SERVER)
    path = packet/relative
    require(identity(path) == {k:record[k] for k in ("bytes", "sha256")}, "Saved identity differs: "+str(relative))
    return path


def metrics(arrays, steps, upper):
    require(type(steps) is int and 0 < steps < len(arrays["actual"]), "Require a nonempty valid physical prefix")
    actual, reference = (arrays[k][:steps+1] for k in ("actual", "reference"))
    heights, forces, slip, tilt = (arrays[k][:steps+1] for k in ("sole_height", "load", "material_point_slip", "tilt"))
    loaded = forces[1:] >= 5.
    loaded_slip = [float(slip[1:, i][loaded[:, i]].mean()) if loaded[:, i].any() else None for i in range(2)]
    events = sorted((int(i), foot) for foot in range(2)
        for i in np.flatnonzero((heights[1:, foot] > .01) & (heights[:-1, foot] <= .01))+1)
    target_q = reference[1:, 3:7]
    target_tilt = np.arccos(np.clip(1-2*(target_q[:, 1]**2+target_q[:, 2]**2), -1, 1))
    jumps = np.linalg.norm(np.diff(arrays["canonical_action"][:steps], axis=0), axis=1)
    return dict(steps=steps, duration_s=steps*.02,
        upper_joint_rmse_rad=float(np.sqrt(np.mean((actual[1:, 7:29][:, upper]-reference[1:, 7:29][:, upper])**2))),
        loaded_slip_mean_m_s=loaded_slip, loaded_samples=loaded.sum(axis=0).tolist(),
        equal_foot_loaded_slip_mean_m_s=None if any(v is None for v in loaded_slip) else float(np.mean(loaded_slip)),
        torso_tilt_rms_rad=float(np.sqrt(np.mean(tilt[1:]**2))),
        tilt_error_rms_rad=float(np.sqrt(np.mean((tilt[1:]-target_tilt)**2))),
        sole_p95_m=np.quantile(heights[1:], .95, axis=0).tolist(),
        lift_events_above_10mm=len(events), alternating_lift_transitions=sum(a[1] != b[1] for a, b in zip(events, events[1:])),
        within_prefix_action_delta_l2_p95=None if not len(jumps) else float(np.quantile(jumps, .95)))


def common_prefix(rows, arrays, upper):
    require(len(rows) == len(arrays) == 2, "Require both physical arms")
    for row, data in zip(rows, arrays):
        attempted, failed = row["attempted_steps"], row["failure_frame"]
        require(attempted == len(data["canonical_action"]) and data["actual"].shape == (attempted+1, 79)
            and data["reference"].shape == data["actual"].shape
            and all(np.isfinite(v).all() for v in data.values()), "Invalid saved fork arrays")
        expected = attempted if failed is None else max(0, attempted-1)
        require(row["valid_prefix_steps"] == expected and (failed is None or failed == row["anchor"]+attempted),
            "Do not count a terminating transition as valid support")
        require(row["complete_short_fork"] == (failed is None and attempted == row["supported_steps"]), "Fork completion differs")
    require(rows[0]["anchor"] == rows[1]["anchor"] and rows[0]["clip"] == rows[1]["clip"]
        and np.array_equal(arrays[0]["actual"][0], arrays[1]["actual"][0]), "Physical arms have different initial states")
    n = min(r["valid_prefix_steps"] for r in rows)
    require(n > 0 and np.array_equal(arrays[0]["reference"][:n+1], arrays[1]["reference"][:n+1]), "Missing identical common target support")
    return n, [metrics(data, n, upper) for data in arrays]


def figures(output, pairs, data, upper, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def finish(fig, stem):
        for ax in fig.axes:
            ax.grid(alpha=.2)
        for suffix in ("png", "pdf"):
            fig.savefig(output/(stem+"."+suffix), dpi=155)
        plt.close(fig)

    for pair, arrays in zip(pairs, data):
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
        for arm, row, a, color in zip(ARMS, pair["arms"], arrays, COLORS):
            t = np.arange(len(a["actual"]))* .02
            error = np.sqrt(np.mean((a["actual"][:, 7:29][:, upper]-a["reference"][:, 7:29][:, upper])**2, axis=1))
            axes[0, 0].plot(t, error, color=color, label=arm)
            for foot, style in enumerate(("-", "--")):
                axes[0, 1].plot(t, 1000*a["sole_height"][:, foot], color=color, linestyle=style,
                    label=arm+(" L" if foot == 0 else " R"))
                loaded = np.where(a["load"][:, foot] >= 5, 1000*a["material_point_slip"][:, foot], np.nan)
                axes[0, 2].plot(t, loaded, color=color, linestyle=style, label=arm+(" L" if foot == 0 else " R"))
            axes[1, 0].plot(t, a["tilt"], color=color, label=arm)
            actions = a["canonical_action"]
            axes[1, 1].plot(np.arange(1, len(actions))*.02, np.linalg.norm(np.diff(actions, axis=0), axis=1), color=color, label=arm)
            for j, style in ((upper[0], "-"), (upper[4], "--")):
                axes[1, 2].plot(t, a["actual"][:, 7+j], color=color, linestyle=style, label=arm+(" L" if j == upper[0] else " R"))
            if row["failure_frame"] is not None:
                axes[0, 0].scatter([t[-1]], [error[-1]], color=color, marker="x", zorder=5)
        a = arrays[0]
        t = np.arange(len(a["reference"]))* .02
        for j, style in ((upper[0], "-"), (upper[4], "--")):
            axes[1, 2].plot(t, a["reference"][:, 7+j], color="#333333", linestyle=style, alpha=.65,
                label="reference "+("L" if j == upper[0] else "R"))
        for ax, title, ylabel in zip(axes.flat,
                ("Upper joint tracking", "Sole clearance (solid L, dashed R)", "Slip where last-substep load >=5N",
                 "Torso tilt", "Action changes inside the saved fork", "Shoulder pitch timing (not hardware latency)"),
                ("RMSE (rad)", "mm", "mm/s", "rad", "canonical action L2", "rad")):
            ax.axvline(pair["common_valid_steps"]*.02, color="#777777", linestyle=":", linewidth=1)
            ax.set(title=title, xlabel="Seconds from identical anchor", ylabel=ylabel)
            ax.legend(fontsize=6)
        fig.suptitle(f"{pair['clip']} | {pair['split']} | anchor {pair['anchor']} | S/T valid "
            f"{pair['arms'][0]['valid_prefix_steps']}/{pair['arms'][1]['valid_prefix_steps']} | common {pair['common_valid_steps']} steps; x=termination", fontsize=11)
        finish(fig, f"anchor_{pair['job_index']:02d}_six_panel")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    x = np.arange(len(pairs))
    panels = (("valid_prefix_steps", "Nonterminating steps (maximum 80)", "steps"),
        ("upper_joint_rmse_rad", "Upper tracking on common valid support", "rad"),
        ("equal_foot_loaded_slip_mean_m_s", "Equal-foot loaded slip on common support", "m/s"),
        ("torso_tilt_rms_rad", "Torso tilt on common valid support", "rad"),
        ("within_prefix_action_delta_l2_p95", "Within-prefix action jump p95", "canonical action L2"),
        ("lift_events_above_10mm", "Lift events on common valid support", "events"))
    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        for i, (arm, color) in enumerate(zip(ARMS, COLORS)):
            values = [p["arms"][i][key] if key == "valid_prefix_steps" else p["matched"][i][key] for p in pairs]
            ax.bar(x+(i-.5)*.36, [np.nan if v is None else v for v in values], .36, color=color, label=arm)
        ax.set(title=title, ylabel=ylabel, xlabel="Fixed anchor; same clip anchors are dependent")
        ax.set_xticks(x, [f"{p['clip'].split('__')[-1]}\n{p['anchor']}" for p in pairs], fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle("Q1 teacher recovery | 8 fixed anchors / 4 clips | matched valid horizons exclude termination | no new training", fontsize=12)
    finish(fig, "overview_six_panel")


def bundle_saved(packet, output, archive):
    summary = read(output/"summary.json")
    require(summary.get("status") == "PASSED" and summary.get("stage") == "TEACHER_RECOVERY_MATCHED_SAVED_REVIEW"
        and summary.get("new_policy_calls") == summary.get("new_physical_steps") == summary.get("new_training_updates") == 0,
        "Package only the completed saved analysis")
    all_files = sorted(p for p in packet.rglob("*") if p.is_file())
    require(summary["source_files"] == {str(p.relative_to(packet)):identity(p) for p in all_files}, "Raw package source changed")
    expected = [f"anchor_{i:02d}_six_panel.{s}" for i in range(8) for s in ("png", "pdf")]
    expected += ["overview_six_panel.png", "overview_six_panel.pdf", "matched_prefixes.csv", "README.md"]
    require(all((output/name).is_file() for name in expected), "Saved figures or table missing")
    archive.parent.mkdir(exist_ok=True)
    # TarInfo defaults to1970 for the ledger snapshot; ZIP needs a representable metadata date.
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, strict_timestamps=False) as bundle:
        for p in sorted(output.iterdir()):
            if p.is_file(): bundle.write(p, p.name)
        for p in all_files: bundle.write(p, "raw/"+str(p.relative_to(packet)))
    with zipfile.ZipFile(archive) as bundle:
        require(bundle.testzip() is None and len(bundle.namelist()) == len(set(bundle.namelist())), "Archive CRC or duplicate member failure")
        for p in all_files:
            require(bundle.read("raw/"+str(p.relative_to(packet))) == p.read_bytes(), "Archive raw bytes changed")
    return dict(status="PASSED", counts=summary["counts"], source_files=len(all_files),
        archive=dict(path=str(archive), **identity(archive)), packaging_code_identity=identity(Path(__file__)),
        new_policy_calls=0, new_physical_steps=0, new_training_updates=0)


def deliver(packet, output, archive):
    require(not output.exists() and not archive.exists(), "Use fresh saved-analysis output and archive")
    receipt_path = packet/"logs/session_records/q1_teacher_recovery_completed_20261108.json"
    receipt = read(receipt_path)
    require(receipt.get("status") == "COMPLETED_AND_REAPED" and receipt.get("completed_forks") == 16
        and receipt.get("restoration_gates_passed") == 8 and receipt.get("outstanding_reservations") == 0
        and receipt.get("gpu_lock_free") is True and not receipt.get("gpu_compute_pids")
        and receipt.get("original_shared_training_resumed") is False, "Wait for actual completion, source and cleanup gates")
    source = packet/"data/supermimic_for_q1"/EXPERIMENT
    result = read(saved_identity(packet, receipt["result"]))
    value = read(saved_identity(packet, receipt["fork_result"]))
    guard = read(saved_identity(packet, receipt["guard"]))
    metadata = read(saved_identity(packet, receipt["physical_joint_metadata"]))
    require(result.get("status") == "COMPLETED" and result.get("source_integrity") == "PASSED"
        and result.get("cost_settlement", {}).get("state") == "SETTLED"
        and guard.get("status") == "WORKER_COMPLETED" and guard.get("cleanup", {}).get("worker_reaped") is True
        and guard["cleanup"]["worker_return_code"] == 0, "Missing saved terminal gates")
    upper, names = receipt["upper_joint_indices"], receipt["upper_joint_names"]
    require(upper == metadata["cases"][0]["upper_joint_indices"] and names == metadata["cases"][0]["upper_joint_names"], "Physical joint order differs")
    gates = value["restoration_gates"]
    require(len(gates) == 8 and {r["job_index"] for r in gates} == set(range(8))
        and all(r.get("full_snapshot_next_state_bitwise_equal") is True and r.get("prefix_max_abs_difference") == 0 for r in gates), "Full-state restoration gate missing")
    cases = value["cases"]
    require(len(cases) == 16 and {(r["job_index"], r["arm"]) for r in cases} == {(i, a) for i in range(8) for a in ARMS}, "Missing or duplicate fixed fork")
    all_files = sorted(p for p in packet.rglob("*") if p.is_file())
    before = {str(p.relative_to(packet)):identity(p) for p in all_files}
    pairs, arrays_by_pair = [], []
    for index in range(8):
        rows = [next(r for r in cases if (r["job_index"], r["arm"]) == (index, arm)) for arm in ARMS]
        arrays = []
        for row in rows:
            p = saved_identity(packet, row["trace"])
            with np.load(p, allow_pickle=False) as data:
                a = {k:data[k].copy() for k in data.files}
            timing = read(saved_identity(packet, row["timing"]))
            require(len(timing) == row["attempted_steps"] and all(t["frame"] == row["anchor"]+i for i, t in enumerate(timing))
                and all(t.get("human_real_future_frames") == (0 if row["arm"] == ARMS[0] else 10) for t in timing), "Saved target visibility or sequential timing differs")
            recomputed = metrics(a, row["attempted_steps"], upper)
            require(np.isclose(recomputed["upper_joint_rmse_rad"], row["diagnostics"]["upper_joint_rmse_rad"], atol=1e-8, rtol=1e-6), "Saved physical joint order does not reproduce original upper metric")
            arrays.append(a)
        steps, matched = common_prefix(rows, arrays, upper)
        phase = [[dict(joint=name, **analyze_signal(a["reference"][:steps+1, 7+j], a["actual"][:steps+1, 7+j]))
            for j, name in zip(upper, names)] for a in arrays]
        pairs.append(dict(job_index=index, clip=rows[0]["clip"], split=rows[0]["split"], anchor=rows[0]["anchor"],
            common_valid_steps=steps, arms=rows, matched=matched, descriptive_upper_timing=phase,
            valid_step_difference_teacher_minus_student=rows[1]["valid_prefix_steps"]-rows[0]["valid_prefix_steps"],
            common_scope="Identical anchor and reference, excludes both terminating transitions; never a full-clip metric"))
        arrays_by_pair.append(arrays)
    counts = dict(student_complete=sum(p["arms"][0]["complete_short_fork"] for p in pairs),
        teacher_complete=sum(p["arms"][1]["complete_short_fork"] for p in pairs),
        teacher_longer=sum(p["valid_step_difference_teacher_minus_student"] > 0 for p in pairs),
        teacher_shorter=sum(p["valid_step_difference_teacher_minus_student"] < 0 for p in pairs),
        equal_valid_steps=sum(p["valid_step_difference_teacher_minus_student"] == 0 for p in pairs))
    output.mkdir(parents=True)
    figures(output, pairs, arrays_by_pair, upper, names)
    flat = []
    for pair in pairs:
        for arm, row, metric in zip(ARMS, pair["arms"], pair["matched"]):
            flat.append(dict(job_index=pair["job_index"], clip=pair["clip"], split=pair["split"], anchor=pair["anchor"],
                arm=arm, attempted_steps=row["attempted_steps"], valid_steps=row["valid_prefix_steps"],
                common_steps=pair["common_valid_steps"], complete_short_fork=row["complete_short_fork"],
                failure_frame=row["failure_frame"], **{k:v for k,v in metric.items() if not isinstance(v, list)}))
    with (output/"matched_prefixes.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    summary = dict(stage="TEACHER_RECOVERY_MATCHED_SAVED_REVIEW", status="PASSED", counts=counts, pairs=pairs,
        restoration_gates=gates, source_receipt=identity(receipt_path), source_files=before,
        code_identity=identity(Path(__file__)), timing_code_identity=identity(HERE.parent/"supermimic_teleop/tracking_response.py"),
        new_policy_calls=0, new_physical_steps=0, new_training_updates=0, new_rendered_frames=0,
        original_fork_physical_steps=value["actual_physical_steps"], original_fork_model_forwards=value["actual_model_forwards"],
        actual_fork_gpu_charged_seconds=result["cost_settlement"]["charged_seconds"],
        cumulative_gpu_charged_seconds=result["cumulative_gpu_charged_seconds"],
        upper_timing_is_descriptive_only=True, hardware_end_to_end_latency_ms=None,
        teacher_is_deployable=False, natural_teleoperation_qualified=False, user_accepted_checkpoint=None)
    (output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    lines = ["# Q1 Teacher Recovery: Saved Evidence", "",
        "Eight fixed dependent anchors on four train/validation clips; maximum 80 steps / 1.6 seconds per fork.", "",
        f"Complete short forks: student {counts['student_complete']}/8; privileged H10 teacher {counts['teacher_complete']}/8.",
        f"Teacher valid support: longer {counts['teacher_longer']}, shorter {counts['teacher_shorter']}, equal {counts['equal_valid_steps']}.", "",
        "All numerical comparisons use identical nonterminating horizons. Plots preserve full attempted traces and termination points.",
        "This is not full-clip success, randomized recovery evidence, hardware latency or deployable f0 teleoperation.", "",
        "![Overview](overview_six_panel.png)", "", "[Matched CSV](matched_prefixes.csv) | [Full summary](summary.json)", ""]
    for pair in pairs:
        stem = f"anchor_{pair['job_index']:02d}_six_panel"
        lines += [f"## {pair['clip']} / {pair['anchor']} ({pair['split']})", "", f"![Six-panel]({stem}.png)", "", f"[PDF]({stem}.pdf)", ""]
    lines += ["## Raw Provenance", "", "No new video was rendered; the archive contains the original saved numerical traces and receipts.", ""]
    for p in all_files:
        rel = str(p.relative_to(packet)); lines.append(f"- [{rel}](raw/{rel})")
    (output/"README.md").write_text("\n".join(lines)+"\n")
    require(before == {str(p.relative_to(packet)):identity(p) for p in all_files}, "Raw source changed during saved analysis")
    return bundle_saved(packet, output, archive)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("packet", "output", "archive"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()
    operation = bundle_saved if args.package_only else deliver
    print(json.dumps(operation(args.packet, args.output, args.archive), indent=2))
