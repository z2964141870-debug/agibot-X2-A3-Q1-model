"""Read saved matched forks; distinguish global and pelvis-relative velocity errors."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import zipfile

import numpy as np

from deliver_teacher_recovery import ARMS, EXPERIMENT, identity, read, require, saved_identity

DT = .02
PREFIX_STEPS = (4, 10, 20, 40, 80)
LOAD_N = 5.


def velocity_series(body, target, load, dt=DT):
    body, target, load = (np.asarray(a, dtype=np.float64) for a in (body, target, load))
    require(body.shape == target.shape and body.ndim == 3 and body.shape[1:] == (27, 3)
        and len(body) >= 2 and load.shape == (len(body), 2)
        and all(np.isfinite(a).all() for a in (body, target, load))
        and np.isfinite(dt) and dt > 0., "Invalid paired saved body positions, loads or interval")
    require(not body[:, 0].any() and not target[:, 0].any(), "Body0 must be the saved static world")
    # Each row ends at the visible current frame; no centered difference or future sample.
    error = (np.diff(body[:, 1:], axis=0)-np.diff(target[:, 1:], axis=0))/dt
    root = error[:, 0]
    relative = error-root[:, None]
    root_energy = np.square(root).sum(axis=-1)
    relative_energy = np.square(relative).sum(axis=-1).mean(axis=-1)
    cross = 2.*np.einsum("nbi,ni->nb", relative, root).mean(axis=-1)
    world_energy = np.square(error).sum(axis=-1).mean(axis=-1)
    require(np.allclose(world_energy, root_energy+relative_energy+cross, atol=1e-10, rtol=1e-10),
        "World error decomposition does not close")
    modes = (load[1:, 0] >= LOAD_N).astype(int)+2*(load[1:, 1] >= LOAD_N).astype(int)
    return dict(world_energy=world_energy, root_energy=root_energy,
        relative_energy=relative_energy, cross=cross, mode=modes)


def summarize(series, steps):
    require(type(steps) is int and 0 < steps <= len(series["world_energy"]), "Require exact nonempty support")
    values = {k:v[:steps] for k,v in series.items()}
    result = dict(steps=steps, interval_s=steps*DT,
        world_body_velocity_rmse_m_s=float(np.sqrt(values["world_energy"].mean())),
        root_velocity_rmse_m_s=float(np.sqrt(values["root_energy"].mean())),
        relative_body_velocity_rmse_m_s=float(np.sqrt(values["relative_energy"].mean())),
        world_mean_squared_vector_error=float(values["world_energy"].mean()),
        root_mean_squared_vector_error=float(values["root_energy"].mean()),
        relative_mean_squared_vector_error=float(values["relative_energy"].mean()),
        signed_cross_term=float(values["cross"].mean()))
    result["load_modes"] = [dict(mode=m, samples=int(np.sum(values["mode"] == m)),
        world_body_velocity_rmse_m_s=float(np.sqrt(values["world_energy"][values["mode"] == m].mean()))
            if np.any(values["mode"] == m) else None) for m in range(4)]
    return result


def matched_support(rows, arrays, expected):
    require(len(rows) == len(arrays) == 2 and [r["arm"] for r in rows] == list(ARMS), "Require both ordered forks")
    for row, data in zip(rows, arrays):
        n = row["attempted_steps"]
        require(data["actual"].shape == data["reference"].shape == (n+1, 79)
            and data["canonical_action"].shape == (n, 29)
            and data["body_actual"].shape == data["body_reference"].shape == (n+1, 27, 3)
            and data["load"].shape == (n+1, 2)
            and all(np.isfinite(a).all() for a in data.values()), "Saved array shape or finiteness changed")
        require(np.array_equal(data["actual"][:, :3], data["body_actual"][:, 1])
            and np.array_equal(data["reference"][:, :3], data["body_reference"][:, 1]),
            "Body1 is not exactly the saved root translation")
        valid = n if row["failure_frame"] is None else max(0, n-1)
        require(row["valid_prefix_steps"] == valid
            and (row["failure_frame"] is None or row["failure_frame"] == row["anchor"]+n),
            "Terminating transitions may not enter the common prefix")
    n = min(r["valid_prefix_steps"] for r in rows)
    require(type(expected) is int and n == expected and n > 0
        and rows[0]["clip"] == rows[1]["clip"] and rows[0]["anchor"] == rows[1]["anchor"]
        and rows[0]["split"] == rows[1]["split"]
        and np.array_equal(arrays[0]["actual"][0], arrays[1]["actual"][0])
        and np.array_equal(arrays[0]["reference"][:n+1], arrays[1]["reference"][:n+1])
        and np.array_equal(arrays[0]["body_reference"][:n+1], arrays[1]["body_reference"][:n+1]),
        "Common nonterminal support differs from the already-audited paired result")
    return n


def draw(output, pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    labels = [("T:" if p["split"] == "train" else "V:")+p["clip"].split("__")[-1]+"/"+str(p["anchor"])
        for p in pairs]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    x = np.arange(len(pairs))
    for ax, key, title in zip(axes.flat[:3],
            ("world_body_velocity_rmse_m_s", "root_velocity_rmse_m_s", "relative_body_velocity_rmse_m_s"),
            ("World velocity error (26 moving bodies)", "Root translation velocity error",
             "Pelvis-relative velocity error (same bodies)")):
        for j, color in enumerate(("#ad5050", "#247e72")):
            ax.bar(x+(j-.5)*.36, [p["matched"][j][key] for p in pairs], .36, color=color, label=ARMS[j])
        ax.set(title=title, ylabel="Vector RMSE (m/s)")
        ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=7)
    matrix = np.full((len(pairs), len(PREFIX_STEPS)), np.nan)
    for i,pair in enumerate(pairs):
        for j,window in enumerate(pair["prefixes"]):
            if window["available"]:
                s,t = (a["world_body_velocity_rmse_m_s"] for a in window["arms"])
                if s > 0.: matrix[i, j] = 100.*(t/s-1.)
    ax = axes[1, 0]
    maximum = max(5., float(np.nanmax(abs(matrix))))
    plot = ax.imshow(np.ma.masked_invalid(matrix), cmap="RdYlGn_r",
        norm=TwoSlopeNorm(vmin=-maximum, vcenter=0., vmax=maximum), aspect="auto")
    for i,j in np.ndindex(matrix.shape):
        value = matrix[i, j]
        ax.text(j, i, "missing" if np.isnan(value) else f"{value:+.1f}%", ha="center", va="center", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xticks(range(len(PREFIX_STEPS)), [str(n*20) for n in PREFIX_STEPS])
    ax.set(title="Teacher vs student world velocity by exact prefix", xlabel="Prefix duration (ms); missing is excluded")
    fig.colorbar(plot, ax=ax, label="RMSE change (%); lower is better", shrink=.7)
    ax = axes[1, 1]
    for j,style in enumerate(("-", "--")):
        for key,color,label in (("root_mean_squared_vector_error", "#396dac", "Root"),
                ("relative_mean_squared_vector_error", "#31926b", "Relative"),
                ("signed_cross_term", "#aa6949", "Signed cross")):
            ax.plot(x, [p["matched"][j][key] for p in pairs], style, color=color,
                label=("S " if j == 0 else "T ")+label)
    ax.axhline(0., color="#555555", linewidth=.6)
    ax.set(title="Exact world MSE = root + relative + cross", ylabel="Mean squared vector error ((m/s)^2)")
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=.2)
    ax = axes[1, 2]
    xx = np.arange(2*len(pairs)); bottom = np.zeros(len(xx))
    for mode,color,label in ((0, "#b8bdc5", "No >=5N load"), (1, "#3b75a8", "Left only"),
            (2, "#d49a45", "Right only"), (3, "#4d9a78", "Both")):
        values = np.array([100.*p["matched"][j]["load_modes"][mode]["samples"]/p["common_valid_steps"]
            for p in pairs for j in range(2)])
        ax.bar(xx, values, bottom=bottom, color=color, label=label); bottom += values
    ax.set_xticks(np.arange(len(pairs))*2+.5, labels, rotation=45, ha="right", fontsize=8)
    ax.set(title="Last-substep load modes on common support (S/T)", ylabel="Intervals (%)", ylim=(0, 100))
    ax.legend(fontsize=7)
    fig.suptitle("Q1 saved-fork velocity review | backward 20ms intervals, not instantaneous velocity or latency\n"
        "8 dependent anchors / 4 train-validation clips; no new policy, physics or training; common support excludes termination", fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(output/("velocity_six_panel."+suffix), dpi=155)
    plt.close(fig)


def review(packet, prior_summary, plan_path, output, archive):
    started = time.monotonic()
    plan, prior = read(plan_path), read(prior_summary)
    require(plan["schema"] == "q1_saved_fork_velocity_review_v1" and plan["step_seconds"] == DT
        and plan["prefix_steps"] == list(PREFIX_STEPS) and plan["load_threshold_n"] == LOAD_N
        and plan["new_policy_calls"] == plan["new_physical_steps"] == plan["new_optimizer_updates"] == 0,
        "Registered local-only measurement changed")
    require(prior["stage"] == "TEACHER_RECOVERY_MATCHED_SAVED_REVIEW" and prior["status"] == "PASSED"
        and prior["new_policy_calls"] == prior["new_physical_steps"] == prior["new_training_updates"] == 0,
        "Require the existing complete paired analysis")
    paths = sorted(p for p in packet.rglob("*") if p.is_file())
    require(len(paths) == 62 and sum(p.stat().st_size for p in paths) <= 64*(1 << 20), "Original packet scope or64MiB bound changed")
    before = {str(p.relative_to(packet)):identity(p) for p in paths}
    require(before == prior["source_files"], "Original saved packet changed")
    receipts = read(packet/"logs/session_records/q1_teacher_recovery_completed_20261108.json")
    result = read(saved_identity(packet, receipts["result"]))
    forks = read(saved_identity(packet, receipts["fork_result"]))
    require(result["status"] == "COMPLETED" and result["source_integrity"] == "PASSED"
        and result["cost_settlement"]["state"] == "SETTLED"
        and len(forks["cases"]) == 16 and len(prior["pairs"]) == 8,
        "Require the original completed16 forks, not new physical replay")
    require(not output.exists() and not archive.exists(), "Preserve prior outputs; use a fresh analysis root")
    pairs, rows = [], []
    for index in range(8):
        cases = [next(r for r in forks["cases"] if (r["job_index"], r["arm"]) == (index, arm)) for arm in ARMS]
        old = next(p for p in prior["pairs"] if p["job_index"] == index)
        arrays = []
        for case in cases:
            with np.load(saved_identity(packet, case["trace"]), allow_pickle=False) as data:
                arrays.append({k:data[k] for k in data.files})
        n = matched_support(cases, arrays, old["common_valid_steps"])
        series = [velocity_series(a["body_actual"][:n+1], a["body_reference"][:n+1], a["load"][:n+1]) for a in arrays]
        matched = [summarize(s, n) for s in series]
        prefixes = [dict(steps=count, duration_ms=count*20, available=n >= count,
            arms=[summarize(s, count) for s in series] if n >= count else None) for count in PREFIX_STEPS]
        pair = dict(job_index=index, clip=cases[0]["clip"], split=cases[0]["split"], anchor=cases[0]["anchor"],
            common_valid_steps=n, original_valid_steps=[c["valid_prefix_steps"] for c in cases],
            matched=matched, prefixes=prefixes)
        pairs.append(pair)
        for arm, s in zip(ARMS, series):
            for i in range(n):
                rows.append(dict(job_index=index, clip=pair["clip"], split=pair["split"], anchor=pair["anchor"],
                    arm=arm, interval_end_frame=pair["anchor"]+i+1, interval_end_ms=(i+1)*20,
                    **{k:float(v[i]) if k != "mode" else int(v[i]) for k,v in s.items()}))
    require(before == {str(p.relative_to(packet)):identity(p) for p in paths}, "Input bytes changed during read-only measurement")
    require(time.monotonic()-started <= 120., "Local numerical120s bound")
    output.mkdir(parents=True)
    with (output/"per_interval.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    matched_rows = [dict(job_index=p["job_index"], clip=p["clip"], split=p["split"], anchor=p["anchor"],
        arm=arm, **{k:v for k,v in a.items() if k != "load_modes"}) for p in pairs for arm,a in zip(ARMS, p["matched"])]
    with (output/"matched_velocity.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(matched_rows[0])); writer.writeheader(); writer.writerows(matched_rows)
    draw(output, pairs)
    value = dict(stage="SAVED_FORK_VELOCITY_REVIEW", status="PASSED", pairs=pairs,
        prefix_cohorts=[dict(steps=n,pairs=sum(p["common_valid_steps"] >= n for p in pairs)) for n in PREFIX_STEPS],
        interval_rows=len(rows), source_files=before, original_analysis=dict(path=str(prior_summary), **identity(prior_summary)),
        plan=dict(path=str(plan_path), **identity(plan_path)), code_identity=identity(Path(__file__)),
        elapsed_seconds=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == "darwin" else 1024),
        velocity_kind="Backward finite-interval translation proxy from saved world positions; not simulator instantaneous velocity",
        root_body_index=1, static_world_body_excluded=0, moving_bodies=26,
        angular_velocity_analyzed=False, reference_preprocessing_causality="offline_unverified",
        hardware_end_to_end_latency_ms=None, new_policy_calls=0, new_physical_steps=0, new_optimizer_updates=0,
        new_rendered_frames=0, gpu_charged_seconds=0, original_shared_training_resumed=False,
        natural_teleoperation_qualified=False, user_accepted_checkpoint=None)
    require(value["elapsed_seconds"] <= 120. and value["peak_rss_bytes"] <= 512*(1 << 20), "Local time or512MiB bound")
    (output/"summary.json").write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    lines = ["# Q1 Saved-Fork Velocity Review", "", "Backward20ms mean translation velocity errors from preserved world positions.", "",
        "This is a new read-only measurement on16 existing forks, not new physics, training or hardware latency. Eight anchors on four clips are dependent.", "",
        "All comparisons use exactly the same nonterminal prefix. A longer requested prefix is missing, not padded or truncated.", "",
        "World velocity error decomposes into root error, pelvis-relative error and a signed cross term. A root fraction cannot be interpreted by dropping the cross term.", "",
        "Loads are last-substep>=5N labels; they are not full-contact impulses or planned reference contacts. Interval velocities are not instantaneous simulator body velocities. No angular velocity result is claimed.", "",
        "![Velocity six-panel](velocity_six_panel.png)", "", "[PDF](velocity_six_panel.pdf) | [Summary](summary.json) | [Matched CSV](matched_velocity.csv) | [Intervals](per_interval.csv)", "",
        "The original saved packet is retained separately and recorded by exact per-file identities in summary.json. No existing video was regenerated."]
    (output/"README.md").write_text("\n".join(lines)+"\n")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as package:
        for p in sorted(output.iterdir()): package.write(p, p.name)
    with zipfile.ZipFile(archive) as package:
        require(package.testzip() is None and len(package.namelist()) == len(set(package.namelist())), "Archive CRC or duplicate failure")
        members = [dict(path=n,bytes=len(package.read(n)),sha256=hashlib.sha256(package.read(n)).hexdigest()) for n in package.namelist()]
        require(all(package.read(p.name) == p.read_bytes() for p in output.iterdir()), "Archive bytes differ")
    require(before == {str(p.relative_to(packet)):identity(p) for p in paths}, "Input bytes changed during rendering")
    return dict(status=value["status"], pairs=len(pairs), interval_rows=len(rows), prefix_cohorts=value["prefix_cohorts"],
        numerical_and_plot_seconds=value["elapsed_seconds"], peak_rss_bytes=value["peak_rss_bytes"],
        archive=dict(path=str(archive), **identity(archive)), members=members, original_inputs_unchanged=len(before),
        new_policy_calls=0,new_physical_steps=0,new_optimizer_updates=0,gpu_charged_seconds=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("packet", "prior-summary", "plan", "output", "archive"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.packet, args.prior_summary, args.plan, args.output, args.archive), indent=2))
