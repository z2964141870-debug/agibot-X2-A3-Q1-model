"""Create a saved-data performance comparison and an independently verified ZIP."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from script.x2_fgp_realtime.assets import file_digest


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--report", type=Path)
    a = p.parse_args()
    if a.output.exists() or a.zip.exists():
        p.error("Preserve previous deliveries")
    names = ("control", "sleep1", "sleep1_fk", "all_sleep1_fk")
    directories = ("control_analysis_v2", "sleep1_analysis", "sleep1_fk_analysis", "all_sleep1_fk_analysis")
    summaries = []
    for name, directory in zip(names, directories):
        s = json.loads((a.root/directory/"summary.json").read_text())
        if s["status"] != "JOINT_SAVED_ANALYSIS_PASSED" or s["hardware_commands"] or s["preview_span_ms"] != 180:
            raise ValueError("Invalid experiment completion or changed preview")
        c = json.loads((a.root/name/"performance_config.json").read_text())
        if c["variant"] != name:
            raise ValueError("Variant mismatch")
        summaries.append(s)
    if len({s["boot_id"] for s in summaries}) != 1:
        raise ValueError("Experiments crossed boots")
    before, after = [a.root/name for name in ("integrity_before.json", "integrity_after.json")]
    if before.read_bytes() != after.read_bytes():
        raise ValueError("Protected source/asset receipts differ")
    numerical = json.loads((a.root/"numerics_orin.json").read_text())
    if numerical["status"] != "POLICY_THREAD_EQUIVALENCE_PASSED":
        raise ValueError("Thread numerical comparison failed")
    frames = [[r["frame"] for r in map(json.loads, (a.root/name/"fgp.jsonl").read_text().splitlines())
               if r["output"]] for name in names]
    common = 0
    for group in zip(*frames):
        if len(set(group)) != 1:
            break
        common += 1
    if common == 0:
        raise ValueError("No identical processed prefix")
    difference = {}
    with np.load(a.root/names[0]/"fgp_outputs.npz", allow_pickle=False) as reference:
        for name in names[1:]:
            with np.load(a.root/name/"fgp_outputs.npz", allow_pickle=False) as candidate:
                difference[name] = {k:float(np.max(np.abs(candidate[k][:common]-reference[k][:common])))
                                    for k in ("pose", "translation")}
    a.output.mkdir(parents=True)
    rows = []
    for name, s in zip(names, summaries):
        rows.append(dict(variant=name, human_hz=s["fgp_output_wall_rate_hz"], outputs=s["fgp_outputs"],
            skipped=s["dropped_input_frames"], fgp_p50_ms=s["fgp_ms"]["p50"],
            fgp_p95_ms=s["fgp_ms"]["p95"], bridge_p50_ms=s["bridge_ms"]["p50"],
            rpc_ready_p95_ms=s["rpc_ready_ms"]["p95"], rpc_ready_p99_ms=s["rpc_ready_ms"]["p99"],
            rpc_over_20ms=s["rpc_ready_ms"]["over_budget"], native_ready=s["native_ready"],
            native_accepted=s["native_accepted"], age_lower_p50_ms=s["target_age_lower_ms"]["p50"],
            age_upper_p50_ms=s["target_age_upper_ms"]["p50"], age_upper_p95_ms=s["target_age_upper_ms"]["p95"]))
    with (a.output/"comparison.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for i, metric in enumerate(("p50", "p95", "p99")):
        axes[0, 0].bar(x+(i-1)*.24, [s["fgp_ms"][metric] for s in summaries], .24, label=metric)
    axes[0, 0].axhline(1000/30, color="firebrick", ls="--")
    axes[0, 0].set(title="FGP compute", ylabel="ms"); axes[0, 0].legend()
    axes[0, 1].bar(x, [r["human_hz"] for r in rows], color="#257e6b")
    axes[0, 1].axhline(30, color="firebrick", ls="--")
    axes[0, 1].set(title="Actual new human information", ylabel="Hz")
    axes[0, 2].bar(x-.16, [s["bridge_ms"]["p50"] for s in summaries], .32, label="FK p50")
    axes[0, 2].bar(x+.16, [s["human_rpc_ms"]["p95"] for s in summaries], .32, label="ACK p95")
    axes[0, 2].set(title="Serial conversion and communication", ylabel="ms"); axes[0, 2].legend()
    for i, metric in enumerate(("p95", "p99")):
        axes[1, 0].bar(x+(i-.5)*.32, [s["rpc_ready_ms"][metric] for s in summaries], .32, label=metric)
    axes[1, 0].axhline(20, color="firebrick", ls="--")
    axes[1, 0].set(title="Morph RPC, all ready requests", ylabel="ms"); axes[1, 0].legend()
    axes[1, 1].bar(x-.16, [r["age_lower_p50_ms"] for r in rows], .32, label="p50 lower")
    axes[1, 1].bar(x+.16, [r["age_upper_p95_ms"] for r in rows], .32, label="p95 upper")
    axes[1, 1].axhline(180, color="#555555", ls="--", label="preview span")
    axes[1, 1].set(title="Oldest reference age, includes preview", ylabel="ms"); axes[1, 1].legend()
    axes[1, 2].bar(x, [r["rpc_over_20ms"] for r in rows], color="#af453f")
    axes[1, 2].set(title="RPC deadline refusals", ylabel="Requests")
    for ax in axes.flat:
        ax.set_xticks(x, names, rotation=17, ha="right")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("X2 SoC1: same recorded IMU source + real feedback | SDK publishing 0", fontsize=13)
    fig.savefig(a.output/"six_panel.png", dpi=160)
    fig.savefig(a.output/"six_panel.pdf")
    plt.close(fig)
    receipt = dict(status="SAVED_PERFORMANCE_COMPARISON_PASSED", rows=rows,
        new_model_calls=0, hardware_commands=0, training_updates=0, preview_span_ms=180,
        causal_attribution="Sequential fixed-source configurations, not randomized repetitions; skipped frames differ",
        hardware_end_to_end_ms=None, protected_receipts_identical=True,
        identical_processed_prefix_outputs=common, prefix_max_absolute_difference=difference,
        full_sequence_numerical_equivalence=False,
        numerical_limit="Only common processed prefix; later skipped-frame histories differ")
    (a.output/"summary.json").write_text(json.dumps(receipt, indent=2)+"\n")
    members = {}
    for name in names:
        for path in sorted((a.root/name).rglob('*')):
            if path.is_file():
                members[str(path.relative_to(a.root))] = path
    for directory in directories:
        for path in sorted((a.root/directory).iterdir()):
            if path.is_file():
                members[str(path.relative_to(a.root))] = path
    for path in sorted(a.output.iterdir()):
        members["comparison/"+path.name] = path
    for path in sorted(a.root.glob("*.json")):
        members["checks/"+path.name] = path
    if a.report:
        members["README.md"] = a.report
    manifest = {name:dict(bytes=path.stat().st_size, sha256=file_digest(path)) for name,path in members.items()}
    a.zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.zip, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name, path in members.items():
            z.write(path, name)
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(a.zip) as z:
        if z.testzip():
            raise ValueError("ZIP CRC failure")
        for name, row in manifest.items():
            data = z.read(name)
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("ZIP member verification failed")
    result = dict(status="PERFORMANCE_EVIDENCE_ZIP_VERIFIED", members=len(manifest)+1,
        path=str(a.zip), bytes=a.zip.stat().st_size, sha256=file_digest(a.zip))
    (a.output/"delivery.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
