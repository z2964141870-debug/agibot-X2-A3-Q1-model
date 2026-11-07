"""Plot only the saved, audited pure-student label collection."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def identity(path):
    raw = path.read_bytes()
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def deliver(root):
    source = root / "logs/session_records/q1_f0_onpolicy_completed_20261108.json"
    resources = root / "logs/session_records/q1_f0_onpolicy_resource_final_20261108.json"
    initial = {str(p): identity(p) for p in (source, resources)}
    value = json.loads(source.read_text())
    rows = value["saved_diagnostic"]["rows"]
    assert value["status"] == "PASSED" and value["source_inputs_unchanged"]
    assert value["result"]["source_integrity"] == "PASSED"
    assert value["result"]["cost_settlement"]["state"] == "SETTLED"
    assert value["guard"]["cleanup"]["worker_reaped"] is True
    assert value["guard"]["cleanup"]["worker_return_code"] == 0
    assert len(rows) == 9 and sum(r["label_records"] for r in rows) == 625
    assert sum(r["physical_steps"] for r in rows) == 1259
    assert value["saved_diagnostic"]["next_state_terminal_pairing_verified"]
    assert value["collection"]["teacher_behavior_fraction"] == 0
    assert value["collection"]["new_optimizer_updates"] == 0

    output = root / "data/analysis/Q1_F0_ONPOLICY_LABELS_FIGURES_20261108"
    archive = root / "data/deliveries/Q1_F0_ONPOLICY_LABELS_EVIDENCE_20261108.zip"
    assert not archive.exists()
    assert not output.exists() or not any(output.iterdir())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    labels = [r["clip"].split("__")[-1] for r in rows]
    x = np.arange(len(rows))
    colors = ["#167d8d" if r["split"] == "train" else "#b44962" for r in rows]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.suptitle("Pure-student collection: 1,259 physical steps / 625 teacher labels", fontsize=17)
    ax = axes.flat[0]
    ax.bar(x, [r["requested_steps"] for r in rows], color="#e2e6e9", label="Requested cap")
    ax.bar(x, [r["physical_steps"] for r in rows], color=colors, label="Actual steps")
    ax.set_title("Physical prefixes (8 terminated; 1 reached its cap)")
    ax.set_ylabel("Physics steps")
    ax.legend(fontsize=9)
    ax = axes.flat[1]
    ax.bar(x, [r["label_records"] for r in rows], color=colors)
    ax.set_title("Paired labels: 406 train / 219 validation")
    ax.set_ylabel("Labels")
    ax = axes.flat[2]
    ax.bar(x, [r["canonical_action_delta_rms"] for r in rows], color=colors)
    ax.set_title("Teacher-student action difference, entire prefix")
    ax.set_ylabel("RMS (canonical action units)")
    ax = axes.flat[3]
    ax.bar(x, [r["last25_delta_rms"] for r in rows], color=colors)
    ax.set_title("Action difference over final 25 physical steps")
    ax.set_ylabel("RMS (canonical action units)")
    ax = axes.flat[4]
    ax.plot(x, [r["per_label_delta_rms_p50"] for r in rows], "o-", color="#167d8d", label="p50")
    ax.plot(x, [r["per_label_delta_rms_p95"] for r in rows], "s-", color="#b44962", label="p95")
    ax.set_title("Distribution of per-label action differences")
    ax.set_ylabel("RMS (canonical action units)")
    ax.legend(fontsize=9)
    ax = axes.flat[5]
    ax.bar(x - .18, [r["last25_physical_steps_label_count"] for r in rows], .36,
           color="#167d8d", label="Labels in final 25 steps")
    ax.bar(x + .18, [r["terminal_label_records"] for r in rows], .36,
           color="#b44962", label="Labels on terminal action")
    ax.set_title("Boundary coverage (teacher queried every 2 frames)")
    ax.set_ylabel("Labels")
    ax.legend(fontsize=9)
    for ax in axes.flat:
        ax.set_xticks(x, labels, rotation=45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
    fig.text(.5, .015, "Teal: training sources. Red: validation sources. No parameter updates; teacher recovery remains unverified.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .94))
    for suffix in ("png", "pdf"):
        fig.savefig(output / ("collection_six_panel." + suffix), dpi=160)
    plt.close(fig)
    with (output / "per_source.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for item in (source, resources):
        shutil.copyfile(item, output / item.name)
    (output / "README.md").write_text(
        "# Pure-student collection evidence\n\n"
        "2026-11-08. Nine predetermined source groups; 1,259 physical steps and 625 paired labels "
        "(406 train / 219 validation). Eight prefixes terminated; one reached its requested cap. "
        "This is collection integrity, not a comparison with the earlier seven evaluation cases.\n\n"
        "[Six-panel PNG](collection_six_panel.png) / [PDF](collection_six_panel.pdf) / "
        "[per-source CSV](per_source.csv).\n\n"
        "[Original completion and pairing audit](q1_f0_onpolicy_completed_20261108.json) / "
        "[Original resource record](q1_f0_onpolicy_resource_final_20261108.json).\n\n"
        "The student alone drove physics. Privileged H10 teacher actions were labels only. "
        "Next states and terminal boundaries match the saved traces. No new optimizer updates "
        "or rendered videos. Label differences do not prove teacher recovery or natural teleoperation. "
        "Hardware end-to-end latency remains unmeasured.\n")
    assert all(identity(Path(p)) == meta for p, meta in initial.items())
    files = [dict(relative=p.name, **identity(p)) for p in sorted(output.iterdir())]
    manifest = dict(files=files, source_inputs_unchanged=True, new_inference_calls=0,
                    new_physical_steps=0, new_optimizer_updates=0)
    (output / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as stream:
        for p in sorted(output.iterdir()):
            stream.write(p, p.name)
    with zipfile.ZipFile(archive) as stream:
        assert stream.testzip() is None
        assert set(stream.namelist()) == {r["relative"] for r in files} | {"bundle_manifest.json"}
        for row in files:
            raw = stream.read(row["relative"])
            assert len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"]
    return dict(archive=str(archive), **identity(archive), members=len(files) + 1,
                new_training_updates=0, source_inputs_unchanged=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(deliver(args.root)))
