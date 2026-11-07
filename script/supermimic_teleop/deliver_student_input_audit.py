"""Plot the completed saved-input audit without loading models or sample arrays."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def identity(path, content):
    return dict(path=str(path), bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


def summarize(audit):
    result = []
    for origin in (0, 1):
        for split in ("train", "validation"):
            rows = [r for r in audit["source_rows"] if r["origin"] == origin and r["split"] == split]
            if not rows or any(r["samples"] <= 0 for r in rows):
                raise ValueError("Missing or empty source metrics")
            weights = [r["samples"]*10*r["metrics"]["valid_history_fraction"] for r in rows]
            neighbors = [r for r in audit["nearest_neighbor_proxy"]["rows"]
                         if r["origin"] == origin and r["split"] == split]
            if {r["source_clip_id"] for r in rows} != {r["source_clip_id"] for r in neighbors}:
                raise ValueError("Nearest-neighbor proxy has different source coverage")
            row = dict(origin=origin, split=split, samples=sum(r["samples"] for r in rows),
                first_200ms_samples=sum(r["first_200ms_samples"] for r in rows),
                first_second_samples=sum(r["first_second_samples"] for r in rows),
                maximum_abs_z=max(r["metrics"]["abs_z_max"] for r in rows))
            for key in ("clipped_coordinate_fraction", "history_rows_with_clipping_fraction",
                        "outside_original_train_range_fraction"):
                row[key] = sum(w*r["metrics"][key] for w, r in zip(weights, rows))/sum(weights)
            row.update(source_mean_neighbor_distance=sum(r["input_distance_median"] for r in neighbors)/len(neighbors),
                       source_mean_neighbor_label_mse=sum(r["nearest_label_mse_mean"] for r in neighbors)/len(neighbors))
            result.append(row)
    return result


def deliver(source, completion, output):
    before = {p: p.read_bytes() for p in (source, completion)}
    audit, completed = [json.loads(before[p]) for p in (source, completion)]
    registered = completed["input_audit"]
    if (audit.get("stage") != "SAVED_CAUSAL_INPUT_COVERAGE_AUDIT" or audit.get("status") != "PASSED"
            or audit.get("source_integrity") != "PASSED" or completed.get("status") != "PASSED"
            or completed.get("process_exited") is not True
            or registered["bytes"] != len(before[source])
            or registered["sha256"] != hashlib.sha256(before[source]).hexdigest()
            or any(audit[k] != 0 for k in ("new_model_inference_calls", "new_physical_transitions",
                                           "actual_optimizer_updates"))
            or audit.get("raw_human_or_test_payload_read") is not False
            or audit.get("normalization_refitted") is not False
            or any(not c["checks"][s]["state_and_label_trace_match"]
                   or not c["checks"][s]["saved_flags_clock_match"]
                   for c in audit["collection_checks"] for s in ("train", "validation"))):
        raise ValueError("Require the original completed, source-checked, CPU-only audit")
    if not output.is_absolute() or output.exists():
        raise FileExistsError("Use a fresh absolute diagnostic output root")
    aggregates = summarize(audit)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output.mkdir(parents=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    coverage = audit["source_coverage"]
    x = np.arange(2)
    for offset, key, label, color in ((-.18, "source_groups", "Source groups", "#167c6b"),
                                      (.18, "clips", "Clips", "#b94d4e")):
        values = [100*coverage[s]["collected_"+key]/coverage[s]["available_"+key]
                  for s in ("train", "validation")]
        bars = axes[0, 0].bar(x+offset, values, .32, label=label, color=color)
        axes[0, 0].bar_label(bars, labels=[str(coverage[s]["collected_"+key])+"/"+
            str(coverage[s]["available_"+key]) for s in ("train", "validation")], fontsize=8)
    axes[0, 0].set(xticks=x, xticklabels=["Train", "Validation"], ylim=(0, 115),
                   title="Actual collected source coverage", ylabel="Percent of owned corpus")
    axes[0, 0].legend(fontsize=8)
    labels = ["R1 train", "R1 val", "Mixed train", "Mixed val"]
    x = np.arange(4)
    for offset, key, label, color in ((-.18, "first_200ms_samples", "First 200 ms", "#b94d4e"),
                                      (.18, "first_second_samples", "First second", "#167c6b")):
        bars = axes[0, 1].bar(x+offset, [100*r[key]/r["samples"] for r in aggregates], .32,
                              label=label, color=color)
        axes[0, 1].bar_label(bars, labels=[str(r[key]) for r in aggregates], fontsize=8)
    axes[0, 1].set(xticks=x, xticklabels=labels, title="Startup samples; counts above bars",
                   ylabel="Percent of saved samples")
    axes[0, 1].legend(fontsize=8)
    bars = axes[0, 2].bar(x, [100*r["clipped_coordinate_fraction"] for r in aggregates], color="#4e789b")
    axes[0, 2].bar_label(bars, fmt="%.4f", fontsize=8)
    axes[0, 2].set(xticks=x, xticklabels=labels, title="Original normalization: coordinate clipping",
                   ylabel="Percent of valid geometry coordinates", ylim=(0, .055))
    validation = [[r for r in audit["source_rows"] if r["origin"] == origin and r["split"] == "validation"]
                  for origin in (0, 1)]
    x = np.arange(len(validation[0]))
    for origin, offset, color in ((0, -.18, "#4e789b"), (1, .18, "#b94d4e")):
        axes[1, 0].bar(x+offset, [100*r["metrics"]["history_rows_with_clipping_fraction"]
                                 for r in validation[origin]], .32,
                       label="R1 states" if origin == 0 else "Mixed states", color=color)
    axes[1, 0].set(xticks=x, xticklabels=[r["group_id"] for r in validation[0]],
                   title="Validation clipping concentrated by source", ylabel="Percent of valid history slots")
    axes[1, 0].legend(fontsize=8)
    for ax, key, title, ylabel in ((axes[1, 1], "input_distance_median", "Finite nearest-input distance proxy", "RMS-weighted distance"),
                                  (axes[1, 2], "nearest_label_mse_mean", "Finite nearest-label mismatch proxy", "Mean squared action difference")):
        groups = [[r[key] for r in audit["nearest_neighbor_proxy"]["rows"]
                   if r["origin"] == a["origin"] and r["split"] == a["split"]] for a in aggregates]
        ax.boxplot(groups, tick_labels=labels, showmeans=True)
        ax.set(title=title, ylabel=ylabel)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.2)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Q1 saved-input diagnostics | no new inference, physics or training | proxies are not policy scores", fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(output/("input_audit_six_panel."+suffix), dpi=160)
    plt.close(fig)
    columns = ("origin", "split", "source_clip_id", "group_id", "relative", "samples", "first_200ms_samples",
               "first_second_samples", "clipped_coordinate_fraction", "history_rows_with_clipping_fraction",
               "outside_original_train_range_fraction", "abs_z_max", "historical_action_rms")
    with (output/"source_metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in audit["source_rows"]:
            saved = row["historical_action_residual"]
            record = {key:row[key] for key in columns if key in row}
            record.update({key:row["metrics"][key] for key in columns if key in row["metrics"]})
            record["historical_action_rms"] = saved[0]["groups"]["all"]["rms"] if saved else None
            writer.writerow(record)
    if any(p.read_bytes() != content for p, content in before.items()):
        raise ValueError("Saved audit records changed during delivery")
    summary = dict(stage="SAVED_INPUT_AUDIT_DIAGNOSTICS", status="PASSED", aggregates=aggregates,
        source_coverage=coverage, sources=[identity(p, content) for p, content in before.items()],
        source_bytes_unchanged=True, new_model_inference_calls=0, new_physical_transitions=0,
        actual_optimizer_updates=0, coverage_is_proven_failure_cause=False,
        nearest_neighbor_proxy_is_a_policy_score=False, hardware_latency_ms=None,
        natural_teleoperation_qualified=False, user_accepted_checkpoint=None)
    (output/"input_coverage_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source", "completion", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(deliver(args.source, args.completion, args.output)))
