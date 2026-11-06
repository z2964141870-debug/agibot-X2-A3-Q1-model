"""Summarize saved fixed-case comparisons; no model loading or new simulation."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

EXPECTED_ITERATIONS = (16000, 32000, 48000)


def metrics(case):
    result = dict(complete=case["complete"], valid_steps=case["valid_steps"], reference_steps=case["reference_steps"])
    if not case["complete"]:
        return result
    saved, diag = case["full_clip_metrics"], case["diagnostics"]
    result.update(joint_rmse_rad=saved["joint_rmse_rad"], body_rmse_m=saved["body_relative_rmse_m"],
        upper_rmse_rad=diag["upper_joint_rmse_rad"], root_xy_m=diag["root_xy_final_error_m"],
        tilt_rms_rad=diag["torso_tilt_rms_rad"], alternations=diag["alternating_lift_transitions"],
        sole_left_m=diag["sole_p95_m"][0], sole_right_m=diag["sole_p95_m"][1],
        sole_reference_error_m=statistics.mean(abs(a-b) for a, b in zip(diag["sole_p95_m"], diag["reference_sole_p95_m"])),
        slip_left_m_s=diag["loaded_slip_mean_m_s"][0], slip_right_m_s=diag["loaded_slip_mean_m_s"][1])
    return result


def compare_group(records):
    by_iteration = {record["iteration"]: {case["clip"]: metrics(case) for case in record["cases"]} for record in records}
    assert len(by_iteration) == len(records), "Duplicate saved iteration"
    names = set.intersection(*(set(cases) for cases in by_iteration.values())) if by_iteration else set()
    assert all(set(cases) == names for cases in by_iteration.values()), "Changed fixed case set"
    means = {}
    for iteration, cases in by_iteration.items():
        complete = [case for case in cases.values() if case["complete"]]
        value = dict(complete_cases=len(complete), total_cases=len(cases))
        if len(complete) == len(cases) and complete:
            for key in complete[0]:
                if key in ("complete", "valid_steps", "reference_steps"):
                    continue
                values = [case[key] for case in complete]
                value[key] = statistics.mean(values) if all(v is not None and math.isfinite(v) for v in values) else None
        means[iteration] = value
    changes = []
    for left, right in ((16000, 32000), (32000, 48000)):
        if left not in by_iteration or right not in by_iteration:
            continue
        for name in sorted(names):
            a, b = by_iteration[left][name], by_iteration[right][name]
            value = dict(clip=name, left=left, right=right, both_complete=a["complete"] and b["complete"],
                         complete_before=a["complete"], complete_after=b["complete"])
            if value["both_complete"]:
                value["absolute_change"] = {key: b[key]-a[key] for key in a
                    if key not in ("complete", "valid_steps", "reference_steps") and a[key] is not None and b[key] is not None}
                value["relative_change"] = {key: delta/a[key] for key, delta in value["absolute_change"].items() if a[key] != 0}
            changes.append(value)
    mean_changes = []
    for left, right in zip(EXPECTED_ITERATIONS, EXPECTED_ITERATIONS[1:]):
        if left not in means or right not in means:
            continue
        a, b = means[left], means[right]
        if a["complete_cases"] != a["total_cases"] or b["complete_cases"] != b["total_cases"]:
            continue
        absolute = {key: b[key] - a[key] for key in a
                    if key not in ("complete_cases", "total_cases") and a[key] is not None and b[key] is not None}
        mean_changes.append(dict(left=left, right=right, absolute_change=absolute,
            relative_change={key: delta/a[key] for key, delta in absolute.items() if a[key] != 0}))
    return dict(available_iterations=sorted(by_iteration),
        missing_iterations=sorted(set(EXPECTED_ITERATIONS) - set(by_iteration)),
        per_clip=by_iteration, full_fixed_set_means=means, changes=changes, mean_changes=mean_changes,
        failed_case_policy="No full-set means or full-motion differences when any required case failed")


def summarize(source, output):
    data = source.read_bytes()
    value = json.loads(data)
    groups = {name: compare_group(records) for name, records in value["groups"].items()}
    result = dict(groups=groups, source=dict(path=str(source), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()),
        optimizer_updates=0, physical_transitions=0, final_acceptance=False, user_accepted_checkpoint=None,
        interpretation="Three nominal training motions, not statistical convergence or natural-gait acceptance",
        missing_group_policy="Empty or partially evaluated groups remain explicit and cannot establish shared-model convergence")
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    rows = []
    for name, group in groups.items():
        for iteration, cases in group["per_clip"].items():
            rows.extend(dict(group=name, iteration=iteration, clip=clip, **case) for clip, case in cases.items())
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "per_case.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return result


def plot(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = (("joint_rmse_rad", "Joint RMSE (rad)"), ("upper_rmse_rad", "Upper RMSE (rad)"),
        ("slip_left_m_s", "Loaded slip (m/s); dashed=right"), ("sole_reference_error_m", "Sole p95 error against reference (m)"),
        ("root_xy_m", "Root final XY error (m)"), ("tilt_rms_rad", "Torso tilt RMS (rad)"))
    colors = ("#287b8e", "#a44d46", "#5b7b36")
    for name, group in result["groups"].items():
        if not group["per_clip"]:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        iterations = sorted(group["per_clip"])
        names = sorted({clip for cases in group["per_clip"].values() for clip in cases})
        for axis, (key, title) in zip(axes.flat, labels):
            for clip, color in zip(names, colors):
                series = [group["per_clip"][i][clip].get(key) for i in iterations]
                axis.plot(iterations, [v if v is not None else math.nan for v in series], "o-", color=color, label=clip)
                if key == "slip_left_m_s":
                    axis.plot(iterations, [group["per_clip"][i][clip].get("slip_right_m_s", math.nan) for i in iterations],
                              "o--", color=color, linewidth=1.)
            axis.set(title=title, xlabel="Saved checkpoint iteration", xticks=iterations)
            axis.grid(alpha=.2)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(name + " | Same fixed seed0 cases; full valid motions; missing values are not zero", fontsize=12)
        fig.savefig(output / (name.replace("/", "_") + ".png"), dpi=160)
        fig.savefig(output / (name.replace("/", "_") + ".pdf"))
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    result = summarize(args.source, args.output)
    if args.plot:
        plot(result, args.output)
