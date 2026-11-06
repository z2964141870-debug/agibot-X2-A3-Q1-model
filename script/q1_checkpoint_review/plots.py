"""Six-panel reports from immutable scalar snapshots; preserve missing metrics."""

import ast
import csv
import json
import math
from pathlib import Path


TAGS = ("collect/mean_norm_return", "collect/failures", "collect/episodes_finished",
    "collect/mean_episode_length", "collect/mean_reward", "wm/pred_err_8step", "imag/survival_H")
TITLES = ("Episode normalized return", "Hard failures / finished episodes", "Finished episode length",
    "Step reward / source maximum", "World model eight-step error", "Imagined survival at H32")


def finite(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def pooled(rows, end, window=500):
    selected = [r for r in rows if end - window < r["iteration"] <= end]
    out = []
    for index in range(6):
        values = [r for r in selected if finite(r["panels"][index])]
        if index < 3:
            values = [r for r in values if finite(r.get("finished")) and r["finished"] > 0]
            weight = sum(r["finished"] for r in values)
            out.append(sum(r["panels"][index] * r["finished"] for r in values) / weight if weight else None)
        else:
            out.append(sum(r["panels"][index] for r in values) / len(values) if values else None)
    return out


def panel_rows(scalars, maximum):
    result = []
    for step, row in sorted(scalars.items()):
        finished, failures = row.get("collect/episodes_finished"), row.get("collect/failures")
        ratio = failures / finished if finite(finished) and finished > 0 and finite(failures) else None
        reward = row.get("collect/mean_reward")
        panels = [row.get(TAGS[0]), ratio, row.get(TAGS[3]),
            reward / maximum if finite(reward) else None, row.get(TAGS[5]), row.get(TAGS[6])]
        result.append(dict(iteration=step, finished=finished, panels=panels))
    return result


def draw(rows, output, label, warmup, titles=TITLES, episode_weights=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    csv_rows = []
    for row in rows:
        if episode_weights:
            smooth = pooled(rows, row["iteration"])
        else:
            window = [r for r in rows if row["iteration"] - 500 < r["iteration"] <= row["iteration"]]
            smooth = []
            for j in range(6):
                values = [r["panels"][j] for r in window if finite(r["panels"][j])]
                smooth.append(sum(values) / len(values) if values else None)
        csv_rows.append(dict(iteration=row["iteration"], **{
            f"{prefix}_{i}": value for prefix, values in (("raw", row["panels"]), ("mean500", smooth))
            for i, value in enumerate(values)}))
    x = [r["iteration"] for r in rows]
    missing = []
    for i, (axis, title) in enumerate(zip(axes.flat, titles)):
        raw = [r["panels"][i] if finite(r["panels"][i]) else np.nan for r in rows]
        mean = [r[f"mean500_{i}"] if finite(r[f"mean500_{i}"]) else np.nan for r in csv_rows]
        axis.plot(x, raw, color="#568eb2", alpha=.4, linewidth=.7)
        axis.plot(x, mean, color="#246d4f", linewidth=1.8)
        axis.axvline(warmup, color="#777777", linestyle=":", linewidth=.7)
        axis.set(title=title, xlabel="Completed training iterations")
        axis.ticklabel_format(axis="x", style="plain")
        axis.grid(alpha=.18)
        if not any(finite(r["panels"][i]) for r in rows):
            axis.text(.5, .5, "NOT RECORDED", ha="center", va="center", transform=axis.transAxes, color="#8b403a")
            missing.append(i)
    fig.suptitle(label + "\nThin: recorded windows; thick: trailing500 iterations; training signals only", fontsize=12)
    fig.savefig(output / "training.png", dpi=160)
    fig.savefig(output / "training.pdf")
    plt.close(fig)
    with (output / "metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    return dict(rows=len(rows), first_iteration=x[0], last_iteration=x[-1], missing_panels=missing,
        layout=[2, 3], smoothing_window_iterations=500,
        episode_smoothing="Pooled finished-episode weights for first3 panels" if episode_weights else "Available logged-point mean",
        policy_quality_evaluation=False)


def generate(output, plan, config, runtime):
    from common import identity, require, save
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    root = output / "curves"
    root.mkdir(exist_ok=False)
    constants = Path(plan["source"]) / "src/morphmimic/constants.py"
    maximum = next(ast.literal_eval(node.value) for node in ast.parse(constants.read_text()).body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "R_MAX" for t in node.targets))
    require(maximum > 0 and math.isfinite(maximum), "Invalid pinned reward maximum")
    windows = {}
    for robot in ("g1", "q1"):
        source = Path(plan["output"]) / "run" / robot
        snapshot = root / (robot + "_events")
        snapshot.mkdir()
        sources = []
        for path in sorted(source.glob("events.out.tfevents.*")):
            data = path.read_bytes()
            require(0 < len(data) <= 32 * 1024**2, "Scalar snapshot exceeds its bounded size")
            destination = snapshot / path.name
            destination.write_bytes(data)
            sources.append(dict(original_path=str(path), snapshot=identity(destination),
                original_may_append=True, prefix_bytes=len(data)))
        require(sources, "No training scalar events")
        reader = EventAccumulator(str(snapshot), size_guidance={"scalars": 0})
        reader.Reload()
        scalars = {}
        tags = reader.Tags()["scalars"]
        for tag in TAGS:
            if tag in tags:
                for event in reader.Scalars(tag):
                    scalars.setdefault(event.step, {})[tag] = event.value
        require(scalars, "No usable training metrics")
        rows = panel_rows(scalars, maximum)
        label = (f"Shared MorphMimic {robot.upper()} |1024 envs per robot (2048 total)|"
                 f"WM batch512; motion/KP batch1024|{rows[-1]['iteration']}/160000")
        receipt = draw(rows, root / robot, label, 2000)
        receipt.update(sources=sources, reward_maximum=maximum, reward_maximum_source=identity(constants), tags=tags,
            note="WM and imagined survival are training/model diagnostics, not measured gait quality")
        save(root / robot / "source_receipt.json", receipt)
        windows[robot] = [dict(end=end, points=sum(end-2000 < r["iteration"] <= end for r in rows),
            mean=pooled(rows, end, 2000)) for end in (16000, 32000, 48000, rows[-1]["iteration"])]
    metrics = Path(config["primebot_review"]).parents[1] / "train/run/metrics.jsonl"
    content = metrics.read_bytes()
    native = [json.loads(line) for line in content.splitlines() if line.strip()]
    require(native and native[-1]["iteration"] == 10000, "Need the completed original Primebot log")
    rows = [dict(iteration=r["iteration"], panels=[None, None, None, r.get("collection_reward"), None, None]) for r in native]
    native_titles = list(TITLES)
    native_titles[3] = "Collection reward (raw; maximum unrecorded)"
    receipt = draw(rows, root / "primebot_standard", "Primebot Q1|4096 envs;WM/policy batch2048|10000/250000 early stop", 0,
                   titles=native_titles)
    receipt.update(source=identity(metrics), metric_keys=sorted(set().union(*(r.keys() for r in native))),
        missing_note="Historical episode counts, K8 error and imagined survival were not recorded; no reconstruction",
        reward_note="Panel4 is raw collection reward; its maximum was not recorded and normalization is unavailable")
    save(root / "primebot_standard/source_receipt.json", receipt)
    diagnostic = [dict(iteration=r["iteration"], panels=[r.get(key) for key in
        ("collection_reward", "state_loss", "termination_loss", "actor_loss", "critic_loss", "global_env_steps")]) for r in native]
    receipt = draw(diagnostic, root / "primebot_diagnostics", "Primebot native recorded diagnostics|Not the standard6 metrics", 0,
        ("Collection reward (raw units)", "World model state loss", "Termination loss", "Actor loss", "Critic loss", "Real environment transitions"),
        episode_weights=False)
    receipt["source"] = identity(metrics)
    save(root / "primebot_diagnostics/source_receipt.json", receipt)
    require(identity(metrics)["sha256"] == __import__("hashlib").sha256(content).hexdigest(), "Completed Primebot log changed")
    save(root / "window_statistics.json", dict(panels=TITLES, window_iterations=2000, robots=windows,
        physical_quality_assessed=False))

