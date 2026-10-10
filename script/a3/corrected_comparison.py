"""Validate and compare E04 outputs against immutable official/R05 baselines."""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.corrected_trial import read
from script.a3.fullchain_support import DATA, ROOT
from script.a3.official_results import aggregate
from script.a3.regression_diagnostic import PROTOCOL_FIELDS


OUTPUT = ROOT / "data/experiments/a3_corrected_20261010_E04"
LOGS = ROOT / "logs/a3_corrected_20261010_E04"


def main():
    verification = read(OUTPUT / "verification.json")
    if read(OUTPUT / "evaluation_state.json")["status"] != "complete":
        raise ValueError("The two evaluations have not completed")
    directories = dict(official=DATA / "official_baseline",
                       R05_step2=ROOT / "data/experiments/a3_regression_20261010_E01/step002",
                       R07_step1=OUTPUT / "step001", R07_step2=OUTPUT / "step002")
    summaries = {label: read(directory / "explicit_summary.json") for label, directory in directories.items()}
    base = summaries["official"]
    expected_shas = dict(official=verification["source_sha256"],
        R05_step2=next(r["sha256"] for r in read(
            ROOT / "data/experiments/a3_regression_20261010_E01/plan.json")["checkpoints"] if r["step"] == 2),
        R07_step1=verification["checkpoints"]["1"]["sha256"],
        R07_step2=verification["checkpoints"]["2"]["sha256"])
    artifacts = []
    for label, summary in summaries.items():
        for field in PROTOCOL_FIELDS:
            if summary["protocol"][field] != base["protocol"][field]:
                raise ValueError(f"Different evaluation protocol: {label}/{field}")
        if summary["protocol"]["checkpoint_sha256"] != expected_shas[label]:
            raise ValueError("Wrong evaluated checkpoint")
        if len(summary["motions"]) != 20 or not all(m["full_replay_completed"] for m in summary["motions"]):
            raise ValueError("Incomplete motion coverage")
        directory = directories[label]
        manifest = read(directory / "explicit_manifest.json")
        for motion in summary["motions"]:
            stem = Path(motion["motion"]).stem
            for suffix, expected in (
                ("metrics.json", manifest["runs"][stem]["metrics_sha256"]),
                ("timeseries.json", motion["timeseries_sha256"]),
            ):
                path = directory / f"{stem}.{suffix}"
                if sha256(path) != expected:
                    raise ValueError("Published evaluation artifact changed")
                if label.startswith("R07"):
                    artifacts.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                                          sha256=expected))
    split = read(DATA / "split.json")
    groups = {role: {r["name"] for r in split[role]} for role in ("train", "heldout")}
    rows, common = [], {label: [] for label in directories}
    for label, summary in summaries.items():
        motions = summary["motions"]
        rows.append(dict(label=label, all=aggregate(motions),
                         groups={role: aggregate([m for m in motions if m["motion"] in names])
                                 for role, names in groups.items()}))
    for motion in base["motions"]:
        name = motion["motion"]
        traces, horizon = {}, motion["expected_full_policy_steps"]
        for label, directory in directories.items():
            metric = read(directory / f"{Path(name).stem}.metrics.json")["motions"][0]
            if metric["num_policy_steps"] != motion["expected_full_policy_steps"]:
                raise ValueError("Paired rollout duration changed")
            horizon = min(horizon, metric["fall_tick"] if metric["fall"] else metric["num_policy_steps"])
            traces[label] = read(directory / f"{Path(name).stem}.timeseries.json")["motions"][0]
        for label, trace in traces.items():
            delta = np.asarray(trace["q_state_29"][:horizon]) - np.asarray(trace["reference_q_29"][:horizon])
            body_delta = np.linalg.norm(np.asarray(trace["sim_body_pos_anchor"][:horizon]) -
                                       np.asarray(trace["ref_body_pos_anchor"][:horizon]), axis=-1)
            names = trace["body_names"]
            indices = dict(wrists=[names.index(n) for n in ("left_wrist_yaw_Link", "right_wrist_yaw_Link")],
                           legs=[i for i, n in enumerate(names) if any(t in n for t in ("hip_", "knee_", "ankle_"))])
            item = dict(motion=name, policy_steps=horizon,
                        joint_rmse_rad=float(np.sqrt(np.mean(delta ** 2))) if horizon else None,
                        root_pos_error_m_mean=float(np.mean(trace["root_pos_error_m"][:horizon])) if horizon else None)
            for group, selected in indices.items():
                item[group + "_anchor_local_mean_m"] = float(body_delta[:, selected].mean()) if horizon else None
            common[label].append(item)
    for row in rows:
        items = common[row["label"]]
        count = sum(item["policy_steps"] for item in items)
        row["common_prefix"] = dict(policy_steps=count)
        for metric in ("joint_rmse_rad", "root_pos_error_m_mean", "wrists_anchor_local_mean_m", "legs_anchor_local_mean_m"):
            power = 2 if metric == "joint_rmse_rad" else 1
            total = sum(item["policy_steps"] * (item[metric] or 0) ** power for item in items)
            row["common_prefix"][metric] = (total / count) ** (1 / power) if count else None
    pairs = []
    for motion in base["motions"]:
        name = motion["motion"]
        pairs.append(dict(motion=name, group="train" if name in groups["train"] else "heldout",
            results={label: next(m for m in summary["motions"] if m["motion"] == name)
                     for label, summary in summaries.items()}))
    health, rejected = [], []
    for path in LOGS.rglob("health.jsonl"):
        for ordinal, line in enumerate(path.read_text().splitlines(), 1):
            try:
                health.append(json.loads(line))
            except ValueError:
                rejected.append(dict(path=str(path), line=ordinal))
    result = dict(rows=rows, pairs=pairs, common_prefix_by_motion=common,
        verification=str(OUTPUT / "verification.json"), verification_sha256=sha256(OUTPUT / "verification.json"),
        checkpoints=verification["checkpoints"], split_sha256=sha256(DATA / "split.json"), artifacts=artifacts,
        code=[dict(path=str(path.relative_to(ROOT)), sha256=sha256(path)) for path in (
            ROOT / "script/a3/corrected_trial.py", ROOT / "script/a3/corrected_comparison.py",
            ROOT / "script/a3/verified_finetune.py")],
        health=dict(samples=len(health), rejected=rejected, boot_ids=sorted({h["boot_id"] for h in health}),
                    cpu_max_c=max((h["temperatures_c"].get("x86_pkg_temp", 0) for h in health), default=None),
                    gpu_max_c=max((g["temperature.gpu"] for h in health for g in h["gpus"]), default=None),
                    ram_available_min_bytes=min((h["memory_bytes"]["MemAvailable"] for h in health), default=None)),
        scope="SINGLE_SEED_ENGINEERING_FIX_BUNDLE_DIAGNOSTIC_NOT_SINGLE_FACTOR_ATTRIBUTION",
        heldout_scope="WITHHELD_FROM_FINETUNING_NOW_USED_FOR_DIAGNOSIS_NOT_FRESH_FINAL_TEST",
        backup_status="LOCAL_ONLY", generated_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(OUTPUT / "comparison.json", result)
    atomic_json(ROOT / "data/manifests/a3_corrected_20261010_E04.json", result)
    print(json.dumps(rows, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
