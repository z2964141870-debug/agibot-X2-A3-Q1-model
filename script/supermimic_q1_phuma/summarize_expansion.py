#!/usr/bin/env python3
"""Reconcile completed expansion batches and independently sum saved Q1 time axes."""

import argparse
import collections
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from acquire_source import file_id
from scan_corpus import resource_check


def summarize(selection_path, batch_paths, output, expected_oom):
    if output.exists():
        raise FileExistsError(output)
    if resource_check() != expected_oom:
        raise RuntimeError("Resource/OOM state changed before reconciliation")
    selection_id = file_id(selection_path)
    selection = json.loads(selection_path.read_text())
    base = json.loads(Path(selection["expansion"]["base_selection"]["path"]).read_text())
    if file_id(selection["expansion"]["base_selection"]["path"]) != selection["expansion"]["base_selection"]:
        raise ValueError("Original selection changed")
    previous_groups = {row["source_group"] for row in base["selected"]}
    groups, content, names = set(), set(), set()
    candidates, accepted, batches = [], [], []
    rejections = collections.Counter()
    model_identity = None
    for path in batch_paths:
        done_path = path / "completion.json"
        done = json.loads(done_path.read_text())
        for key in ("selection", "scan_check", "retarget_manifest", "reference_check"):
            if file_id(done[key]["path"]) != done[key]:
                raise ValueError(f"Batch {key} changed")
        batch = json.loads((path / "selection.json").read_text())
        if batch["batch"]["full_selection"] != selection_id:
            raise ValueError("Batch came from a different full selection")
        offset, count = batch["batch"]["offset"], batch["batch"]["count"]
        if batch["selected"] != selection["selected"][offset:offset + count]:
            raise ValueError("Batch rows differ from the fixed full selection")
        manifest = json.loads(Path(done["retarget_manifest"]["path"]).read_text())
        checked = json.loads(Path(done["reference_check"]["path"]).read_text())
        if (checked["input_manifest"] != done["retarget_manifest"]
                or manifest["selection"] != done["selection"]
                or len(manifest["motions"]) != count or len(checked["clips"]) != count):
            raise ValueError("Incomplete or mismatched reference check")
        if model_identity is None:
            model_identity = manifest["q1_assets"]
        if model_identity != manifest["q1_assets"]:
            raise ValueError("Batches used different Q1 models/control")
        by_name = {row["clip"]: row for row in checked["clips"]}
        if len(by_name) != count or [row["phuma_source"] for row in manifest["motions"]] != batch["selected"]:
            raise ValueError("Reference records do not uniquely cover the selected slice")
        batch_accepted = []
        for source, row in zip(batch["selected"], manifest["motions"]):
            if (source["source_group"] in groups or source["sha256"] in content
                    or source["source_group"] in previous_groups or row["clip"] in names):
                raise ValueError("Duplicate group/content or reused original candidate")
            groups.add(source["source_group"])
            content.add(source["sha256"])
            names.add(row["clip"])
            check = by_name[row["clip"]]
            if check["split"] != source["split"] or check["source_group"] != source["source_group"]:
                raise ValueError("Reference split or source group changed")
            candidates.append(source)
            rejections.update(key for key, value in check.get("gates", {}).items() if not value)
            if not check["qualified_for_training_pilot"]:
                continue
            if (len(check["gates"]) != 19 or not all(check["gates"].values())
                    or not row["qualified_for_training_pilot"] or check["reference"] != row["output"]
                    or file_id(row["output"]["path"]) != row["output"]):
                raise ValueError("Accepted reference identity or qualification differs")
            with np.load(row["output"]["path"], allow_pickle=False) as saved:
                times = saved["time_s"]
                frames = len(times)
                duration = (frames - 1) / 50.
                if (frames != check["frames"] or float(saved["fps"]) != 50.
                        or saved["qpos"].shape != (frames, 29)
                        or not np.allclose(times, np.arange(frames) / 50., rtol=0, atol=1e-10)
                        or abs(duration - check["duration_s"]) > 1e-10):
                    raise ValueError("Saved reference frames/time axis differs from the check")
            entry = {"clip": row["clip"], "split": source["split"], "bucket": source["bucket"],
                     "source_group": source["source_group"], "source_sha256": source["sha256"],
                     "reference": row["output"], "frames": frames, "duration_s": duration,
                     "horizontal_displacement_m": check["reference_horizontal_displacement_m"]}
            accepted.append(entry)
            batch_accepted.append(entry)
        for split in ("train", "development", "test"):
            seconds = sum(row["duration_s"] for row in batch_accepted if row["split"] == split)
            if abs(seconds - done["qualified_duration_s"][split]) > 1e-8:
                raise ValueError("Batch duration sum did not reconcile")
        if len(batch_accepted) != done["qualified_count"] or count != done["candidate_count"]:
            raise ValueError("Batch candidate/qualified counts did not reconcile")
        batches.append({"completion": file_id(done_path), "candidate_count": count,
                        "qualified_count": len(batch_accepted), "conversion_errors": done["conversion_errors"],
                        "elapsed_s": done["elapsed_s"], "verification_only": done["verification_only"]})
    summary = {
        "status": "PHUMA_P04_COMPLETED_BATCHES_INDEPENDENTLY_RECONCILED",
        "captured_utc": datetime.now(timezone.utc).isoformat(), "full_selection": selection_id,
        "selection_proof": selection["expansion"]["proof"], "batches": batches,
        "candidate_count": len(candidates), "candidate_unique_groups": len(groups),
        "candidate_unique_source_hashes": len(content),
        "candidate_counts": dict(collections.Counter(x["split"] + "/" + x["bucket"] for x in candidates)),
        "qualified_count": len(accepted),
        "qualified_counts": dict(collections.Counter(x["split"] + "/" + x["bucket"] for x in accepted)),
        "qualified_duration_s": {s: sum(x["duration_s"] for x in accepted if x["split"] == s)
                                 for s in ("train", "development", "test")},
        "qualified_locomotion_duration_s": {s: sum(x["duration_s"] for x in accepted
                                                if x["split"] == s and x["bucket"] == "locomotion")
                                            for s in ("train", "development", "test")},
        "qualified_frames": sum(x["frames"] for x in accepted), "qualified_references": accepted,
        "rejection_gate_counts": dict(rejections), "source": file_id(__file__),
        "oom_kill": resource_check(), "new_ik_conversions": 0,
        "new_physics_steps": 0, "new_optimizer_updates": 0, "storage_status": "LOCAL_ONLY",
        "limits": "Only completed batches count. Durations recomputed from saved frames at 50Hz; no semantic/subject deduplication, dynamic tracking or new policy success is claimed.",
    }
    if summary["oom_kill"] != expected_oom:
        raise RuntimeError("OOM state changed during reconciliation")
    with output.open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("qualified_references", "selection_proof")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--batches", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    args = parser.parse_args()
    summarize(args.selection.resolve(), [p.resolve() for p in args.batches], args.output.resolve(), args.expected_oom_kill)
