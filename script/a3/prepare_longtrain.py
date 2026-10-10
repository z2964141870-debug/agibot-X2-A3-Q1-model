"""Read-only training inputs audit; never launches or resumes a training job."""

import json
from datetime import datetime, timezone
from pathlib import Path
import shutil

import joblib
import numpy as np
import torch

from script.a3.autoresume import VENDOR_COMMIT
from script.a3.checkpoint_store import atomic_json, sha256, validate_payload
from script.a3.corrected_trial import OFFICIAL
from script.a3.evaluation_common import checkpoint_metadata
from script.a3.fullchain_support import DATA, ROOT, VENDOR
from script.a3.host_health import command, sample


def evaluate_readiness(contract, checks):
    required = ("official_bundle", "official_cpu_reload", "motion_split", "motionlib", "storage", "vendor")
    blockers = [name for name in required if not checks.get(name, {}).get("passed")]
    if not checks.get("runtime_likelihood", {}).get("passed"):
        blockers.append("initial_runtime_likelihood_unresolved")
    if not checks.get("candidate_policy", {}).get("passed"):
        blockers.append("no_accepted_finetuned_policy")
    if not checks.get("full_causal_input", {}).get("passed"):
        blockers.append("full_causal_orientation_not_verified")
    if contract.get("candidate_variable") is None or contract.get("actor_update_scope") is None:
        blockers.append("candidate_objective_and_parameter_scope_not_fixed")
    if contract.get("long_training_updates") is None:
        blockers.append("long_training_budget_not_fixed")
    return dict(infrastructure_ready=all(checks.get(n, {}).get("passed") for n in required),
                long_training_ready=not blockers, blockers=blockers,
                final_acceptance_pending=[] if checks.get("fresh_validation", {}).get("passed") else ["fresh_final_validation_missing"],
                launch_performed=False)


def main():
    started = datetime.now(timezone.utc)
    directory = ROOT / "data/experiments/a3_longtrain_readiness_20261010" / started.strftime("check_%Y%m%dT%H%M%SZ")
    directory.mkdir(parents=True, exist_ok=False)
    contract_path = ROOT / "script/a3/longtrain_contract.json"
    contract = json.loads(contract_path.read_text())
    checks = {}
    metadata = checkpoint_metadata(OFFICIAL)
    manifest = json.loads((VENDOR / "a3_hf_manifest.json").read_text())
    files = []
    for filename in ("model_step_200000.pt", "config.yaml", "model_config.yaml", "meta.yaml", "LICENSE"):
        path = OFFICIAL.parent / filename
        remote = f"035_step200000/{filename}"
        matches = [row for row in manifest["files"] if row["remote"] == remote]
        # The release manifest records the license at repository root.
        if filename == "LICENSE" and not matches:
            matches = [row for row in manifest["files"] if row["remote"] == "LICENSE"]
        if not matches or len({(row["sha256"], row["size"]) for row in matches}) != 1:
            raise ValueError(f"Ambiguous bundle identity: {remote}")
        expected = matches[0]
        valid = path.is_file() and path.stat().st_size == expected["size"] and sha256(path) == expected["sha256"]
        files.append(dict(path=str(path), passed=valid, bytes=path.stat().st_size if path.exists() else None,
                          sha256=sha256(path) if path.exists() else None))
    checks["official_bundle"] = dict(passed=all(row["passed"] for row in files), files=files)
    payload = torch.load(OFFICIAL, map_location="cpu", weights_only=False)
    source_step = validate_payload(payload)
    checks["official_cpu_reload"] = dict(passed=source_step == 200000, source_step=source_step,
        sha256=metadata["sha256"], network_and_optimizer_finite=True, restore_source_counter_for_new_trial=False)
    del payload
    split = json.loads((DATA / "split.json").read_text())
    train, heldout = split["train"], split["heldout"]
    identities = [row["sha256"] for row in train + heldout]
    checks["motion_split"] = dict(passed=len(train) == 16 and len(heldout) == 4 and len(set(identities)) == 20
        and all(sha256(Path(row["path"])) == row["sha256"] for row in train + heldout),
        split_sha256=sha256(DATA / "split.json"), train_motions=len(train), diagnostic_validation_motions=len(heldout),
        fresh_final_test=False)
    motionlibs = []
    for part in ("train", "heldout"):
        rows = split[part]
        paths = sorted((DATA / "motionlib" / part).rglob("*.pkl"))
        if {p.stem for p in paths} != {Path(row["name"]).stem for row in rows}:
            raise ValueError("Motionlib differs from registered whole-motion split")
        for path in paths:
            stored = joblib.load(path)
            entry = stored[path.stem]
            valid = (set(stored) == {path.stem} and entry["fps"] == 30 and entry["dof"].shape[1] == 29
                     and all(np.isfinite(entry[key]).all() for key in ("root_trans_offset", "root_rot", "pose_aa", "dof")))
            motionlibs.append(dict(path=str(path), sha256=sha256(path), passed=bool(valid), split=part,
                                   frames=int(entry["dof"].shape[0])))
    checks["motionlib"] = dict(passed=len(motionlibs) == 20 and all(row["passed"] for row in motionlibs), files=motionlibs,
                              mocap_usage="TEST_ONLY_NOT_INCLUDED")
    free = shutil.disk_usage(ROOT).free
    checkpoint_bytes = OFFICIAL.stat().st_size
    short_saves = contract["short_validation_updates"] // contract["checkpoint_frequency"] + 1
    # Provision duplicate checkpoint transactions plus 10GiB diagnostic logs/exports headroom.
    short_required = checkpoint_bytes * short_saves * 2 + 10 * 1024**3
    checks["storage"] = dict(passed=free >= short_required, free_bytes=free,
        short_trial_checkpoint_count=short_saves, short_trial_reserved_bytes=short_required,
        long_training_storage_estimate_pending=contract["long_training_updates"] is None)
    version = command(["git", "-C", str(VENDOR), "rev-parse", "HEAD"])
    checks["vendor"] = dict(passed=version["returncode"] == 0 and version["stdout"] == VENDOR_COMMIT,
                             expected_commit=VENDOR_COMMIT, observed_commit=version["stdout"])
    deferred = ROOT / "data/experiments/a3_gradient_20261010_E10/state.json"
    checks["runtime_likelihood"] = dict(passed=False, state=json.loads(deferred.read_text()),
        note="E10 remains blocked; no runtime consistency receipt. Do not relaunch to bypass no-progress stop.")
    checks["candidate_policy"] = dict(passed=False, selected="OFFICIAL_UNCHANGED", note="R05-R09 have not passed effect acceptance")
    causal_path = ROOT / "data/manifests/a3_causal_inputs_verified_20261010_E18.json"
    if causal_path.exists():
        verified = json.loads(causal_path.read_text())
        checks["full_causal_input"] = dict(passed=verified.get("status") == "passed",
            receipt=str(causal_path), sha256=sha256(causal_path), scope="POLICY_REFERENCE_TOKENS_ONLY",
            initialization="MATCHED_OFFLINE_RESET_QVEL; CAUSAL_INITIALIZATION_NOT_VERIFIED")
    else:
        checks["full_causal_input"] = dict(passed=False, note="E18 independent actual-input verification not yet available")
    checks["fresh_validation"] = dict(passed=False, note="Four reused diagnostic motions are not a fresh final test")
    result = dict(utc=started.isoformat(), contract_sha256=sha256(contract_path), checks=checks,
        readiness=evaluate_readiness(contract, checks), initial_health=sample(),
        git_head=command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]), policy_updates=0, backup_status="LOCAL_ONLY")
    atomic_json(directory / "result.json", result)
    atomic_json(ROOT / "data/manifests/a3_longtrain_readiness_20261010.json", result)
    print(json.dumps(result["readiness"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
