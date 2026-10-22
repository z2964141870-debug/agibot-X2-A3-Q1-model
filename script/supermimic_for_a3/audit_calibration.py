#!/usr/bin/env python3
"""Check a completed calibration's replay provenance, model states and validation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from calibrate_world_model import validation_metrics
from supermimic.a3_assets import A3Assets, identity
from supermimic.data.replay_buffer import WindowBatch
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg
from train import health


def require(condition, message):
    if not condition:
        raise ValueError(message)


def window_key(state, action, frame):
    return state.numpy().tobytes(), action.numpy().tobytes(), int(frame)


def replay_index(fields):
    result = {}
    for env, row in (fields["valid"] & ~fields["terminal"]).nonzero().tolist():
        key = window_key(fields["s"][env, row], fields["a_exec"][env, row], fields["frame"][env, row])
        result.setdefault(key, []).append((env, row))
    return result


def contains_window(fields, index, window, column, horizon):
    key = window_key(window["states"][0, column], window["a_exec"][0, column], window["frame"][0, column])
    for env, row in index.get(key, []):
        rows = (torch.arange(horizon + 1) + row) % fields["s"].shape[1]
        if not fields["valid"][env, rows].all() or fields["terminal"][env, rows[:-1]].any():
            continue
        if not (fields["ep_id"][env, rows] == fields["ep_id"][env, row]).all():
            continue
        if not torch.equal(window["states"][:, column], fields["s"][env, rows]):
            continue
        if not torch.equal(window["xi"][column], fields["xi"][env, row]):
            continue
        pairs = (("a_exec", "a_exec"), ("a_clean", "a"), ("reward", "r"),
                 ("done", "done"), ("frame", "frame"))
        if all(torch.equal(window[left][:, column], fields[right][env, rows[:-1]]) for left, right in pairs):
            return True
    return False


def compare_metrics(actual, recorded):
    require(actual.keys() == recorded.keys(), "Validation horizons changed")
    for horizon, predictions in actual.items():
        require(predictions.keys() == recorded[horizon].keys(), "Validation comparisons changed")
        for model, metrics in predictions.items():
            require(metrics.keys() == recorded[horizon][model].keys(), "Validation metric names changed")
            for key, value in metrics.items():
                require(np.isclose(value, recorded[horizon][model][key], rtol=5e-5, atol=1e-7),
                        f"Recomputed validation differs: {horizon}/{model}/{key}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    require(not args.output.exists(), "Audit output must be new")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    initial_health = health()
    receipt_path = args.calibration / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    require(receipt["status"] == "WORLD_MODEL_CALIBRATION_COMPLETED", "Calibration did not complete")
    admission_path = args.calibration / "admission.json"
    require(identity(admission_path) == receipt["admission"], "Admission identity changed")
    admission = json.loads(admission_path.read_text())
    require(identity(admission["entry"]["path"]) == admission["entry"], "Calibration entry changed")
    require(receipt["source_checkpoint"] == admission["source_checkpoint"], "Source checkpoint contracts differ")
    require(identity(receipt["source_checkpoint"]["path"]) == receipt["source_checkpoint"], "Source checkpoint changed")
    for artifact in receipt["artifacts"]:
        require(identity(artifact["path"]) == artifact, "Calibration artifact changed")
    expected_artifacts = {"fitting_replay.pt", "validation_replay.pt", "validation_windows.pt", "calibrated_world_model.pt"}
    require({Path(a["path"]).name for a in receipt["artifacts"]} == expected_artifacts,
            "Calibration artifact set differs")
    source = torch.load(receipt["source_checkpoint"]["path"], map_location="cpu", weights_only=False)
    final = torch.load(args.calibration / "calibrated_world_model.pt", map_location="cpu", weights_only=False)
    require(final["admission"] == admission and final["source_checkpoint"] == receipt["source_checkpoint"],
            "Fitted model provenance differs")
    require(final["wm_config"] == source["manifest"]["world_model"], "World-model configuration changed")
    require(receipt["policy_unchanged"] is True and receipt["policy_updates"] == receipt["critic_updates"] == 0,
            "Frozen-policy runtime checks failed")
    require("policy" not in final and "critic" not in final, "Calibration unexpectedly exports actor or critic")
    budget = admission["args"]
    expected_interactions = 2 * budget["num_envs"] * budget["collection_steps"]
    require(receipt["physics_env_steps"] == expected_interactions, "Interaction count differs")
    require(receipt["wm_updates"] == final["additional_wm_updates"] == budget["updates"], "Update budget differs")
    replay_fields, indexes, split_counts = {}, {}, {}
    for offset, split in enumerate(("fitting", "validation")):
        replay = torch.load(args.calibration / (split + "_replay.pt"), map_location="cpu", weights_only=False)
        require(replay["seed"] == budget["seed"] + offset, "Collection seed differs")
        require(replay["source_checkpoint"] == receipt["source_checkpoint"], "Replay policy source differs")
        require(replay["train_clips"] == admission["training_clips"], "Replay references differ")
        require(replay["E"] == budget["num_envs"] and replay["T"] == 2 * budget["collection_steps"] + 1,
                "Replay capacity differs")
        fields = replay["fields"]
        transition_mask = fields["valid"] & ~fields["terminal"]
        require(int(transition_mask.sum()) == expected_interactions // 2, "Replay transition count differs")
        require(int(fields["valid"].sum()) == replay["total_rows"], "Replay rows were overwritten or lost")
        require(all(torch.isfinite(v).all() for v in fields.values()), "Nonfinite replay tensor")
        clip_names = [Path(row["path"]).stem for row in replay["train_clips"]]
        counts = torch.bincount(fields["clip"][transition_mask], minlength=len(clip_names))
        require(len(counts) == len(clip_names) and (counts > 0).all(), "Missing or unexpected replay clip")
        split_counts[split] = {name: int(count) for name, count in zip(clip_names, counts)}
        require(split_counts[split] == receipt["clip_transition_counts"][split], "Clip counts differ")
        replay_fields[split] = fields
        indexes[split] = replay_index(fields)
    windows = torch.load(args.calibration / "validation_windows.pt", map_location="cpu", weights_only=False)
    require(set(windows) == {1, 8, 32}, "Fixed validation horizon set differs")
    window_counts = {}
    for horizon, window in windows.items():
        require(window["states"].shape == (horizon + 1, 256, 83), "Validation window shape differs")
        require(all(torch.isfinite(value).all() for value in window.values()), "Nonfinite validation window")
        for column in range(256):
            require(contains_window(replay_fields["validation"], indexes["validation"], window, column, horizon),
                    "A fixed validation window is absent from the validation replay")
            require(not contains_window(replay_fields["fitting"], indexes["fitting"], window, column, horizon),
                    "A fixed validation window is also present in the fitting replay")
        window_counts[str(horizon)] = 256
    changed = []
    require(source["wm"].keys() == final["wm"].keys(), "Model tensor keys differ")
    for name, value in final["wm"].items():
        require(value.shape == source["wm"][name].shape and torch.isfinite(value).all(), "Invalid model tensor")
        if not torch.equal(value, source["wm"][name]):
            changed.append(name)
    require(changed and changed == receipt["changed_wm_tensors"], "Changed model tensors differ")
    before_steps = {int(s["step"]) for s in source["wm_opt"]["state"].values()}
    final_steps = {int(s["step"]) for s in final["wm_opt"]["state"].values()}
    expected_steps = {n + budget["updates"] * admission["wm_trainer"]["batches_per_update"] for n in before_steps}
    require(final_steps == expected_steps and sorted(final_steps) == receipt["optimizer_steps"], "WM Adam steps differ")
    require(sorted(before_steps) == admission["initial_optimizer_steps"], "Initial WM Adam steps differ")
    require(all(torch.isfinite(v).all() for state in final["wm_opt"]["state"].values()
                for v in state.values() if torch.is_tensor(v)), "Nonfinite model optimizer")
    assets = A3Assets(budget["root"])
    require(assets.manifest() == admission["assets"], "Current assets differ")
    config = dict(source["manifest"]["world_model"])
    config["state_spec"] = StateSpecCfg(**config["state_spec"])
    wm = WorldModel(WorldModelCfg(**config), assets.heads(args.device), args.device)
    wm.enable_compile(admission["wm_trainer"]["compile"])
    gpu_windows = {h: WindowBatch(**{k: v.to(args.device) for k, v in w.items()}) for h, w in windows.items()}
    wm.load_state_dict(source["wm"], strict=True)
    baseline = validation_metrics(wm, gpu_windows)
    wm.load_state_dict(final["wm"], strict=True)
    calibrated = validation_metrics(wm, gpu_windows)
    compare_metrics(baseline, receipt["initial_validation"])
    compare_metrics(calibrated, receipt["final_validation"])
    decision = {
        "root_32step_improved_20pct": calibrated["32"]["learned"]["root_position_rmse_m"] <= .8 * baseline["32"]["learned"]["root_position_rmse_m"],
        "one_step_state_regression_below_10pct": calibrated["1"]["learned"]["normalized_state_mse"] <= 1.1 * baseline["1"]["learned"]["normalized_state_mse"],
        "joint_32step_regression_below_10pct": calibrated["32"]["learned"]["active_joint_rmse_rad"] <= 1.1 * baseline["32"]["learned"]["active_joint_rmse_rad"]}
    require(decision == receipt["decision"] and all(decision.values()) == receipt["passed_for_candidate"],
            "Candidate decision differs from recomputation")
    result = {"status": "WORLD_MODEL_CALIBRATION_INDEPENDENTLY_VERIFIED", "receipt": identity(receipt_path),
              "audit_entry": identity(__file__), "source_checkpoint": receipt["source_checkpoint"],
              "artifacts": receipt["artifacts"], "physics_env_steps": expected_interactions,
              "wm_updates": budget["updates"], "optimizer_steps": sorted(final_steps),
              "clip_transition_counts": split_counts, "verified_validation_windows": window_counts,
              "initial_validation": baseline, "final_validation": calibrated, "decision": decision,
              "passed_for_candidate": all(decision.values()), "initial_health": initial_health, "final_health": health(),
              "policy_freeze_evidence": "Hash-bound calibration entry and its tensor equality runtime check; no independently saved actor snapshot.",
              "storage_status": "LOCAL_ONLY", "not_a_policy_effect_result": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
