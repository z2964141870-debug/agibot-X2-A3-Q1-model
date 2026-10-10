"""CPU capability-drift probe on a fixed, SHA-checked official input capture."""

import argparse
from pathlib import Path

import numpy as np
import torch

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.corrected_trial import OFFICIAL, read
from script.a3.evaluation_common import checkpoint_metadata
from script.a3.export_check import A3FastExport
from script.a3.fullchain_support import DATA, ROOT
from script.a3.reference_contract import load_sim


def run(output):
    torch.set_num_threads(1)
    source = DATA / "official_baseline/001_walk_front_slow.inputs.npz"
    registered_sha = read(ROOT / "data/experiments/a3_regression_20261010_E01/fixed_input_diagnostic.json")["input_sha256"]
    if sha256(source) != registered_sha:
        raise ValueError("Fixed official input capture changed")
    with np.load(source, allow_pickle=False) as payload:
        if payload["input_order"].item() != "command_multi_future_then_ori6d_multi_future_then_proprioception":
            raise ValueError("Wrong input ABI")
        observations, captured = payload["observations"].copy(), payload["actions"].copy()
    if observations.shape != (100, 1570) or captured.shape != (100, 29):
        raise ValueError("Incomplete fixed-input capture")
    if not np.isfinite(observations).all() or not np.isfinite(captured).all():
        raise ValueError("Nonfinite fixed input")
    paths = {"official": OFFICIAL,
             "R07_epochs5": Path(read(ROOT / "data/experiments/a3_corrected_20261010_E04/verification.json")["checkpoints"]["2"]["checkpoint"])}
    for label, directory in (("R08_epochs1", "a3_search_20261010_E05"),
                             ("R09_epochs1_lr01", "a3_search_20261010_E06")):
        verification = ROOT / "data/experiments" / directory / "verification.json"
        if verification.exists():
            record = read(verification)["checkpoints"]["2"]
            path = Path(record["checkpoint"])
            if checkpoint_metadata(path)["sha256"] != record["sha256"]:
                raise ValueError("Verified model changed")
            paths[label] = path
    sim = load_sim()
    obs = torch.from_numpy(observations)
    rows = []
    with torch.no_grad():
        for label, path in paths.items():
            policy = sim.A3Policy(path, torch.device("cpu"), encoder="a3_fast")
            actions = A3FastExport(policy).eval()(obs).numpy()
            payload = torch.load(path, map_location="cpu", weights_only=False)
            std = np.clip(payload["policy_state_dict"]["std"].numpy(), 0.001, 0.5)
            if not np.isfinite(actions).all() or not np.isfinite(std).all():
                raise ValueError("Nonfinite fixed-input policy output")
            if label == "official":
                if not np.allclose(actions, captured, atol=1e-4, rtol=1e-3):
                    raise ValueError("CPU adapter no longer agrees with the official capture")
                base, base_std = actions.copy(), std.copy()
            delta = actions - base
            kl = (np.log(std / base_std) +
                  (base_std ** 2 + delta ** 2) / (2 * std ** 2) - 0.5).sum(axis=-1)
            rows.append(dict(label=label, checkpoint_sha256=checkpoint_metadata(path)["sha256"],
                action_rmse_raw=float(np.sqrt(np.mean(delta ** 2))),
                action_max_abs_change_raw=float(np.max(np.abs(delta))),
                gaussian_kl_official_to_current_mean=float(kl.mean()), std_mean=float(std.mean())))
            del policy, payload
    result = dict(rows=rows, samples=100, input_source=str(source), input_sha256=registered_sha,
                  device="cpu", action_units="RAW_POLICY_OUTPUT_NOT_JOINT_RADIANS",
                  scope="FIXED_FIRST_TWO_SECONDS_OF_OFFICIAL_WALK_NOT_CLOSED_LOOP_OR_TRAINING_KL")
    atomic_json(output / "fixed_input_drift.json", result)
    print(result, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("E05", "E06"), required=True)
    args = parser.parse_args()
    run(ROOT / "data/experiments" / f"a3_search_20261010_{args.experiment}")
