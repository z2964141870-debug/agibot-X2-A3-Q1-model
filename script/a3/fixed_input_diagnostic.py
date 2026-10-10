"""Compare saved policies on identical recorded inputs, using CPU only."""

import json

import numpy as np
import torch

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.export_check import A3FastExport
from script.a3.reference_contract import load_sim
from script.a3.regression_diagnostic import OUTPUT, read
from script.a3.fullchain_support import ROOT, DATA


def main():
    torch.set_num_threads(1)
    inputs = DATA / "official_baseline/001_walk_front_slow.inputs.npz"
    with np.load(inputs, allow_pickle=False) as payload:
        if payload["input_order"].item() != "command_multi_future_then_ori6d_multi_future_then_proprioception":
            raise ValueError("Wrong input ABI")
        observations, captured = payload["observations"].copy(), payload["actions"].copy()
    if observations.shape != (100, 1570) or captured.shape != (100, 29):
        raise ValueError("Expected 100 actual policy inputs")
    if not np.isfinite(observations).all() or not np.isfinite(captured).all():
        raise ValueError("Nonfinite policy capture")
    plan = read(OUTPUT / "plan.json")
    checkpoints = [("official", ROOT / "data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt")]
    from pathlib import Path
    checkpoints += [(str(row["step"]), Path(row["checkpoint"])) for row in plan["checkpoints"]]
    sim = load_sim()
    obs = torch.from_numpy(observations)
    command = obs[:, :580].reshape(-1, 10, 58)
    orientation = obs[:, 580:640].reshape(-1, 10, 6)
    encoder_input = torch.cat([command, orientation], dim=-1).reshape(-1, 640)
    rows, arrays = [], {}
    with torch.no_grad():
        for label, checkpoint in checkpoints:
            policy = sim.A3Policy(checkpoint, torch.device("cpu"), encoder="a3_fast")
            action = A3FastExport(policy).eval()(obs).numpy()
            latent = policy.encoder(encoder_input).reshape(-1, 2, 32)
            tokens = policy.fsq(latent).reshape(-1, 64).numpy()
            if not np.isfinite(action).all() or not np.isfinite(tokens).all():
                raise ValueError("Nonfinite fixed-input inference")
            if label == "official":
                if not np.allclose(action, captured, atol=1e-4, rtol=1e-3):
                    raise ValueError("CPU adapter disagrees with actual simulator inputs")
                baseline, base_tokens = action.copy(), tokens.copy()
                base_decoder = policy.decoder
            delta = action - baseline
            decoder_only = policy.decoder(torch.cat([torch.from_numpy(base_tokens), obs[:, 640:]], dim=-1)).numpy()
            encoder_only = base_decoder(torch.cat([torch.from_numpy(tokens), obs[:, 640:]], dim=-1)).numpy()
            rows.append(dict(label=label, action_rmse_raw=float(np.sqrt(np.mean(delta ** 2))),
                             action_max_abs_change_raw=float(np.abs(delta).max()),
                             fsq_token_elements_changed_fraction=float(np.mean(tokens != base_tokens)),
                             current_decoder_official_tokens_rmse_raw=float(np.sqrt(np.mean((decoder_only - baseline) ** 2))),
                             official_decoder_current_tokens_rmse_raw=float(np.sqrt(np.mean((encoder_only - baseline) ** 2)))))
            arrays[label + "_actions"] = action
            arrays[label + "_tokens"] = tokens
            del policy
    if rows[1]["action_max_abs_change_raw"] != 0 or rows[1]["fsq_token_elements_changed_fraction"] != 0:
        raise ValueError("Zero-update inference differs from official")
    np.savez_compressed(OUTPUT / "fixed_input_outputs.npz", **arrays)
    result = dict(input_source=str(inputs), input_sha256=sha256(inputs), samples=100,
                  rows=rows, device="cpu", action_units="RAW_POLICY_OUTPUT_NOT_JOINT_RADIANS",
                  scope="FIRST_2_SECONDS_OF_ONE_OFFICIAL_WALK_TRACE_NOT_CLOSED_LOOP_OR_CAUSAL_ATTRIBUTION",
                  hybrid_scope="SWAPPED_COMPONENTS_ONLY_AT_INFERENCE_NOT_FREEZE_TRAINING_OR_POLICY_ACCEPTANCE")
    atomic_json(OUTPUT / "fixed_input_diagnostic.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
