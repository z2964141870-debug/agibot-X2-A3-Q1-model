"""Export the pinned sim2sim A3-fast policy and verify actual-input parity."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from script.a3.evaluation_common import checkpoint_metadata
from script.a3.fullchain_support import ROOT, digest, write_json
from script.a3.reference_contract import load_sim


class A3FastExport(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs_dict):
        command = obs_dict[:, :580].reshape(-1, 10, 58)
        orientation = obs_dict[:, 580:640].reshape(-1, 10, 6)
        proprioception = obs_dict[:, 640:]
        encoder_input = torch.cat([command, orientation], dim=-1).reshape(-1, 640)
        latent = self.policy.encoder(encoder_input).reshape(-1, 2, 32)
        tokens = self.policy.fsq(latent).reshape(-1, 64)
        return self.policy.decoder(torch.cat([tokens, proprioception], dim=-1))


def run(checkpoint, inputs, output):
    if not output.resolve().is_relative_to(ROOT / "data"):
        raise ValueError("Export artifacts must remain inside project data")
    metadata = checkpoint_metadata(checkpoint)
    torch.set_num_threads(1)
    sim = load_sim()
    policy = sim.A3Policy(checkpoint, torch.device("cpu"), encoder="a3_fast")
    wrapper = A3FastExport(policy).eval()
    with np.load(inputs, allow_pickle=False) as payload:
        if payload["input_order"].item() != "command_multi_future_then_ori6d_multi_future_then_proprioception":
            raise ValueError("Input capture uses a different wire ABI")
        observations, expected = payload["observations"].copy(), payload["actions"].copy()
    if observations.shape != (100, 1570) or expected.shape != (100, 29):
        raise ValueError("Need 100 actual inputs/actions with the published A3-fast shape")
    if not np.isfinite(observations).all() or not np.isfinite(expected).all():
        raise ValueError("Nonfinite captured inputs/actions")
    output.mkdir(parents=True, exist_ok=False)
    model_path = output / "a3_fast.onnx"
    def atanh_symbolic(graph, value):
        # Opset13 has no Atanh; the pinned FSQ applies this to bounded constants.
        one = graph.op("Constant", value_t=torch.tensor(1., dtype=torch.float32))
        half = graph.op("Constant", value_t=torch.tensor(.5, dtype=torch.float32))
        ratio = graph.op("Div", graph.op("Add", one, value), graph.op("Sub", one, value))
        return graph.op("Mul", half, graph.op("Log", ratio))
    torch.onnx.register_custom_op_symbolic("aten::atanh", atanh_symbolic, 13)
    with torch.no_grad():
        recomputed = wrapper(torch.from_numpy(observations)).numpy()
        if not np.allclose(recomputed, expected, atol=1e-4, rtol=1e-3):
            raise ValueError("Export input layout disagrees with captured simulator actions")
        torch.onnx.export(wrapper, (torch.from_numpy(observations[:1]),), model_path,
                          input_names=["obs_dict"], output_names=["action"],
                          opset_version=13, dynamo=False)
    onnx.checker.check_model(onnx.load(model_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape != [1, 1570] or session.get_outputs()[0].shape != [1, 29]:
        raise ValueError("Exported ONNX shape does not match published A3-fast ABI")
    actual = np.concatenate([session.run(["action"], {"obs_dict": obs[None]})[0] for obs in observations])
    if not np.isfinite(actual).all():
        raise ValueError("Nonfinite ONNX actions")
    agreement = bool(np.allclose(actual, expected, atol=1e-4, rtol=1e-3))
    report = {"checkpoint": str(checkpoint), "checkpoint_sha256": metadata["sha256"],
              "inputs": str(inputs), "input_sha256": digest(inputs), "sample_count": 100,
              "input_shape": [1, 1570], "output_shape": [1, 29], "atol": 1e-4, "rtol": 1e-3,
              "max_abs_error": float(np.abs(actual - expected).max()), "allclose": agreement,
              "pt_adapter_vs_simulator_max_abs_error": float(np.abs(recomputed - expected).max()),
              "onnx_bytes": model_path.stat().st_size, "onnx_sha256": digest(model_path),
              "implementation": "PINNED_MUJOCO_A3Policy_ADAPTER_NOT_ISAAC_EXPORTER_VERIFICATION",
              "export_compatibility": "ATANH_OPSET13_SYMBOLIC_HALF_LOG_ONE_PLUS_X_OVER_ONE_MINUS_X",
              "checkpoint_origin": metadata.get("source", "local_experiment_not_official_pretrained"),
              "backup_status": "LOCAL_ONLY", "hardware_verified": False}
    np.savez_compressed(output / "parity.npz", expected=expected, actual=actual)
    write_json(output / "parity.json", report)
    print(json.dumps(report, indent=2))
    if not agreement:
        raise ValueError("PT/ONNX parity failed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.checkpoint.resolve(), args.inputs.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
