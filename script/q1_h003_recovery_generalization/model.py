"""Append a small proprioceptive phase-and-leg policy to the frozen recovery ONNX."""

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def template(source, dest, *, late=False):
    import torch
    from torch import nn
    from onnx import compose

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = nn.Parameter(torch.ones(2, 3))
            pattern = ([-0.35, 0, 0, 0.0, 1.0, 0] if late else [-0.5, 0, 0, 1.0, -0.5, 0]) * 2 + [
                0.0
            ] * 10
            self.register_buffer("leg", torch.tensor(pattern))

        def forward(self, history, state, base_target, base_state):
            up = -history[:, -1, 49]
            change = up + history[:, -5, 49]
            features = torch.stack(
                (torch.ones_like(up), 1.0 - up, torch.clamp(5.0 * change, -1.0, 1.0)), 1
            )
            if late:
                # Signed pitch and average pitch rate from the causal history.
                features = torch.stack(
                    (
                        torch.ones_like(up),
                        history[:, -1, 47],
                        torch.clamp(2.0 * history[:, -4:, 45].mean(1), -1.0, 1.0),
                    ),
                    1,
                )
            output = torch.tanh(features @ self.weights.t())
            tick = state[:, 3]
            rise = (state[:, 1] == 2.0) & (base_state[:, 1] == 2.0)
            window = (
                rise.float()
                * torch.clamp((tick - 8.0) / 12.0, 0.0, 1.0)
                * torch.clamp((100.0 - tick) / 20.0, 0.0, 1.0)
            )
            if late:
                window = (
                    rise.float()
                    * torch.clamp((tick - 64.0) / 8.0, 0.0, 1.0)
                    * torch.clamp((136.0 - tick) / 12.0, 0.0, 1.0)
                )
            phase = (
                base_state[:, 3:4] + ((0.075 if late else 0.45) * window * output[:, 0])[:, None]
            )
            posture = (
                base_target + ((0.15 if late else 0.16) * window * output[:, 1])[:, None] * self.leg
            )
            return posture, torch.cat((base_state[:, :3], phase, base_state[:, 4:]), 1)

    scratch = dest.with_suffix(".adapter.onnx")
    module = Adapter().eval()
    torch.onnx.export(
        module,
        (torch.zeros(1, 10, 74), torch.zeros(1, 26), torch.zeros(1, 22), torch.zeros(1, 26)),
        str(scratch),
        input_names=["history", "state", "base_target", "base_state"],
        output_names=["target_out", "state_out"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=False,
    )
    base = onnx.load(source)
    adapter = compose.add_prefix(onnx.load(scratch), "pace_")
    # Use the same public input tensors for both subgraphs.
    mapping = {
        "pace_history": "rec_obs_hist",
        "pace_state": "rec_state",
        "pace_base_target": "original_target",
        "pace_base_state": "original_state",
    }
    for node in base.graph.node:
        for field in (node.input, node.output):
            for i, k in enumerate(field):
                if k in ("joint_target", "next_state"):
                    field[i] = {"joint_target": "original_target", "next_state": "original_state"}[
                        k
                    ]
    for node in adapter.graph.node:
        for i, k in enumerate(node.input):
            node.input[i] = mapping.get(k, k)
        for i, k in enumerate(node.output):
            node.output[i] = {
                "pace_target_out": "joint_target",
                "pace_state_out": "next_state",
            }.get(k, k)
    base.graph.node.extend(adapter.graph.node)
    base.graph.initializer.extend(adapter.graph.initializer)
    onnx.checker.check_model(base)
    onnx.save(base, dest)


def save_weights(source, weights, dest):
    model = onnx.load(source)
    w = next(x for x in model.graph.initializer if x.name == "pace_weights")
    w.CopyFrom(numpy_helper.from_array(np.asarray(weights, np.float32).reshape(2, 3), w.name))
    onnx.checker.check_model(model)
    onnx.save(model, dest)
