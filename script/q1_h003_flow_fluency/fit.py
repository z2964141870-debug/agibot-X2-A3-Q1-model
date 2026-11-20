"""Fit a causal joint-command lead using transport-matched within-episode targets."""

import json
import numpy as np
import torch
from torch import nn
import onnxruntime as ort
from design import D


def dataset(split):
    xs, ys, kinds = [], [], []
    for f in sorted((D / "runs" / split).glob("*/*/*/signals_*.npz")):
        with np.load(f) as d:
            base = d["base"]
            lag = d["delay"][:-2, None]
            lead = (base[1:-1] - base[:-2]) * np.minimum(lag, 1) + (
                base[2:] - base[1:-1]
            ) * np.maximum(lag - 1, 0)
            mask = d["seen"][:-2]
            xs.append(d["features"][:-2][mask])
            ys.append(lead[mask].astype(np.float32))
            kinds.extend([f.parent.name == "relay"] * int(sum(mask)))
    return (
        torch.from_numpy(np.concatenate(xs)),
        torch.from_numpy(np.concatenate(ys)),
        np.array(kinds),
    )


class Predictor(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.register_buffer("mean", x.mean(0))
        self.register_buffer("scale", x.std(0).clamp(min=0.025))
        self.layers = nn.Sequential(
            nn.Linear(x.shape[1], 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 22),
        )

    def forward(self, x):
        return 0.5 * torch.tanh(self.layers(((x - self.mean) / self.scale).clamp(-8, 8)))


def main():
    torch.set_num_threads(3)
    torch.manual_seed(6112199)
    rng = np.random.default_rng(6112199)
    x, y, k = dataset("fit")
    vx, vy, vk = dataset("val")
    model = Predictor(x)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    history = []
    best = np.inf
    for ep in range(40):
        losses = []
        for _ in range(max(1, len(x) // 512)):
            ix = np.r_[rng.choice(np.flatnonzero(k), 256), rng.choice(np.flatnonzero(~k), 256)]
            e = (model(x[ix]) - y[ix]).square().mean(1)
            loss = (e * (1 + 2 * (y[ix].abs().mean(1) / 0.1).clamp(0, 1))).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        with torch.no_grad():
            pred = model(vx)
            # Deployment falls back when confirmation is absent or its observed delay is zero.
            pred[(vx[:, -48] == 0) | (vx[:, -45] == 0)] = 0
            err = (pred - vy).square().mean(1)
            a, b = [float(err[vk == flag].mean().sqrt()) for flag in (False, True)]
        history.append([ep + 1, float(np.mean(losses)), a, b])
        if a + b < best:
            best = a + b
            picked = ep + 1
            torch.save(model.state_dict(), D / "age.pt")
    model.load_state_dict(torch.load(D / "age.pt", weights_only=True))
    model.eval()
    torch.onnx.export(
        model,
        (x[:1],),
        str(D / "age.onnx"),
        input_names=["features"],
        output_names=["delta"],
        dynamic_axes={"features": {0: "batch"}, "delta": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    cfg = ort.SessionOptions()
    cfg.intra_op_num_threads = cfg.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(D / "age.onnx"), cfg, providers=["CPUExecutionProvider"])
    v = vx[::23]
    diff = float(abs(sess.run(None, {"features": v.numpy()})[0] - model(v).detach().numpy()).max())
    assert diff < 1e-5
    (D / "fit.json").write_text(
        json.dumps(
            dict(
                train=len(x),
                validation=len(vx),
                history=history,
                epoch=picked,
                onnx_max_abs=diff,
                updates=40 * max(1, len(x) // 512),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
