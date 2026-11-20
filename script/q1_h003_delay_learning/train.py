"""Supervised learning of tomorrow's nominal command from today's causal history (one control tick)."""

import json
import sys
import time
import numpy as np
import torch
from torch import nn
import onnxruntime as ort
from protocol import ROOT

torch.set_num_threads(3)
torch.manual_seed(71931)
rng = np.random.default_rng(71931)


def dataset(splits):
    X = []
    Y = []
    kinds = []
    linear = []
    lengths = {}
    for split in splits:
        for surface in ("normal", "lag", "rough"):
            for kind in ("static", "relay"):
                path = ROOT / "runs" / split / surface / "published" / kind
                receipt = json.loads((path / "receipt.json").read_text())
                files = sorted(path.glob("signals_*.npz"))
                assert len(files) == len(receipt["episodes"])
                n = 0
                for file in files:
                    with np.load(file) as data:
                        keep = data["seen"][:-1]
                        if not keep.any():
                            continue
                        base = data["base"]
                        # Discard the final sample of every episode before forming labels.
                        x = data["features"][:-1][keep]
                        y = np.diff(base, axis=0)[keep]
                        dx = (base - np.concatenate([base[:1], base[:-1]]))[:-1][keep]
                        X.append(x)
                        Y.append(y)
                        linear.append(dx)
                        kinds.extend([kind == "relay"] * len(y))
                        n += len(y)
                lengths[f"{split}/{surface}/{kind}"] = n
    return (
        torch.from_numpy(np.concatenate(X)),
        torch.from_numpy(np.concatenate(Y)),
        torch.tensor(kinds),
        torch.from_numpy(np.concatenate(linear)),
        lengths,
    )


class Lead(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.register_buffer("mean", x.mean(0))
        self.register_buffer("scale", x.std(0).clamp(min=0.025))
        self.net = nn.Sequential(
            nn.Linear(x.shape[1], 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 22),
        )

    def forward(self, x):
        return 0.5 * torch.tanh(self.net(((x - self.mean) / self.scale).clamp(-8, 8)))


def main():
    dest = ROOT / "trained"
    dest.mkdir(exist_ok=False)
    x, y, k, _, n = dataset(["collect0", "collect1"])
    vx, vy, vk, vlin, vn = dataset(["valid"])
    model = Lead(x)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    hist = []
    best = float("inf")
    # Match locomotion and recovery exposure despite different trajectory lengths.
    indices = [np.flatnonzero(k.numpy() == flag) for flag in (False, True)]
    for epoch in range(40):
        loss_sum = 0.0
        for _ in range(max(1, len(x) // 512)):
            ix = np.concatenate([rng.choice(s, 256, replace=True) for s in indices])
            pred = model(x[ix])
            err = (pred - y[ix]).square().mean(1)
            weight = 1 + 2 * torch.clamp(y[ix].abs().mean(1) / 0.1, 0, 1)
            loss = (weight * err).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.detach())
        with torch.no_grad():
            err = (model(vx) - vy).square().mean(1)
            a, b = (float(err[vk == flag].mean().sqrt()) for flag in (False, True))
        objective = (a + b) * 0.5
        if objective < best:
            best = objective
            chosen = epoch + 1
            torch.save(model.state_dict(), dest / "best.pt")
        hist.append(
            dict(
                epoch=epoch + 1,
                fit_weighted_mse=loss_sum / max(1, len(x) // 512),
                valid_static_rmse=a,
                valid_relay_rmse=b,
            )
        )
    model.load_state_dict(torch.load(dest / "best.pt", weights_only=True))
    model.eval()
    torch.onnx.export(
        model,
        (vx[:1],),
        str(dest / "predict.onnx"),
        input_names=["features"],
        output_names=["delta"],
        dynamic_axes={"features": {0: "batch"}, "delta": {0: "batch"}},
        dynamo=False,
        opset_version=17,
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(dest / "predict.onnx"), options, providers=["CPUExecutionProvider"]
    )
    with torch.no_grad():
        predictions = model(vx)
    difference = float(
        np.max(
            abs(session.run(None, {"features": vx[::29].numpy()})[0] - predictions[::29].numpy())
        )
    )
    assert difference < 1e-5
    metrics = {}
    for key, f in (
        ("repeat_last", torch.zeros_like(vy)),
        ("linear", vlin.clamp(-0.5, 0.5)),
        ("learned", predictions),
    ):
        e = (f - vy).square().mean(1)
        metrics[key] = [float(e[vk == flag].mean().sqrt()) for flag in (False, True)]
    result = dict(
        seed=71931,
        fit_rows=len(x),
        validation_rows=len(vx),
        fit_sources=n,
        validation_sources=vn,
        epoch=chosen,
        updates=40 * max(1, len(x) // 512),
        onnx_max_abs=difference,
        metrics=metrics,
        history=hist,
    )
    (dest / "fit.json").write_text(json.dumps(result, indent=2))
    print(metrics, chosen, len(x), len(vx), flush=True)


if __name__ == "__main__":
    main()
