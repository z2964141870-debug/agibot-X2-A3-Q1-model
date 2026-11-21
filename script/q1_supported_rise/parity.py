"""Check deployed ONNX outputs and recurrent state against the exporting modules."""

import json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from export import FiniteRoll, SupportedRise
from recorded_rise import RecordedRise


def check(model, path):
    model = FiniteRoll(model).eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )
    rng = np.random.default_rng(88301)
    worst = np.zeros(2)
    steps = 0
    with torch.no_grad():
        for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
            state = np.zeros((1, 26), np.float32)
            exported_state = state.copy()
            hist = np.zeros((1, 10, 74), np.float32)
            for t in range(435):
                frame = rng.normal(0, 0.028, 74).astype(np.float32)
                frame[47:50] = 0.0
                frame[47 + axis] = -sign
                if t % 13 == 7:
                    unit = rng.normal(size=3)
                    frame[47:50] = unit / np.linalg.norm(unit)
                frame[-2:] = [0.17, 1]
                hist[:, :-1] = hist[:, 1:].copy()
                hist[:, -1] = frame
                raw, new = model(torch.from_numpy(hist), torch.from_numpy(state))
                target, exported_state = session.run(
                    None, {"rec_obs_hist": hist, "rec_state": exported_state}
                )
                state = new.numpy()
                diff = [
                    np.max(abs(raw.numpy() - target)),
                    np.max(abs(state - exported_state)),
                ]
                worst = np.maximum(worst, diff)
                steps += 1
    assert np.isfinite(worst).all() and max(worst) < 2e-5, (path, worst)
    return dict(steps=steps, max_target_rad=worst[0], max_state=worst[1])


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    specs = [
        ("fast", "fast", True, 1.0, 0.0),
        ("brake", "fast", True, 1.0, 13.0),
        ("fit01", "fit01", False, 1.0, 0.0),
        ("clear", "clear", True, 1.0, 0.0),
    ]
    results = {}
    for name, fit, supine, stretch, brake in specs:
        with np.load(root / fit / "fit.npz") as f:
            model = SupportedRise(f, supine, stretch, brake).eval()
        results[name] = check(model, root / "axis_safe" / (name + ".onnx"))
    results["paced"] = check(
        RecordedRise(Path.home() / "projects/Q1_deploy/real_logs", True).eval(),
        root / "axis_safe/paced.onnx",
    )
    if (root / "transfer/expert.onnx").exists():
        from bridge import Bridge

        with (
            np.load(root / "toward/fit.npz") as a,
            np.load(root / "transfer/fit.npz") as b,
        ):
            first = SupportedRise(a, False, hold_terminal=True).eval()
            second = SupportedRise(b, False, hold_terminal=True).eval()
        results["toward"] = check(first, root / "toward/expert.onnx")
        results["transfer"] = check(
            Bridge(first, second).eval(), root / "transfer/expert.onnx"
        )
    (root / "axis_safe/parity.json").write_text(json.dumps(results, indent=2))
