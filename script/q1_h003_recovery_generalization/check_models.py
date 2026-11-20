"""Verify exported adapters preserve the public contract outside their rise windows."""

import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from run_suite import EXPERT


def session(path):
    cfg = ort.SessionOptions()
    cfg.intra_op_num_threads = cfg.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), cfg, providers=["CPUExecutionProvider"])


def run():
    original = session(EXPERT)
    results = {}
    for name, lo, hi, phase_cap, angle_cap, selection in (
        ("selected", 8, 100, 0.45, 0.16, "selection.json"),
        ("late_selected", 64, 136, 0.075, 0.15, "selection_late.json"),
    ):
        path = ROOT / "models" / (name + ".onnx")
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == json.loads((ROOT / selection).read_text())["sha256"]
        )
        s = session(path)
        assert [(x.name, x.shape) for x in original.get_inputs()] == [
            (x.name, x.shape) for x in s.get_inputs()
        ]
        rng = np.random.default_rng(73)
        exact = active = 0
        phase_max = target_max = 0.0
        for i in range(960):
            h = rng.normal(0, 0.3, (1, 10, 74)).astype(np.float32)
            h[:, :, -1] = 1
            h[:, :, -2] = 0.38
            state = np.zeros((1, 26), np.float32)
            mode = i // 160 % 3
            tick = i % 160
            state[0, :4] = (60, mode, 1, tick)
            feed = dict(rec_obs_hist=h, rec_state=state)
            a, b = original.run(["joint_target", "next_state"], feed)
            c, d = s.run(["joint_target", "next_state"], feed)
            assert np.array_equal(
                b[:, [0, 1, 2] + list(range(4, 26))], d[:, [0, 1, 2] + list(range(4, 26))]
            )
            if mode != 2 or b[0, 1] != 2 or not lo < tick < hi:
                assert np.array_equal(a, c) and np.array_equal(b, d)
                exact += 1
            else:
                active += 1
                phase_max = max(phase_max, float(abs(b[0, 3] - d[0, 3])))
                target_max = max(target_max, float(abs(a - c).max()))
        assert 0 < phase_max <= phase_cap + 1e-5 and 0 < target_max <= angle_cap + 1e-5
        results[name] = dict(
            exact_outside=exact,
            active=active,
            peak_phase_delta=phase_max,
            peak_target_delta_rad=target_max,
        )
    (ROOT / "contract.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    run()
