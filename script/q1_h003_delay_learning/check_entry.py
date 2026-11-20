"""Exercise the importable prediction branch through the unmodified static evaluator."""

from pathlib import Path
import json
import sys
import numpy as np
from protocol import ROOT, SCHEDULE

sys.path.insert(0, str(ROOT.parent))
import bench
from run_suite import EXPERT
import q1_recovery_runtime
from predictor import PredictiveRecoveryBranch


def main():
    lag = ROOT / "runs/test0/lag"
    selected = json.loads((ROOT / "selection.json").read_text())["selected"]
    gain = float(selected.split("_")[1])

    class Candidate(PredictiveRecoveryBranch):
        def __init__(self, *args):
            super().__init__(*args, ROOT / "trained/predict.onnx", gain, delay_ticks=1)

    q1_recovery_runtime.Q1RecoveryBranch = Candidate
    step = bench.mujoco.mj_step
    prev = data = None

    def queue(model, d, nstep=1):
        nonlocal prev, data
        if nstep == 4:
            command = d.ctrl.copy()
            d.ctrl[:] = prev if data is d else d.qpos[7:]
            data, prev = d, command
        return step(model, d, nstep=nstep)

    bench.mujoco.mj_step = queue
    dest = ROOT / "entry_replay"
    bench.HERE = dest / "source"
    bench.evaluate(dest / "static", SCHEDULE["test0"][0], EXPERT, public_runtime=True)
    ticks = 0
    for i in range(24):
        name = f"trial_{i:02}.npz"
        with np.load(dest / "static" / name) as a, np.load(lag / selected / "static" / name) as b:
            assert all(np.array_equal(a[k], b[k]) for k in ("initial", "qpos", "trace"))
            ticks += len(a["trace"])
    (ROOT / "entry.json").write_text(json.dumps(dict(exact_episodes=24, exact_ticks=ticks)))


if __name__ == "__main__":
    main()
