"""Save exact physical states immediately before the first command of each rise attempt."""

import sys
from pathlib import Path
import numpy as np
import worker
import recovery
from protocol import ROOT
from concurrent.futures import ThreadPoolExecutor
import subprocess


def dump(split, surface, kind):
    rows = []
    marker = False
    original = recovery.Recovery._step

    def call(self, *args):
        nonlocal marker
        old_mode = self.state[0, 1]
        result = original(self, *args)
        marker = bool(self.state[0, 1] == 2 and self.state[0, 3] == 1 and old_mode != 2)
        return result

    recovery.Recovery._step = call
    native = worker.bench.mujoco.mj_step

    def capture(m, d, nstep=1):
        nonlocal marker
        if marker and nstep == 4:
            # worker already placed the pending command on data.ctrl.
            rows.append(np.r_[d.qpos, d.qvel, d.ctrl].copy())
            marker = False
        return native(m, d, nstep=nstep)

    worker.bench.mujoco.mj_step = capture
    worker.ROOT = ROOT / "starts_replay"
    worker.execute(split, surface, "published", kind)
    np.save(
        ROOT / "starts_replay" / f"{split}_{surface}_{kind}.npy", np.array(rows).reshape(-1, 79)
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dump(*sys.argv[1:])
    else:
        jobs = [
            (s, f, k)
            for s in ("collect0", "collect1")
            for f in ("lag", "rough")
            for k in ("static", "relay")
        ]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(
                pool.map(
                    lambda j: subprocess.run(
                        [sys.executable, __file__, *j], check=True, stdout=subprocess.DEVNULL
                    ),
                    jobs,
                )
            )
