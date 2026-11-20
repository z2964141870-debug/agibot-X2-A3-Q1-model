"""Measure the public step including live reference and CGFA callbacks during five relay trials."""
import argparse
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from bench import HERE
from q1_recovery_runtime import Q1RecoveryBranch
from relay import run


def profile(expert, output, tracks, cpu):
    os.sched_setaffinity(0, {cpu})
    timings = {"track": [], "recover": [], "handoff": []}
    step = Q1RecoveryBranch.step
    def timed(self, *args):
        before = time.perf_counter_ns()
        target, tau, weight = step(self, *args)
        duration = (time.perf_counter_ns() - before) / 1e6
        mode = "recover" if weight == 1 else "track" if weight == 0 else "handoff"
        timings[mode].append(duration)
        return target, tau, weight
    with patch.object(Q1RecoveryBranch, "step", timed):
        run(expert, output, True, tracks)
    result = {"cpu_affinity": sorted(os.sched_getaffinity(0)),
              "scope": "live step with gate, expert, FK, active CGFA/reference callbacks, V2b and PD; physics outside",
              "paths": {}}
    for mode, values in timings.items():
        stats=dict(zip(("p50","p95","p99","max"), np.percentile(values, [50,95,99,100]).tolist()))
        result["paths"][mode] = dict(samples=len(values), over_2p5ms=sum(t>=2.5 for t in values), **stats)
    (output / "timing.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("expert",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("tracks",type=Path)
    p.add_argument("--cpu",type=int,default=12)
    a=p.parse_args()
    profile(a.expert, a.output.resolve(), a.tracks, a.cpu)

