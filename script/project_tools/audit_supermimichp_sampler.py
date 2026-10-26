#!/usr/bin/env python3
"""Reproduce the short-clip sampler boundary with isolated code and fake inputs."""

import argparse
import ast
import json
import subprocess
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    path = "src/unisupermimic/rl/algos/supermimic/learner.py"
    source = subprocess.check_output(["git", "-C", args.repository, "show", args.commit + ":" + path], text=True)
    tree = ast.parse(source)
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "_sample_valid_starts")
    namespace = {}
    isolated = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(isolated), path, "exec"), namespace)

    class ProbeLimit(Exception):
        pass

    class UnusedValues:
        def __getitem__(self, key):
            return self

    class EmptyMask:
        def nonzero(self, **kwargs):
            return self

        def squeeze(self, dimension):
            return []

    class FakeReplay:
        calls = 0

        def sample_starts(self, *values):
            self.calls += 1
            if self.calls > 4:
                raise ProbeLimit("Bounded probe stopped a fifth unsuccessful sampling attempt")
            return {"f": UnusedValues()}, {"s": UnusedValues()}

    replay = FakeReplay()
    owner = SimpleNamespace(buffer=replay, cfg=SimpleNamespace(hist=10, horizon=32),
        mdp=SimpleNamespace(valid_imagination_start=lambda frames, horizon: EmptyMask()))
    limit_reached = False
    try:
        namespace["_sample_valid_starts"](owner, 2)
    except ProbeLimit:
        limit_reached = True
    if not limit_reached:
        raise AssertionError("The hypothesized unbounded retry was not reproduced")
    print(json.dumps({"status": "SHORT_CLIP_UNBOUNDED_RETRY_REPRODUCED", "commit": args.commit,
        "path": path, "line": method.lineno, "unsuccessful_attempts_before_probe_interrupt": 4,
        "trigger": "No reference frame satisfies frame + H32 <= inclusive clip end, e.g. every clip has <=32 frames",
        "actual_training_or_upstream_imports": False, "optimizer_updates": 0,
        "scope": "Isolated actual sampler AST with fake replay/mask; current 100-clip X2 run is not shown to hit this boundary",
        "suggestion": "Check that at least one motion admits H32 at initialization; bound retries and raise a clear error"}, indent=2))


if __name__ == "__main__":
    main()
