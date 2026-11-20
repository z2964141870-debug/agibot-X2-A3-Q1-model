"""Independent rerun of the frozen delayed motion on the first held-out ten pushes."""

import json
from pathlib import Path
import numpy as np

import worker
from protocol import ROOT
from run import read, sha


def main():
    winner = read(ROOT / "motion_selection.json")
    name = winner["selected"]
    assert winner["onnx_sha256"] == sha(ROOT / "motion" / (name + ".onnx"))
    target = ROOT / "replay"
    target.mkdir(exist_ok=False)
    (target / "motion").symlink_to(ROOT / "motion", target_is_directory=True)
    worker.ROOT = target
    worker.execute("test0", "lag", name, "relay")
    sub = Path("runs/test0/lag") / name / "relay"
    ticks = 0
    assert read(target / sub / "relay.json") == read(ROOT / sub / "relay.json")
    for i in range(10):
        path = sub / f"relay_{i}.npz"
        with np.load(ROOT / path) as a, np.load(target / path) as b:
            for key in ("qpos", "trace", "reference", "loads", "kicks", "reanchors"):
                assert np.array_equal(a[key], b[key]), (i, key)
            ticks += len(a["trace"])
    (ROOT / "replay.json").write_text(
        json.dumps(dict(episodes=10, ticks=ticks, sha256=winner["onnx_sha256"]), indent=2)
    )


if __name__ == "__main__":
    main()
