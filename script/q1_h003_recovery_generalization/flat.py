"""Verify both selected models leave the two nominal walking references unchanged."""

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from run_suite import TRACKS, schedule
from relay import run


def main():
    records = {}
    for arm in ("selected", "late_selected"):
        folder = ROOT / "flat" / arm
        if not folder.exists():
            run(ROOT / "models" / (arm + ".onnx"), folder, True, TRACKS, cases=schedule("flat"))
        ticks = 0
        for i in range(2):
            name = f"relay_{i}.npz"
            with np.load(folder / name) as a, np.load(ROOT.parent / "flat/published" / name) as b:
                assert all(
                    np.array_equal(a[k], b[k]) for k in ("qpos", "trace", "reference", "loads")
                )
                assert not a["trace"][:, 5].any()
                ticks += len(a["trace"])
        records[arm] = ticks
    (ROOT / "flat.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
