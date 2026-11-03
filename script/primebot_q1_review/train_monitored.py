"""Use the original Primebot recipe with externally evaluated early stopping."""

import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "primebot_q1"))
from common import require


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    require(plan["purpose"] == "train", "Monitoring is attached to a fresh formal run")
    sys.path.insert(0, str(HERE))
    os.environ["PYTHONPATH"] = os.pathsep.join((str(HERE), os.environ["PYTHONPATH"]))
    os.environ["Q1_LIVE_MONITOR"] = "1"
    # The existing matching user-space graphics libraries are optional; the
    # caller must explicitly supply their verified location to use them.
    graphics = os.environ.get("Q1_REVIEW_GRAPHICS")
    if graphics:
        os.environ.update(LD_LIBRARY_PATH=str(Path(graphics) / "lib"),
            __EGL_VENDOR_LIBRARY_FILENAMES=str(HERE.parent / "morphmimic_cross_robot/nvidia_egl_595_20261102.json"),
            PYOPENGL_PLATFORM="egl", EGL_PLATFORM="surfaceless")
    from train import run
    run(plan)
