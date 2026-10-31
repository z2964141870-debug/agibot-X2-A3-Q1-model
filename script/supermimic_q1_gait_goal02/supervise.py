"""Reuse the original resource guard with independent GOAL02 entrypoints."""

import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("goal02_original_supervisor",
    HERE.parent / "supermimic_q1_gait_goal/supervise.py")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
original.HERE = HERE
original.__file__ = __file__


if __name__ == "__main__":
    sys.exit(original.main())
