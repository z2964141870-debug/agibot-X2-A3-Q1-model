#!/usr/bin/env python3
"""Bind the verified slotted collector and audited partial-parent loader."""

from functools import partial
import importlib.util
from pathlib import Path
import sys

import parent


HERE = Path(__file__).resolve().parent
EXTENSION = HERE.parent / "supermimic_q1_extension"
sys.path.insert(0, str(EXTENSION))
spec = importlib.util.spec_from_file_location("q1_preserved_extension_train", EXTENSION / "train.py")
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
sys.path.insert(0, str(HERE.parent / "supermimic_q1_expanded"))
from slotted_sim import SlottedQ1Sim


def main():
    if trainer.Q1Sim.__name__ != "Q1Sim":
        raise ValueError("Unexpected preserved continuation collector binding")
    trainer.initialization = parent
    trainer.Q1Sim = partial(SlottedQ1Sim, slots=4)
    trainer.main()


if __name__ == "__main__":
    main()
