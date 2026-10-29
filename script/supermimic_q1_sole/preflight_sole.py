"""Reuse the proven native RF16 preflight with the isolated sole candidate."""

import argparse
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def run(args):
    sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_contact_comp")]
    from sole_adapter import Adapter
    from sole_reference import SoleReference

    path = HERE.parent / "supermimic_q1_contact_comp/preflight.py"
    spec = importlib.util.spec_from_file_location("q1_contact_preflight_for_sole", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    scene = args.runtime / "q1_assets/scene.xml"

    class SoleAdapter(Adapter):
        def __init__(self, env, teacher, thresholds, upper_alpha=.5):
            super().__init__(env, teacher, thresholds, upper_alpha, scene=scene)

    original.Adapter = SoleAdapter
    original.ContactReference = lambda mdp: SoleReference(mdp, scene)
    original.run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "library", "checkpoint", "calibration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--envs", type=int, choices=(16, 4096), required=True)
    parser.add_argument("--upper-alpha", type=float, default=.5)
    run(parser.parse_args())
