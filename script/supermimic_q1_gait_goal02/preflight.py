"""Exercise the new rewards on bounded native preflights with zero updates."""

import argparse
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "library", "checkpoint", "calibration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--line", choices=("base", "ppo"), required=True)
    parser.add_argument("--envs", type=int, choices=(16, 4096), required=True)
    parser.add_argument("--upper-alpha", type=float, default=.5, choices=(.5,))
    args = parser.parse_args()
    from goal02_runtime import install
    native, make_ppo = install(args.runtime, args.output)
    if args.line == "base":
        module = load(HERE.parent / "supermimic_q1_gait_goal/preflight.py", "goal02_base_preflight")
    else:
        module = load(HERE.parent / "supermimic_q1_contact_comp/preflight.py", "goal02_ppo_preflight")

        class Adapter(native.CandidateAdapter):
            def __init__(self, env, teacher, thresholds, upper_alpha=.5):
                super().__init__(env, teacher, thresholds, upper_alpha, runtime=args.runtime)

        _, reference = native.reward_types(args.runtime)
        module.Adapter = Adapter
        module.make_ppo = make_ppo
        module.ContactReference = lambda mdp: reference(mdp, args.runtime / "q1_assets/scene.xml")
    module.run(args)


if __name__ == "__main__":
    main()
