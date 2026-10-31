"""Run GOAL02 with the established training clock, gates and review callbacks."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_gait_goal_review"),
               str(HERE.parent / "supermimic_q1_gait_goal"), str(HERE.parent / "project_tools")]


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
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--check-imports", action="store_true")
    args = parser.parse_args()
    from goal02_runtime import install
    review = load(HERE.parent / "supermimic_q1_gait_goal_review/train.py", "goal02_review_training")
    original_loader = review.load_training_module

    def loader(runtime):
        native, make_ppo = install(runtime, args.output)
        module = original_loader(runtime)
        module.make_ppo = make_ppo
        return module

    review.load_training_module = loader
    if args.check_imports:
        loader(args.runtime)
        print(json.dumps(dict(status="GOAL02_TRAINING_IMPORTS_PASSED", new_physical_steps=0,
                              policy_inference_calls=0, optimizer_updates=0)))
    else:
        review.run(args)


if __name__ == "__main__":
    main()
