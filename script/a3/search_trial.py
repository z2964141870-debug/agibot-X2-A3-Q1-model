"""Bounded conservative-update search using the verified training/eval adapter."""

import argparse

from script.a3 import corrected_trial
from script.a3.corrected_comparison import main as comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "verify", "evaluate", "report"))
    parser.add_argument("--trial", choices=("R08", "R09"), required=True)
    args = parser.parse_args()
    corrected_trial.configure_trial(args.trial)
    if args.action == "verify":
        corrected_trial.verify()
        return 0
    if args.action == "report":
        comparison(output=corrected_trial.OUTPUT, logs=corrected_trial.LOGS,
                   trial=args.trial, evaluated_steps=corrected_trial.EVALUATED_STEPS,
                   manifest_name=corrected_trial.OUTPUT.name + ".json")
        return 0
    return corrected_trial.train() if args.action == "train" else corrected_trial.evaluate()


if __name__ == "__main__":
    raise SystemExit(main())
