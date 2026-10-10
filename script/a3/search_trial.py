"""Bounded conservative-update search using the verified training/eval adapter."""

import argparse

from script.a3 import corrected_trial
from script.a3.checkpoint_store import atomic_json, sha256
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
        result = comparison(output=corrected_trial.OUTPUT, logs=corrected_trial.LOGS,
                   trial=args.trial, evaluated_steps=corrected_trial.EVALUATED_STEPS,
                   manifest_name=corrected_trial.OUTPUT.name + ".json")
        additions = [corrected_trial.OUTPUT / "config_diff.json", corrected_trial.OUTPUT / "fixed_input_drift.json"]
        result["search_evidence"] = {path.stem: corrected_trial.read(path) for path in additions if path.exists()}
        result["search_artifacts"] = [dict(path=str(path.relative_to(corrected_trial.ROOT)), sha256=sha256(path),
                                          bytes=path.stat().st_size) for path in additions if path.exists()]
        for name in ("search_trial.py", "search_drift.py", "test_search_trial.py"):
            path = corrected_trial.ROOT / "script/a3" / name
            result["code"].append(dict(path=str(path.relative_to(corrected_trial.ROOT)), sha256=sha256(path)))
        atomic_json(corrected_trial.OUTPUT / "comparison.json", result)
        atomic_json(corrected_trial.ROOT / "data/manifests" / (corrected_trial.OUTPUT.name + ".json"), result)
        return 0
    return corrected_trial.train() if args.action == "train" else corrected_trial.evaluate()


if __name__ == "__main__":
    raise SystemExit(main())
