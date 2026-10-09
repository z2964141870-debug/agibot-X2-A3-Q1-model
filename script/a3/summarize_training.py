"""Save a dated snapshot of TensorBoard training metrics, not evaluation scores."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    events = EventAccumulator(str(args.run_dir / "tensorboard"), size_guidance={"scalars": 0})
    events.Reload()
    tags = set(events.Tags()["scalars"])
    names = [
        "Train/it", "Train/tot_timesteps", "Train/collection_time", "Train/learn_time",
        "Objective/rewards", "Objective/length", "Loss/policy_avg", "Loss/value_avg",
        "Policy/approxkl_avg", "Policy/clipfrac_avg", "Env/Metrics/motion/error_body_pos",
        "Env/Metrics/motion/error_joint_pos", "Terminations/time_out",
    ]
    metrics = {}
    for name in names:
        if name not in tags:
            continue
        samples = events.Scalars(name)
        if not samples:
            continue
        values = [sample.value for sample in samples]
        metrics[name] = {
            "first": values[0], "latest": values[-1], "latest_step": samples[-1].step,
            "last20_mean": sum(values[-20:]) / len(values[-20:]),
            "finite": all(math.isfinite(value) for value in values),
        }
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(args.run_dir.resolve()),
        "scope": "training metrics snapshot; not a fixed-condition evaluation",
        "metrics": metrics,
    }
    if not metrics or not all(value["finite"] for value in metrics.values()):
        raise RuntimeError("Missing training metrics or nonfinite scalar")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
