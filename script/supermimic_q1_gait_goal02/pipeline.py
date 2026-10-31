"""Use established delivery and resource bounds for independent GOAL02 runs."""

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("goal02_existing_delivery",
    HERE.parent / "supermimic_q1_gait_goal_delivery/pipeline.py")
delivery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(delivery)
delivery.REVIEW = HERE
original_save = delivery.save


def save(path, value):
    if path.name == "plan.json":
        from goal02_config import contract
        value = {**value, "training_only_reporting_adapter": False,
                 "goal02_independent_reward_variant": contract()}
    original_save(path, value)


delivery.save = save


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "library", "checkpoint", "calibration", "video-suite", "output", "preflight16", "preflight4096"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--line", choices=("base", "ppo"), required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--expected-boot", required=True)
    parser.add_argument("--expected-gpu", required=True)
    parser.add_argument("--expected-oom-kill", type=int, required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--controller", action="store_true")
    args = parser.parse_args()
    args.mode = "train"
    for name in ("runtime", "library", "checkpoint", "calibration", "video_suite", "preflight16", "preflight4096"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    if len(args.code_commit) != 40 or any(char not in "0123456789abcdef" for char in args.code_commit):
        parser.error("Full independently published40-digit commit required")
    args.output = args.output.resolve()
    if not args.output.is_relative_to(delivery.ROOT / "data/supermimic_for_q1"):
        parser.error("Use the unique server project root")
    if args.controller:
        delivery.execute(args)
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
               *[value for value in sys.argv[1:] if value != "--detach"], "--controller"]
    with (delivery.ROOT / f"logs/{args.output.name}_controller.log").open("x") as stream:
        child = subprocess.Popen(command, cwd=delivery.ROOT, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    delivery.save(args.output / "launch.json", dict(pid=child.pid, command=command,
                  started_utc=delivery.stamp(), automatic_retry=False))
    return 0 if args.detach else child.wait()


if __name__ == "__main__":
    sys.exit(main())
