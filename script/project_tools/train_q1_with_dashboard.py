#!/usr/bin/env python3
"""Future Q1 training entry: passive episode logging and a terminal dashboard."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

from plot_training_dashboard import create_dashboard, identity, load_run


class EpisodeAccumulator:
    def __init__(self, num_envs, reward_max):
        if num_envs < 1 or not math.isfinite(reward_max) or reward_max <= 0:
            raise ValueError("Positive environment count and maximum reward are required")
        self.reward_max = reward_max
        self.returns = [0.] * num_envs
        self.lengths = [0] * num_envs
        self.step = 0
        self.start = 1
        self.clear_window()

    def clear_window(self):
        self.count = self.hard = self.length_sum = 0
        self.normalized_sum = 0.

    def update(self, rewards, terminated, timeout):
        if not len(rewards) == len(terminated) == len(timeout) == len(self.lengths):
            raise ValueError("Episode telemetry needs every environment at each control step")
        self.step += 1
        for index, (reward, hard, truncated) in enumerate(zip(rewards, terminated, timeout)):
            if not math.isfinite(reward):
                raise ValueError("Nonfinite recorded reward")
            self.returns[index] += reward
            self.lengths[index] += 1
            if hard or truncated:
                self.count += 1
                self.hard += int(bool(hard))
                self.length_sum += self.lengths[index]
                self.normalized_sum += self.returns[index] / (self.reward_max * self.lengths[index])
                self.returns[index] = 0.
                self.lengths[index] = 0

    def flush(self):
        row = {"schema": "Q1_COMPLETED_EPISODE_WINDOW_V1", "step": self.step,
               "window_start_step": self.start, "completed_episodes": self.count,
               "hard_terminations": self.hard, "normalized_return_sum": self.normalized_sum,
               "length_steps_sum": self.length_sum, "reward_max": self.reward_max}
        self.start = self.step + 1
        self.clear_window()
        return row


class RecordingSim:
    def __init__(self, sim, output):
        self.sim = sim
        self.output = output
        self.episodes = EpisodeAccumulator(sim.E, sim.heads.reward_cfg.max_step_reward)
        self.stream = None

    def __getattr__(self, name):
        return getattr(self.sim, name)

    def record(self):
        if self.stream is None:
            self.stream = (self.output / "episodes.jsonl").open("x")
        self.stream.write(json.dumps(self.episodes.flush(), allow_nan=False) + "\n")
        self.stream.flush()

    def step(self, *args, **kwargs):
        result = self.sim.step(*args, **kwargs)
        self.episodes.update(*(result[key].detach().cpu().tolist()
                               for key in ("reward", "terminated", "timeout")))
        if self.episodes.step % 25 == 0:
            self.record()
        return result

    def close(self):
        try:
            if self.episodes.step >= self.episodes.start:
                self.record()
        finally:
            if self.stream is not None:
                self.stream.close()
            self.sim.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dashboard-label", default="Q1")
    args, forwarded = parser.parse_known_args()
    root, output = args.root.resolve(strict=True), args.output.resolve()
    if not output.is_relative_to(root / "data/supermimic_for_q1") or output.exists():
        parser.error("Start an independent training run in the unified Q1 data directory")
    sys.path.insert(0, str(root / "script/supermimic_q1_reproduction"))
    import train_reproduction
    original_factory = train_reproduction.Q1Sim

    def recorded_factory(*sim_args, **sim_kwargs):
        sim = original_factory(*sim_args, **sim_kwargs)
        metadata = {"schema": "Q1_TRAINING_REPORTING_V1", "entry": identity(__file__),
            "reward_max": sim.heads.reward_cfg.max_step_reward,
            "reward_cfg": asdict(sim.heads.reward_cfg),
            "reward_max_source": "Live AnalyticHeads.reward_cfg.max_step_reward",
            "episode_normalization": "mean_i[sum_t(reward_i)/(length_i * reward_max)] over completed episodes",
            "episode_completion": "terminated OR timeout; hard flag retained even when both are true",
            "incomplete_episodes": "Excluded from completed-episode aggregates",
            "logging_interval": 25, "plot_window_training_iterations": 500}
        (output / "reporting_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        return RecordingSim(sim, output)

    train_reproduction.Q1Sim = recorded_factory
    sys.argv = [str(root / "script/supermimic_q1_reproduction/train_reproduction.py"),
                "--root", str(root), "--output", str(output), *forwarded]
    training_error = None
    try:
        train_reproduction.main()
    except BaseException as exc:
        training_error = exc
        raise
    finally:
        if (output / "metrics.jsonl").exists() and (output / "status.json").exists():
            try:
                run = load_run(output, args.dashboard_label)
                receipt = create_dashboard([run], output.parent / (output.name + "_dashboard"),
                                           title="Q1 SuperMimic: " + args.dashboard_label)
                print(json.dumps(receipt, allow_nan=False), flush=True)
            except Exception as exc:
                failure = {"status": "DASHBOARD_FAILED", "error": f"{type(exc).__name__}: {exc}",
                           "training_error": str(training_error) if training_error is not None else None}
                (output / "dashboard_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
                if training_error is None:
                    raise


if __name__ == "__main__":
    main()
