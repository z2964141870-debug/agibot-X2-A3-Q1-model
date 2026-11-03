"""Add observation boundaries to an independent copy of the audited V3 trainer."""

import argparse
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "primebot_q1"))
from common import identity, require, save


def adapt(source, output):
    shutil.copytree(source, output, ignore=shutil.ignore_patterns("__pycache__", ".git", ".venv"))
    path = output / "src/supermimic/train.py"
    original = identity(path)
    text = path.read_text()
    patches = (
        ("  save_zero(checkpoint, (model, actor, critic, robot), cfg, capture_rng)\n",
         "  save_zero(checkpoint, (model, actor, critic, robot), cfg, capture_rng)\n"
         "  from live import observe as observe_q1\n"
         "  observe_q1(checkpoint, cfg, 0, (model, actor, critic, robot), capture_rng)\n"),
        ('      exposure_snapshot(checkpoint, global_steps=metrics["global_env_steps"], cfg=cfg)\n',
         '      exposure_snapshot(checkpoint, global_steps=metrics["global_env_steps"], cfg=cfg)\n'
         "      if observe_q1(checkpoint, cfg, completed_iteration, (model, actor, critic, robot), capture_rng):\n"
         "        checkpoint(include_replay=True)\n"
         "        break\n"))
    for before, after in patches:
        require(text.count(before) == 1, "Monitored source anchor changed")
        text = text.replace(before, after)
    path.write_text(text)
    save(output / "live_review_adaptation.json", dict(before=original, after=identity(path),
        patches=[dict(before=a, after=b) for a, b in patches],
        learning_equations_changed=False, maximum_recipe_iterations=250000,
        early_stop="user-authorized three consecutive scheduled fixed tracking gates",
        source_directory=str(source)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    adapt(args.source, args.output)
