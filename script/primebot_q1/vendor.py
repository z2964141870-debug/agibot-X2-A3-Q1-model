"""Create an audited V3 adaptation while preserving the pinned source."""

import argparse
import shutil
from pathlib import Path

from common import UPSTREAM, identity, require, save


def adapt(source, output):
    require(source.name == "SuperMimic-" + UPSTREAM, "Unexpected pinned source directory")
    shutil.copytree(source, output)
    patches = []

    def replace(relative, before, after):
        path = output / relative
        content = path.read_text()
        require(content.count(before) == 1, "Pinned patch anchor changed: " + relative)
        prior = identity(path)
        path.write_text(content.replace(before, after))
        patches.append(dict(file=relative, before=prior, after=identity(path),
                            removed=before, added=after))

    replace("src/supermimic/data.py",
            'str(data["robot_name"].item()) != "q1_v2"',
            'str(data["robot_name"].item()) != "q1_v3"')
    replace("src/supermimic/filter_motions.py",
            'metadata.get("robot_name") != "q1_v2"',
            'metadata.get("robot_name") != "q1_v3"')
    replace("src/supermimic/train.py", "  try:\n    # Warmup and extra collection",
            "  from primebot_hooks import save_zero, install_counts, exposure_snapshot\n"
            "  install_counts(algorithm)\n"
            "  save_zero(checkpoint, (model, actor, critic, robot), cfg, capture_rng)\n"
            "  try:\n    # Warmup and extra collection")
    replace("src/supermimic/train.py", "      completed_iteration = iteration + 1\n",
            "      metrics.update({\"actual_\" + k: v for k, v in algorithm._q1_actual_optimizer_counts.items()})\n"
            "      completed_iteration = iteration + 1\n")
    replace("src/supermimic/train.py", '      "iteration": completed_iteration,\n',
            '      "actual_optimizer_counts": dict(algorithm._q1_actual_optimizer_counts),\n'
            '      "iteration": completed_iteration,\n')
    replace("src/supermimic/train.py", "      if stopping:\n        break\n",
            "      exposure_snapshot(checkpoint, global_steps=metrics[\"global_env_steps\"], cfg=cfg)\n"
            "      if stopping:\n        break\n")
    originals = [identity(p) for p in sorted(source.rglob("*")) if p.is_file()
                 and not any(s in p.parts for s in (".venv", "__pycache__", ".git"))]
    record = dict(upstream_commit=UPSTREAM, original_source=originals,
                  adaptations=patches, robot_version="q1_v3",
                  original_diagnostic_strings_may_still_say_q1_v2=True,
                  learning_equations_changed=False)
    save(output / "q1_v3_adaptation.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    adapt(args.source.resolve(strict=True), args.output.resolve())
