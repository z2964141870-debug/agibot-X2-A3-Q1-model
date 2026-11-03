"""Call the pinned Primebot trainer with its frozen single-GPU recipe."""

import argparse
from dataclasses import fields
import json
from pathlib import Path
import tomllib

from common import require


def run(plan):
    from supermimic.train import TrainCfg, train
    recipe = tomllib.loads(Path(plan["recipe"]).read_text())["train"]
    recipe.update(plan["overrides"])
    for key in ("robot_xml", "motion_dir", "output_dir", "motion_cache_dir"):
        if recipe.get(key) is not None:
            recipe[key] = Path(recipe[key])
    require(set(recipe) <= {f.name for f in fields(TrainCfg)}, "Unknown upstream training settings")
    require(recipe["iterations"] == (8 if plan["purpose"] == "profile" else 250000),
            "Do not shorten the formal recipe")
    require(not recipe.get("resume") and not recipe.get("warm_start") and not recipe.get("small_networks"),
            "Require fresh full-network training")
    train(TrainCfg(**recipe))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.plan.read_text()))
