"""Check video delivery using existing CPU rollout traces, without new dynamics."""

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "primebot_q1"))
from common import FIXED, identity, now, require, save
from video import render_case


def render(review, xml, output):
    source = json.loads(review.read_text())
    require(source["status"] == "PASSED" and len(source["cases"]) == 9,
            "Use the completed fixed CPU smoke observations")
    cases = [case for case in source["cases"] if case["seed"] == 0]
    require(len(cases) == 3 and {c["clip"] for c in cases} == set(FIXED),
            "All three fixed clips are required")
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="RUNNING", started_utc=now(), source=identity(review),
                  new_physics_steps=0, new_optimizer_updates=0, videos=[])
    try:
        for case in cases:
            before = identity(case["trace"])
            directory = output / case["clip"]
            directory.mkdir()
            result["videos"].append(render_case(xml, case, directory / "tracking.mp4"))
            require(identity(case["trace"]) == before, "Saved rollout changed during rendering")
            save(output / "progress.json", result)
        result.update(status="PASSED", ended_utc=now())
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}", ended_utc=now())
        raise
    finally:
        save(output / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("review", "xml", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    render(args.review, args.xml, args.output)
