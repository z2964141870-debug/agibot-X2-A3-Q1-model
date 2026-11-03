"""Native initial/final comparison; never mix it with the Primebot CPU plant."""

import argparse
from pathlib import Path

from comparison import METRICS, case_index, summary
from contracts import validate_frozen
from delivery import read
from common import FIXED, identity, require, save


def compare_traces(left, right):
    import numpy as np
    from native_metrics import tracking
    require((left["clip"], left["seed"], left["robot"], left["interface"])
            == (right["clip"], right["seed"], right["robot"], right["interface"]),
            "Native case, robot, or interface differs")
    require(left["reference_sha256"] == right["reference_sha256"]
            and left["human_sha256"] == right["human_sha256"], "Native reference changed")
    traces = []
    for row in (left, right):
        with np.load(row["trace"], allow_pickle=False) as saved:
            actual, reference, mask = (saved[key].copy() for key in ("actual", "reference", "joint_presence"))
        require(len(actual) == row["attempted_steps"] + 1
                and len(reference) == row["reference_steps"] + 1
                and actual.shape[1:] == reference.shape[1:] == (269,),
                "Native trace and episode counters differ")
        require((row["complete"] and row["failure_frame"] is None
                    and row["valid_steps"] == row["attempted_steps"] == row["reference_steps"])
                or (not row["complete"] and row["failure_frame"] == row["attempted_steps"]
                    and row["valid_steps"] == row["attempted_steps"] - 1),
                "Failed terminal frame entered native valid metrics")
        if row["complete"]:
            recomputed = tracking(actual, reference, mask, row["valid_steps"])
            require(all(np.isclose(recomputed[key], row["full_clip_metrics"][key], rtol=1e-9, atol=1e-12)
                        for key in METRICS), "Saved full-motion metric disagrees with the native trace")
        traces.append((actual, reference, mask))
    require(np.array_equal(traces[0][1], traces[1][1]) and np.array_equal(traces[0][2], traces[1][2]),
            "Native reference states or present-joint mask changed")
    steps = min(left["valid_steps"], right["valid_steps"])
    result = dict(clip=left["clip"], seed=left["seed"], common_steps=steps,
                  initial_complete=left["complete"], final_complete=right["complete"],
                  trace_identities=[identity(row["trace"]) for row in (left, right)])
    if steps:
        for label, (actual, reference, mask) in zip(("initial", "final"), traces):
            result.update({label + "_" + key: value for key, value in tracking(actual, reference, mask, steps).items()})
    return result


def execute(args):
    require(not args.output.exists(), "Preserve previous native comparison evidence")
    left, right, frozen = map(read, (args.left, args.right, args.frozen))
    require(left.get("plant") == right.get("plant") == "native_mjlab"
            and left.get("robot") == right.get("robot") == args.robot
            and left.get("interface") == right.get("interface") == args.interface,
            "Compare the same native robot, interface, and plant")
    require(left.get("iteration") == 0 and type(right.get("iteration")) is int and right["iteration"] > 0
            and left.get("training_plan") and left["training_plan"] == right.get("training_plan")
            and left.get("frozen_data") == right.get("frozen_data") == identity(args.frozen),
            "Require initial/final evaluations of the same frozen shared training plan")
    names = validate_frozen(frozen)
    names, seeds = (names, (0,)) if args.library else (FIXED, (0, 1, 2))
    a, b = (case_index(record, names, seeds, args.interface) for record in (left, right))
    pairs = [compare_traces(a[key], b[key]) for key in a]
    result = summary(a, b, pairs, names)
    result.update(robot=args.robot, interface=args.interface, pairs=pairs,
        sources=[identity(path) for path in (args.left, args.right, args.frozen)],
        comparison_scope="native mjlab initial/final; present joints and 13 pelvis-relative body positions",
        common_primebot_plant_comparison=False, g1_single_robot_control_comparison=False,
        training_set_only=True, unseen_morphology_zero_shot=False, physical_transitions=0, optimizer_updates=0)
    save(args.output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("left", "right", "frozen", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--robot", choices=("g1", "q1"), required=True)
    parser.add_argument("--interface", choices=("motion", "human"), required=True)
    parser.add_argument("--library", action="store_true")
    execute(parser.parse_args())
