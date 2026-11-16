"""Replay saved live advice and attribute a joint step by swapping ONNX input groups."""
import argparse
from itertools import product
import json
import math
from pathlib import Path

import numpy as np

from .artifacts import MODEL_SHA, sha
from .replay_reentry_attribution import Capture, run_target
from .runtime import Runtime

CLOCK = "saved_live_jump_replay"


def attribute(runtime, session, old, new, joint):
    from deploy.sim2sim.obs import PROPRIO_DIM

    histories = [item["feeds"]["obs_history"].reshape(-1, PROPRIO_DIM) for item in (old, new)]
    cut = PROPRIO_DIM-len(runtime.runner.last_action)
    if cut <= 0 or set(old["feeds"]) != set(new["feeds"]):
        raise ValueError("Unexpected policy input contract")
    changed_constants = {name for name, value in old["feeds"].items()
                         if name != "obs_history" and not name.startswith("human_") and name != "goal_human"
                         and not np.array_equal(value, new["feeds"][name])}
    if changed_constants:
        raise ValueError(f"Other policy inputs changed: {sorted(changed_constants)}")
    cases = {}
    for reference, state, action in product((0, 1), repeat=3):
        source = (old, new)[reference]
        history = np.concatenate([histories[state][:, :cut], histories[action][:, cut:]], axis=-1)
        feeds = dict(source["feeds"], obs_history=history.reshape(source["feeds"]["obs_history"].shape))
        q = run_target(session, runtime.robot, feeds)
        if reference == state == action and not np.array_equal(q, (old, new)[reference]["target"]):
            raise ValueError("Captured feeds did not reproduce replay outputs")
        cases[f"reference{reference}_measured{state}_actions{action}"] = dict(
            target_deg=math.degrees(q[joint]), delta_deg=math.degrees(q[joint]-old["target"][joint]))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("policy-root", "trace", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--joint", required=True)
    parser.add_argument("--max-parity-error-deg", type=float, default=math.degrees(2e-5))
    args = parser.parse_args()
    if not 0 < args.max_parity_error_deg <= .05:
        parser.error("Specify a bounded positive diagnostic replay tolerance")
    runtime = Runtime(args.policy_root, max_age_ns=120_000_000, max_interp_gap_ns=100_000_000)
    joint = runtime.robot.native_joint_names.index(args.joint)
    session = runtime.runner.session
    captured = {}
    seen = clock = 0
    max_error = 0.
    with args.trace.open() as trace:
        for line in trace:
            row = json.loads(line)
            if row["kind"] == "trace_end":
                break
            now = row["received_ns"]
            if now < clock:
                raise ValueError("Saved request timestamps are not ordered")
            clock = now
            request = row["request"]
            measure = request["kind"] == "x2_measurement"
            capture = measure and request["sequence"] in (args.sequence-1, args.sequence)
            observer = Capture(session) if capture else None
            runtime.runner.session = observer if capture else session
            reply = runtime.handle(request, now, CLOCK)
            expected = row["reply"]
            if reply.get("publish_allowed") or reply.get("reset") or reply["ready"] != expected["ready"]:
                raise ValueError("Replayed response differs from the saved no-publish contract")
            if request["kind"] == "morph_human_frame" and reply.get("accepted_sequence") != expected.get("accepted_sequence"):
                raise ValueError("HumanFrame admission changed on replay")
            if not reply["ready"]:
                continue
            seen += 1
            q = np.asarray(reply["q_target_rad"])
            want = np.asarray(expected["q_target_rad"])
            if not np.isfinite(q).all() or q.shape != want.shape:
                raise ValueError("Invalid ONNX output shape")
            max_error = max(max_error, float(np.max(np.abs(q-want))))
            if capture:
                captured[request["sequence"]] = dict(target=q, expected=want, feeds=observer.feeds)
                if request["sequence"] == args.sequence:
                    break
    if max_error > math.radians(args.max_parity_error_deg) or set(captured) != {args.sequence-1, args.sequence}:
        raise ValueError(f"Replay parity failed: {max_error:.6g} rad")
    a,b = (captured[i] for i in (args.sequence-1, args.sequence))
    result = dict(joint=args.joint, compared_sequences=[args.sequence-1, args.sequence], replayed_ready=seen,
                  max_saved_error_rad=max_error, saved_targets_deg=[math.degrees(x["expected"][joint]) for x in (a,b)],
                  max_parity_error_deg=args.max_parity_error_deg,
                  crossed_inputs=attribute(runtime, session, a, b, joint), source_sha256=sha(args.trace),
                  policy_sha256=MODEL_SHA, hardware_commands=0)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
