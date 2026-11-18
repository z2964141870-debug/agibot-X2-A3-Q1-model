"""Replay the historical token-service route on saved garment and feedback data."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from replay_sonic_stand import (
    MJ_TO_IL, feedback, idle_windows, observations, predict,
)

LOWER_JOINTS = (1, 2, 4, 5, 7, 8, 10, 11)


def stabilize_capture(capture, *, legs=False, root=False):
    edited = capture.copy()
    anchor = int(np.searchsorted(capture[:, 0], capture[0, 0] + 8.0, side="right") - 1)
    if legs:
        points = edited[:, 1:73].reshape(-1, 24, 3)
        points[:, LOWER_JOINTS, :] = points[anchor, LOWER_JOINTS, :].copy()
    if root:
        edited[:, 73:77] = edited[anchor, 73:77]
    return edited


def run_tokens(session, capture, base, cycles):
    from token_service_core import TokenServiceCore

    ori = session.get_modelmeta().custom_metadata_map.get("ori_mode")
    names = [item.name for item in session.get_outputs()]
    output = "motion_token" if "motion_token" in names else names[0]
    core = TokenServiceCore(lambda obs: session.run([output], {"smpl_obs": obs})[0].ravel(),
                            ori_mode=ori, planner_cmd=True, now=10.)
    capture = capture[np.argsort(capture[:, 0])]
    human_clock = capture[:, 0] - capture[0, 0] - 8.25 + 10.
    tokens, ticks, events = [], [], Counter()
    position = 0
    for i in range(cycles):
        now = core.now()
        while position < len(capture) and human_clock[position] <= now:
            row = capture[position]
            core.feed_intent(float(human_clock[position]), row[1:73].reshape(24, 3), row[73:77],
                             engaged=1., wrist_pr=np.zeros(4, dtype=np.float32))
            position += 1
        core.feed_debug(now, base[i, :4], body_q_mj=base[i, 7:38].astype(np.float32))
        result = core.tick()
        events[result.reason] += 1
        if "wb_state_sent" in result.diag:
            events["planner_state_events"] += 1
        if result.wrist_targets is not None:
            events["planner_wrist_events"] += 1
        if result.published:
            tokens.append(result.token)
            ticks.append(i)
    return np.asarray(tokens), np.asarray(ticks), dict(events), ori


def measure(decoder, tokens, proprio):
    reference = np.zeros_like(proprio)
    reference[:, 680:] = proprio[:, 680:]
    reference[:, :64] = tokens
    held = reference.copy()
    held[:, :64] = tokens[0]
    dynamic = predict(decoder, reference)
    fixed = predict(decoder, held)
    if dynamic.shape != (len(tokens), 31) or not np.isfinite([dynamic, fixed]).all():
        raise ValueError("invalid saved-token policy output")
    delta = np.abs(dynamic - fixed)
    return dict(ticks=len(tokens), swing_from_changing_token={
        region: dict(p95_abs_action_delta=float(np.percentile(np.max(delta[:, indices], axis=1), 95)),
                     max_abs_action_delta=float(delta[:, indices].max()))
        for region, indices in (("legs", MJ_TO_IL[:12]), ("waist", MJ_TO_IL[12:15]),
                                ("left_arm", MJ_TO_IL[15:22]), ("right_arm", MJ_TO_IL[22:29]))
    }, waist_roll_abs_target_p95_deg=float(np.percentile(np.abs(np.degrees(dynamic[:, 8] * .48510106)), 95)),
                dynamic_action_clip_ticks=int((np.max(np.abs(dynamic), axis=1) > 20).sum()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("probe", "feedback", "idle", "receiver", "policies"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--isolate-input", action="store_true",
                        help="Test held leg joints and root orientation on the historical pair only")
    args = parser.parse_args()
    import onnxruntime as ort

    base, _, wall = feedback(args.feedback, 180)
    capture = np.loadtxt(args.receiver, delimiter=",")
    if (capture.shape[1:] != (77,) or not np.isfinite(capture).all()
            or np.ptp(capture[:, 0]) < 11.85):
        raise ValueError("expected the complete saved SONIC receiver recording")
    obs = observations(args.probe, base, idle_windows(args.idle, wall, np.zeros(len(base)), 0))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    def load(name):
        return ort.InferenceSession(str(args.policies / name), sess_options=options,
                                    providers=["CPUExecutionProvider"])

    inputs = {
        "historical_runbook_pair": ("x2_smpl_tokenizer_v11release.onnx", "x2_sonic_s1ft16000_g1_token.onnx"),
        "v16_pair": ("x2_sonic_v16ft8_45000_smpl_tokenizer.onnx", "x2_sonic_v16ft8_45000_g1_token.onnx"),
    }
    cases = {}
    if args.isolate_input:
        capture = capture[np.argsort(capture[:, 0])]
        encoder, decoder = (load(name) for name in inputs["historical_runbook_pair"])
        for name, legs, root in (("unmodified", False, False), ("legs_held", True, False),
                                 ("root_held", False, True), ("legs_and_root_held", True, True)):
            modified = stabilize_capture(capture, legs=legs, root=root)
            tokens, ticks, events, _ = run_tokens(encoder, modified, base, len(base))
            cases[name] = (dict(events=events, **measure(decoder, tokens, obs[ticks]))
                           if len(tokens) else dict(events=events, ticks=0))
        print(json.dumps(dict(cases=cases, planner_messages=0, hardware_commands=0), allow_nan=False, indent=2))
        return
    for label, (encoder, decoder) in inputs.items():
        tokens, ticks, events, orientation = run_tokens(load(encoder), capture, base, len(base))
        if len(tokens) < 100 or tokens.shape[1:] != (64,) or not np.isfinite(tokens).all():
            raise ValueError(f"insufficient valid motion tokens for {label}")
        cases[label] = dict(tokenizer=encoder, decoder=decoder, orientation=orientation,
                            events=events, **measure(load(decoder), tokens, obs[ticks]))
    print(json.dumps(dict(cases=cases, history="identical saved feedback and previous actions per pair",
                          comparison="changing versus first-held token; not a tracking-error measurement",
                          planner_messages=0, hardware_commands=0), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
