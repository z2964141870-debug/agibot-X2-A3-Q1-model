#!/usr/bin/env python3
"""Robot-side SMPL token service — X2 port of upstream
gear_sonic/scripts/pc2_pico_token_service.py (pinned c0036a4).

The laptop/wearer-side sender streams human INTENT only (SMPL joints,
engage flag, wrist targets) on :5573; THIS service, running next to the
deploy like pc2_kplanner_onnx.py does, assembles the 840-d SMPL
observation using LOCAL robot state (deploy x2_debug echo), runs the
exported smpl-tokenizer ONNX (CPU-cheap MLP), and publishes tokens +
engaged + wrist reference into the deploy's token source on :5574.
Wifi loss stops the intent stream -> tokens stop -> the hybrid deploy
falls back to the kplanner/pad, by construction.

Wire (sender -> here, topic ``pico_intent``, SUB-binds :5573):
    smpl_joints    f32[72]   root-relative joints (x fwd, y left, z up)
    human_quat     f32[4]    global orient (wxyz)
    engaged        f32[1]
    wrist_pr       f32[4]    [l_pitch l_roll r_pitch r_roll] targets (rad)
    frame_index    i64[1]

Usage (robot, after ritual is up with the whole-body flags):
    python3 x2_smpl_token_service.py \
        --tokenizer ~/gear-sonic/policies/x2_smpl_tokenizer_v11release.onnx \
        --debug-port 5557
All per-tick logic lives in token_service_core.TokenServiceCore (1:1
upstream port); this file is only sockets + the 50 Hz loop.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from token_service_core import CONTROL_DT, TokenServiceCore  # noqa: E402
from wire_format import pack_pose_message, unpack_message  # noqa: E402

_PC2_PREFIX = os.environ.get("PC2_PREFIX", os.path.expanduser("~/gear-sonic"))


def sync_and_tick(core: TokenServiceCore, now: float):
    """Sync core's internal clock with wall monotonic time before ticking."""
    core._now = float(now)
    return core.tick()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokenizer", required=True,
                    help="x2_smpl_tokenizer_*.onnx (smpl_obs 840 -> token 64)")
    ap.add_argument("--intent-port", type=int, default=5573)
    ap.add_argument("--token-port", type=int, default=5574)
    ap.add_argument("--debug-host", default="127.0.0.1")
    ap.add_argument("--debug-port", type=int, default=5557,
                    help="deploy x2_debug (robot ritual 5557; sim stack 5659)")
    ap.add_argument("--planner-cmd-host", default="127.0.0.1")
    ap.add_argument("--planner-cmd-port", type=int, default=0,
                    help="kplanner planner_cmd SUB port (robot 5563). >0 = "
                         "forward wb_state + wrist_targets. 0 = off.")
    ap.add_argument("--operator-root-level", default=None,
                    help="level the OPERATOR root before tokenization: "
                         "'off'|'zero'|'clamp:<deg>'|'offset:<deg>'. "
                         "Default: $TOKEN_SVC_OPERATOR_ROOT_LEVEL or off.")
    ap.add_argument("--operator-lean-cap", type=float, default=None,
                    help="cap operator forward torso lean (deg); 0 = off.")
    ap.add_argument("--operator-reach-cap", type=float, default=None,
                    help="cap operator forward hand reach (m); 0 = off.")
    ap.add_argument("--ori-mode", choices=("full", "heading"), default=None,
                    help="obs root-ori anchoring; default: tokenizer ONNX "
                         "'ori_mode' metadata (REFUSES to start if neither)")
    ap.add_argument("--root-heading-band-deg", type=float, default=25.0,
                    help="root-heading guard band (deg, 0=off): after the "
                         "engage yaw anchor, garment-frame yaw leaving this "
                         "band for --root-heading-hold-s releases the chain "
                         "(11-14 incident: yaw drift/turn -> phantom turn "
                         "command -> constant-direction stepping)")
    ap.add_argument("--root-heading-hold-s", type=float, default=2.0)
    ap.add_argument("--no-stand-plausibility", action="store_true",
                    help="skip the first-engaged-window stand topology check")
    ap.add_argument("--record-dir", default=None,
                    help="rolling npz record of raw intents + robot base "
                         "quats for offline replay/forensics "
                         f"(default {_PC2_PREFIX}/log/intent_record; "
                         "empty string disables)")
    ap.add_argument("--span-dir", default=None,
                    help="where engaged-span npz dumps go "
                         f"(default {_PC2_PREFIX}/log/intent_spans; "
                         "empty string disables)")
    args = ap.parse_args()

    root_level = (args.operator_root_level
                  if args.operator_root_level is not None
                  else os.environ.get("TOKEN_SVC_OPERATOR_ROOT_LEVEL", "")).strip()
    lean_cap = (args.operator_lean_cap if args.operator_lean_cap is not None
                else float(os.environ.get("TOKEN_SVC_OPERATOR_LEAN_CAP", "0") or 0))
    reach_cap = (args.operator_reach_cap if args.operator_reach_cap is not None
                 else float(os.environ.get("TOKEN_SVC_OPERATOR_REACH_CAP", "0") or 0))
    span_dir = (args.span_dir if args.span_dir is not None
                else os.path.join(_PC2_PREFIX, "log/intent_spans"))

    import zmq
    ctx = zmq.Context.instance()

    # tokenizer session (refuse to guess ori_mode — upstream FATAL path)
    import onnxruntime as ort
    sess = ort.InferenceSession(args.tokenizer, providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    ori_mode = args.ori_mode or meta.get("ori_mode")
    if ori_mode not in ("full", "heading"):
        print(f"[token-svc] FATAL: tokenizer carries no 'ori_mode' metadata "
              f"and no --ori-mode given (got {ori_mode!r}).", flush=True)
        return 2
    io = {i.name: i for i in sess.get_inputs()}
    out_name = "motion_token" if "motion_token" in [o.name for o in sess.get_outputs()] \
        else sess.get_outputs()[0].name
    if "smpl_obs" not in io:
        print(f"[token-svc] FATAL: tokenizer input is 'smpl_obs', got "
              f"{list(io)}", flush=True)
        return 2

    def tokenizer_run(obs: np.ndarray) -> np.ndarray:
        return sess.run(["motion_token" if out_name == "motion_token" else out_name],
                        {"smpl_obs": obs})[0].ravel()

    core = TokenServiceCore(tokenizer_run, ori_mode=ori_mode,
                            operator_root_level=root_level,
                            lean_cap_deg=lean_cap, reach_cap_m=reach_cap,
                            planner_cmd=int(args.planner_cmd_port) > 0,
                            now=time.monotonic(),
                            root_heading_band_deg=args.root_heading_band_deg,
                            root_heading_hold_s=args.root_heading_hold_s,
                            stand_plausibility=not args.no_stand_plausibility)
    print(f"[token-svc] tokenizer {args.tokenizer} (ori_mode={ori_mode})", flush=True)
    print(f"[token-svc] operator-root leveling: {root_level or 'off'}", flush=True)
    print(f"[token-svc] operator pose caps: lean {lean_cap:.1f} deg, "
          f"reach {reach_cap:.2f} m (0 = off)", flush=True)

    intent = ctx.socket(zmq.SUB)
    # Garment path: the RECEIVER is the publisher and BINDs :5573; this
    # service CONNECTS as the consumer (classic PUB bind / SUB connect).
    intent.connect(f"tcp://127.0.0.1:{args.intent_port}")
    intent.setsockopt_string(zmq.SUBSCRIBE, "pico_intent")
    intent.setsockopt(zmq.RCVTIMEO, 0)   # non-blocking drain: at 50 Hz
    tokens = ctx.socket(zmq.PUB)
    tokens.bind(f"tcp://*:{args.token_port}")
    debug = ctx.socket(zmq.SUB)
    debug.connect(f"tcp://{args.debug_host}:{args.debug_port}")
    debug.setsockopt_string(zmq.SUBSCRIBE, "x2_debug")
    debug.setsockopt(zmq.RCVTIMEO, 0)
    wb_cmd = None
    if int(args.planner_cmd_port) > 0:
        wb_cmd = ctx.socket(zmq.PUB)
        wb_cmd.setsockopt(zmq.LINGER, 0)
        wb_cmd.connect(f"tcp://{args.planner_cmd_host}:{args.planner_cmd_port}")
        print(f"[token-svc] wb_state PUB -> planner_cmd "
              f"{args.planner_cmd_host}:{args.planner_cmd_port}", flush=True)
    import json as _json

    # ---- raw intent/robot-state recorder (offline replay / forensics) ----
    # 11-14 post-mortem: the run could not be replayed because nothing
    # recorded the input stream. Default ON under ~/gear-sonic/log/intent_record.
    rec_dir = (args.record_dir if args.record_dir is not None
               else os.path.join(_PC2_PREFIX, "log/intent_record"))
    rec = {"t": [], "sj": [], "hq": [], "eng": [],
           "bt": [], "bq": [], "n": 0}
    rec_t0 = time.monotonic()

    def rec_roll() -> None:
        if not rec_dir or not rec["t"]:
            return
        os.makedirs(rec_dir, exist_ok=True)
        fn = os.path.join(rec_dir,
                          f"rec_{int(rec['t'][0])}_{rec['n']:03d}.npz")
        np.savez_compressed(
            fn,
            t=np.asarray(rec["t"], np.float64),
            smpl_joints=np.asarray(rec["sj"], np.float32),
            human_quat=np.asarray(rec["hq"], np.float32),
            engaged=np.asarray(rec["eng"], np.float32),
            robot_t=np.asarray(rec["bt"], np.float64),
            base_quat=np.asarray(rec["bq"], np.float32))
        print(f"[token-svc] record: {len(rec['t'])} frames -> {fn}", flush=True)
        rec["n"] += 1
        for k in ("t", "sj", "hq", "eng", "bt", "bq"):
            rec[k].clear()

    print(f"[token-svc] intent SUB *:{args.intent_port}  token PUB "
          f"*:{args.token_port}  debug {args.debug_host}:{args.debug_port}",
          flush=True)
    print(f"[token-svc] guards: root-heading band {args.root_heading_band_deg:.0f}"
          f" deg @{args.root_heading_hold_s:.1f}s "
          f"{'ON' if args.root_heading_band_deg > 0 else 'OFF'}, "
          f"stand-plausibility {'ON' if not args.no_stand_plausibility else 'OFF'}, "
          f"record {'OFF' if not rec_dir else rec_dir}", flush=True)

    import json as _json

    def send_wb_state(flag: bool) -> None:
        if wb_cmd is not None:
            wb_cmd.send_multipart([
                b"planner_cmd",
                _json.dumps({"intent": "wb_state", "engaged": int(bool(flag)),
                             "source": "wb"}).encode("utf-8")])

    def send_wrist(w) -> None:
        if wb_cmd is not None:
            wb_cmd.send_multipart([
                b"planner_cmd",
                _json.dumps({"intent": "wrist_targets", "source": "wb",
                             "lp": float(w[0]), "lr": float(w[1]),
                             "rp": float(w[2]), "rr": float(w[3])}
                            ).encode("utf-8")])

    def save_span(span: dict) -> None:
        if not (span and span["t"]) or not span_dir:
            return
        os.makedirs(span_dir, exist_ok=True)
        fn = os.path.join(span_dir, f"span_{int(span['t'][0])}.npz")
        np.savez_compressed(
            fn,
            t=np.asarray(span["t"], np.float64),
            smpl_joints=np.asarray(span["sj"], np.float32),
            human_quat=np.asarray(span["hq"], np.float32),
            clock="pc2_monotonic (same as deploy csv t)")
        print(f"[token-svc] span capture: {len(span['t'])} frames -> {fn}",
              flush=True)

    n_tok = n_int = n_dbg = 0
    t_rep = time.monotonic()
    engaged_prev_published = None
    next_t = time.monotonic()
    while True:
        # drain intent
        while True:
            try:
                raw = intent.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                f = unpack_message(raw, expected_topic="pico_intent").fields
                now_i = time.monotonic()
                sj = np.asarray(f["smpl_joints"], np.float32)
                hq = np.asarray(f["human_quat"], np.float32).ravel()[:4]
                eg = float(np.asarray(f.get("engaged", [0.0])).ravel()[0])
                core.feed_intent(
                    now_i, sj, hq,
                    engaged=eg,
                    wrist_pr=(np.asarray(f["wrist_pr"], np.float32).ravel()[:4]
                              if "wrist_pr" in f else None))
                if rec_dir:
                    rec["t"].append(now_i)
                    rec["sj"].append(sj)
                    rec["hq"].append(hq)
                    rec["eng"].append(eg)
                n_int += 1
            except Exception as e:
                print(f"[token-svc] intent decode error: {e!r}", flush=True)
        # drain debug
        while True:
            try:
                raw = debug.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                f = unpack_message(raw, expected_topic="x2_debug").fields
                now_d = time.monotonic()
                bq = f.get("base_quat", f.get("base_quat_wxyz"))
                core.feed_debug(
                    now_d,
                    np.asarray(bq, np.float32).ravel()[:4],
                    body_q_mj=(np.asarray(f["body_q"], np.float32).ravel()
                               if f.get("body_q") is not None else None))
                if rec_dir and bq is not None:
                    rec["bt"].append(now_d)
                    rec["bq"].append(np.asarray(bq, np.float32).ravel()[:4])
                n_dbg += 1
            except Exception:
                pass
        rec_now = time.monotonic()
        if rec_dir and rec_now - rec_t0 >= 10.0:
            rec_roll()
            rec_t0 = rec_now

        now = time.monotonic()
        if now < next_t:
            time.sleep(min(next_t - now, 0.004))
            continue
        next_t += CONTROL_DT

        res = sync_and_tick(core, now)

        # engage transition logging + span flush mirrors upstream
        if core.engaged_prev != engaged_prev_published:
            if engaged_prev_published is not None:  # skip startup line
                print(f"[token-svc] {'ENGAGED' if res.engaged else 'released'} "
                      f"(intent_fresh={res.intent_fresh} "
                      f"state_fresh={res.state_fresh})", flush=True)
            if not res.engaged and engaged_prev_published:
                save_span(core.drain_span())
            engaged_prev_published = core.engaged_prev

        # 5 s heartbeat (operator flow confirmation)
        if now - t_rep >= 5.0:
            dt = now - t_rep
            print(f"[token-svc] in: intent {n_int / dt:.1f} Hz  "
                  f"robot-state {n_dbg / dt:.1f} Hz  |  out: tokens "
                  f"{n_tok / dt:.1f} Hz  |  "
                  f"{'ENGAGED' if res.engaged else 'disengaged'}"
                  f"{'' if res.intent_fresh else '  [NO INTENT STREAM]'}"
                  f"{'' if res.intent_alive else '  [FROZEN INTENT]'}"
                  f"{'' if res.state_fresh else '  [NO ROBOT STATE]'}",
                  flush=True)
            n_tok, n_int, n_dbg, t_rep = 0, 0, 0, now

        if res.published and res.msg is not None:
            tokens.send(res.msg)
            n_tok += 1
        if "wb_state_sent" in res.diag:
            send_wb_state(bool(res.diag["wb_state_sent"]))
        if res.wrist_targets is not None:
            send_wrist(res.wrist_targets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
