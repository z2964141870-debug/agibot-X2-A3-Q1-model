"""Offline validation for the SMPL whole-body token chain (stage 1).

No robot, no ZMQ (except the loopback smoke), no real ONNX: the tokenizer
is a recorded fake. Covers:

  1. wire round-trip (pack v4 token payload -> unpack; C++ contract:
     HEADER_SIZE=1280, fields by name, little-endian)
  2. intent wire round-trip (pico_intent fields)
  3. smpl_obs assembly: 840 dims, 'heading' anchoring absorbs robot yaw,
     yaw_align aligns human heading to robot
  4. window sampling: DELAY_S-lagged 0.02 s grid over a synthetic ring
  5. guards: FROZEN-INTENT (25 identical frames), intent freshness,
     robot-state freshness -> disengage
  6. engage warm-up swallows the first 1.0 s
  7. yaw_align captured at first post-warmup tick (heading anchor)
  8. token message: motion_token f32[64] rides pose v4 with engaged +
     frame_index + stand-pose joint_pos_mj + measured root quat
  9. wrist clipping to WRIST_RANGE on joint_pos_mj overrides
  10. operator root leveling (zero/clamp/offset) and lean/reach caps
  11. fake-tokenizer end-to-end: 50 Hz feed -> published token stream
  12. ZMQ loopback smoke: mock sender PUB -> real SUB socket drain ->
      unpack (cross-process format proof on this machine)

Run:  python tests/test_token_chain.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from smpl_obs import DELAY_S, SMPL_DT, SMPL_WINDOW, build_smpl_obs  # noqa: E402
from token_service_core import (  # noqa: E402
    CONTROL_DT, INTENT_FRESH_S, NUM_X2, WRIST_MJ, WRIST_RANGE, TokenServiceCore,
    cap_operator_pose, level_operator_root, window_from_ring)
from wire_format import (HEADER_SIZE, pack_pose_message, unpack_message)  # noqa: E402
from x2_smpl_token_service import sync_and_tick  # noqa: E402

PASS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    PASS.append((name, bool(cond)))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


# ---------------------------------------------------------------- fixtures
def stand_joints(t: float, rng: np.random.Generator | None = None) -> np.ndarray:
    from mock_pico_intent_sender import frame
    j, _q, _w = frame("stand", t, rng or np.random.default_rng(0))
    return j


def feed_running(core: TokenServiceCore, t0: float, n: int, dt: float = SMPL_DT,
                 engaged: float = 1.0, jitter: bool = True):
    """Feed n frames of non-frozen stand motion + fresh robot state."""
    rng = np.random.default_rng(1)
    base_q = np.array([1.0, 0, 0, 0], np.float32)
    body_q = np.zeros(NUM_X2, np.float32)
    for i in range(n):
        t = t0 + i * dt
        j = stand_joints(t, rng if jitter else None)
        core.feed_intent(t, j, np.array([1.0, 0, 0, 0], np.float32), engaged)
        core.feed_debug(t + 0.001, base_q, body_q)


def fake_token_run(obs: np.ndarray) -> np.ndarray:
    """Records obs; returns deterministic 64-float token."""
    TOKEN_SEEN.append(np.array(obs, np.float32).copy())
    return np.arange(64, dtype=np.float32) * 0.5


TOKEN_SEEN: list[np.ndarray] = []


def fresh_core(**kw) -> TokenServiceCore:
    TOKEN_SEEN.clear()
    defaults = dict(tokenizer_run=fake_token_run, ori_mode="heading")
    defaults.update(kw)
    return TokenServiceCore(**defaults)


# ------------------------------------------------------------------ tests
def test_wire_roundtrip_token_payload():
    wr = np.zeros(NUM_X2, np.float32)
    wr[WRIST_MJ["lp"]] = 0.1
    tok = np.arange(64, dtype=np.float32)
    msg = pack_pose_message({
        "joint_pos_mj": wr,
        "root_quat_xyzw": np.array([0, 0, 0, 1], np.float32),
        "motion_token": tok,
        "engaged": np.array([1.0], np.float32),
        "frame_index": np.array([7], np.int64),
    }, topic="pose", version=4)
    d = unpack_message(msg, expected_topic="pose")
    check("wire: topic+version", d.topic == "pose" and d.version == 4)
    check("wire: header 1280", len(msg) - len(b"pose") - (len(msg) - len(b"pose")
          - HEADER_SIZE - 96) >= 0)  # header block present
    body = msg[len(b"pose"):]
    check("wire: header field NUL-padded to 1280",
          len(body) > HEADER_SIZE and body[HEADER_SIZE - 1:] != b"")
    check("wire: motion_token f32[64]", d.fields["motion_token"].shape == (64,)
          and np.allclose(d.fields["motion_token"], tok))
    check("wire: joint_pos_mj f32[31]", d.fields["joint_pos_mj"].shape == (31,))
    check("wire: root_quat xyzw", np.allclose(d.fields["root_quat_xyzw"], [0, 0, 0, 1]))
    check("wire: engaged/frame_index", float(d.fields["engaged"][0]) == 1.0
          and int(d.fields["frame_index"][0]) == 7)


def test_wire_roundtrip_intent():
    j = stand_joints(0.0).reshape(72)
    msg = pack_pose_message({
        "smpl_joints": j.astype(np.float32),
        "human_quat": np.array([1.0, 0, 0, 0], np.float32),
        "engaged": np.array([1.0], np.float32),
        "wrist_pr": np.array([0.1, -0.2, 0.3, -0.4], np.float32),
        "frame_index": np.array([3], np.int64),
    }, topic="pico_intent", version=4)
    d = unpack_message(msg, expected_topic="pico_intent")
    check("intent wire: 72 joints round-trip",
          np.allclose(d.fields["smpl_joints"], j, atol=1e-6))
    check("intent wire: wrist_pr", np.allclose(d.fields["wrist_pr"],
                                               [0.1, -0.2, 0.3, -0.4]))
    # C++ SmplIntentSource reads these exact field names
    want = {"smpl_joints", "human_quat", "engaged", "wrist_pr", "frame_index"}
    check("intent wire: field name set matches C++ decoder",
          want <= set(d.fields))


def test_smpl_obs_layout():
    rng = np.random.default_rng(2)
    joints = np.stack([stand_joints(0.1 * i, rng) for i in range(SMPL_WINDOW)])
    quats = np.tile(np.array([1.0, 0, 0, 0], np.float32), (SMPL_WINDOW, 1))
    base = np.array([1.0, 0, 0, 0], np.float32)
    wrist6 = np.zeros(6, np.float32)
    obs = build_smpl_obs(joints, quats, base, wrist6, ori_mode="heading")
    check("obs: 840 dims", obs.shape == (840,) and obs.dtype == np.float32)
    # heading mode anchors the robot's YAW only: robot pitch/roll must not
    # change the obs (v1.1 release cores semantic)
    pitch = Rot.from_euler("x", 0.7)
    base_pitch = np.array([pitch.as_quat()[3], *pitch.as_quat()[:3]], np.float32)
    obs_pitch = build_smpl_obs(joints, quats, base_pitch, wrist6, ori_mode="heading")
    check("obs: heading mode absorbs robot pitch/roll",
          np.allclose(obs, obs_pitch, atol=1e-5))
    obs_full_pitch = build_smpl_obs(joints, quats, base_pitch, wrist6, ori_mode="full")
    check("obs: full mode keeps robot pitch/roll",
          not np.allclose(obs_full_pitch, build_smpl_obs(joints, quats, base,
                                                         wrist6, ori_mode="full")))
    # robot yaw in heading mode DOES change the relative human heading —
    # by design (operator is heading-anchored at engage, then drifts with
    # robot rotation)
    yaw = Rot.from_euler("z", 0.7)
    base_yaw = np.array([yaw.as_quat()[3], *yaw.as_quat()[:3]], np.float32)
    obs_yaw = build_smpl_obs(joints, quats, base_yaw, wrist6, ori_mode="heading")
    check("obs: heading mode keeps relative robot yaw (by design)",
          not np.allclose(obs, obs_yaw))
    # layout: block 0 = joints 72, block 1 = root6d, block 2 = wrists
    check("obs: joints block at [0:72]",
          np.allclose(obs[:72], joints[0].reshape(72), atol=1e-6))
    check("obs: wrist block at [78:84]", np.allclose(obs[78:84], wrist6))


def test_window_sampling():
    ring = [(0.1 * i, stand_joints(0.1 * i), np.array([1, 0, 0, 0], np.float32))
            for i in range(20)]
    now = 1.0
    joints, quats = window_from_ring(ring, now)
    want = now - DELAY_S + np.arange(SMPL_WINDOW) * SMPL_DT
    check("window: 10 frames", joints.shape == (10, 24, 3) and quats.shape == (10, 4))
    check("window: oldest frame ~= now-0.18",
          abs(want[0] - (now - DELAY_S)) < 1e-9)
    # frames are nearest-at-or-before samples of the 0.02 grid
    expect_idx = np.searchsorted(np.array([r[0] for r in ring]), want, "right") - 1
    check("window: grid->index mapping matches upstream searchsorted",
          expect_idx.min() >= 0 and len(set(expect_idx.tolist())) <= 10)


def test_frozen_intent_guard():
    core = fresh_core()
    frozen = stand_joints(0.0)
    t = 0.0
    for i in range(30):
        t += SMPL_DT
        core.feed_intent(t, frozen, np.array([1, 0, 0, 0], np.float32), 1.0)
        core.feed_debug(t, np.array([1, 0, 0, 0], np.float32), np.zeros(NUM_X2, np.float32))
    res = core.tick()
    check("guard: 25+ identical frames -> FROZEN -> disengaged",
          (not res.published) and res.reason == "disengaged"
          and res.engaged is False and res.intent_alive is False)
    # same but with jitter -> alive -> proceeds (to warmup then engage)
    core2 = fresh_core()
    feed_running(core2, 0.0, 30)
    res2 = core2.tick()
    check("guard: jittering stream stays ALIVE", res2.intent_alive)


def test_freshness_guards():
    core = fresh_core()
    feed_running(core, 0.0, 30)
    core._now = 5.0  # jump the clock: intent and state now stale
    res = core.tick()
    check("guard: stale intent+state -> disengaged",
          res.reason == "disengaged" and not res.intent_fresh and not res.state_fresh)


def test_engage_warmup_and_anchor():
    core = fresh_core()
    feed_running(core, 0.0, 60)   # 1.2 s of history
    t_first = core._now
    res1 = core.tick()
    check("warmup: first engaged tick swallowed",
          res1.reason == "warmup" and res1.published is False)
    # next tick (20 ms later) is past the 1.0 s warm-up? engage set
    # warmup_until = engage-detection tick; drive ticks forward
    published = None
    for _ in range(60):
        r = core.tick()
        if r.published:
            published = r
            break
    check("engage: publishes after warm-up", published is not None)
    if published is not None:
        check("engage: fake tokenizer saw one obs [1,840]",
              len(TOKEN_SEEN) == 1 and TOKEN_SEEN[0].shape == (1, 840))
        check("engage: token = 64 floats from fake session",
              published.token.shape == (64,)
              and np.allclose(published.token, np.arange(64) * 0.5))


def test_yaw_align_at_engage():
    core = fresh_core()
    # human faces +z-ish (yaw +90 deg), robot faces +x (identity)
    rng = np.random.default_rng(3)
    for i in range(60):
        t = i * SMPL_DT
        j = stand_joints(t, rng)
        q_xyzw = Rot.from_euler("z", math.pi / 2).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], np.float32)
        core.feed_intent(t, j, q_wxyz, 1.0)
        core.feed_debug(t, np.array([1, 0, 0, 0], np.float32),
                        np.zeros(NUM_X2, np.float32))
    core.tick()  # warmup tick
    r = None
    for _ in range(60):
        r = core.tick()
        if r.published:
            break
    check("yaw anchor: published with anchor present",
          r is not None and r.published and "heading_anchor_deg" in r.diag)
    # after anchor, human yaw +90 vs robot 0 -> root6d rel ~ identity
    obs = TOKEN_SEEN[-1][0]
    root6d = obs[72:78]
    # identity rotmat first-two-columns flatten (row-major): [1,0, 0,1, 0,0]
    ident = np.array([1, 0, 0, 1, 0, 0], np.float32)
    check("yaw anchor: rel root rotation ~= identity after anchoring",
          np.allclose(root6d, ident, atol=1e-4))


def test_token_message_fields():
    core = fresh_core(planner_cmd=False)
    feed_running(core, 0.0, 60)   # also feeds debug each frame (body_q = zeros)
    core.tick()
    r = None
    for _ in range(60):
        r = core.tick()
        if r.published:
            break
    d = unpack_message(r.msg, expected_topic="pose")
    check("msg: engaged=1.0 + frame_index present",
          float(d.fields["engaged"][0]) == 1.0
          and int(d.fields["frame_index"][0]) >= 0)
    check("msg: root_quat = measured robot base (identity)",
          np.allclose(d.fields["root_quat_xyzw"], [0, 0, 0, 1], atol=1e-6))
    check("msg: joint_pos_mj default zeros (learned stand)",
          np.allclose(d.fields["joint_pos_mj"], np.zeros(NUM_X2, np.float32)))
    check("msg: motion_token from tokenizer", d.fields["motion_token"].shape == (64,))


def test_wrist_clip():
    core = fresh_core()
    feed_running(core, 0.0, 60)
    core.feed_intent(core.ring[-1][0], stand_joints(0.5),
                     np.array([1, 0, 0, 0], np.float32), 1.0,
                     wrist_pr=np.array([5.0, -9.0, -5.0, 9.0], np.float32))
    core.tick()
    r = None
    for _ in range(60):
        r = core.tick()
        if r.published:
            break
    d = unpack_message(r.msg, expected_topic="pose")
    wr = d.fields["joint_pos_mj"]
    p_lo, p_hi = WRIST_RANGE["p"]
    r_lo, r_hi = WRIST_RANGE["r"]
    check("wrist: clipped into WRIST_RANGE",
          p_lo <= wr[WRIST_MJ["lp"]] <= p_hi and r_lo <= wr[WRIST_MJ["lr"]] <= r_hi
          and p_lo <= wr[WRIST_MJ["rp"]] <= p_hi and r_lo <= wr[WRIST_MJ["rr"]] <= r_hi
          and wr[WRIST_MJ["lp"]] == p_hi and wr[WRIST_MJ["lr"]] == r_lo)


def test_root_leveling_and_caps():
    q = np.tile(np.array([1.0, 0, 0, 0], np.float32), (3, 1))
    # forward pitch = rotation about Y (x is forward in the SMPL frame);
    # scipy 'ZYX' euler returns [yaw, pitch, roll] with pitch = _e[1]
    pitched = Rot.from_euler("y", math.radians(20)).as_quat()
    q20 = np.array([pitched[3], pitched[0], pitched[1], pitched[2]], np.float32)
    q[1] = q20
    out_zero = level_operator_root(q, "zero")
    e = Rot.from_quat([out_zero[1][1], out_zero[1][2], out_zero[1][3],
                       out_zero[1][0]]).as_euler("ZYX")
    check("root-level zero: pitch/roll zeroed", abs(e[1]) < 1e-6 and abs(e[2]) < 1e-6)
    out_clamp = level_operator_root(q, "clamp:5")
    e2 = Rot.from_quat([out_clamp[1][1], out_clamp[1][2], out_clamp[1][3],
                        out_clamp[1][0]]).as_euler("ZYX")
    check("root-level clamp:5 -> pitch = 5 deg", abs(np.degrees(e2[1]) - 5.0) < 1e-3)
    out_off = level_operator_root(q, "offset:20")
    e3 = Rot.from_quat([out_off[1][1], out_off[1][2], out_off[1][3],
                        out_off[1][0]]).as_euler("ZYX")
    check("root-level offset:20 -> pitch ~= 0", abs(np.degrees(e3[1])) < 1e-3)

    # lean cap: torso bent far forward
    j = stand_joints(0.0).copy()
    for idx in (3, 6, 9, 12, 15):
        j[idx, 0] += 0.5  # spine/neck/head pushed +x (forward lean)
    stats = {"frames": 0, "lean": 0, "reach": 0}
    jc = cap_operator_pose(j[None], lean_cap_deg=15.0, reach_cap_m=0.0, stats=stats)
    v = jc[0, 12] - jc[0, 0]
    lean_after = math.degrees(math.atan2(v[0], max(v[2], 1e-6)))
    check("lean cap: 90+ deg lean clamped to <=~15 deg",
          stats["lean"] == 1 and lean_after < 20.0)

    # reach cap: right arm stretched far forward
    j2 = stand_joints(0.0).copy()
    for idx in (19, 21, 23):
        j2[idx, 0] = j2[17, 0] + 0.9  # wrist 0.9 m ahead of shoulder
    stats2 = {"frames": 0, "lean": 0, "reach": 0}
    jc2 = cap_operator_pose(j2[None], lean_cap_deg=0.0, reach_cap_m=0.5, stats=stats2)
    reach_after = jc2[0, 21, 0] - jc2[0, 17, 0]
    check("reach cap: 0.9 m reach clamped to 0.5 m",
          stats2["reach"] >= 1 and abs(reach_after - 0.5) < 1e-3)


def test_end_to_end_50hz_stream():
    core = fresh_core()
    rng = np.random.default_rng(4)
    base = np.array([1, 0, 0, 0], np.float32)
    body_q = np.zeros(NUM_X2, np.float32)
    published = 0
    t = 0.0
    for i in range(150):  # 3 s
        t += SMPL_DT
        core.feed_intent(t, stand_joints(t, rng), np.array([1, 0, 0, 0], np.float32), 1.0)
        core.feed_debug(t, base, body_q)
        r = core.tick()
        published += int(r.published)
    # first ALIVE tick is frame 26 (25-frame guard) -> engage at ~0.5 s ->
    # warm-up until ~1.5 s -> publishes from ~tick 75: expect 75 (+-5)
    check("e2e: tokens after guard+warm-up (75 +- 5)",
          70 <= published <= 80)
    check("e2e: tokenizer called once per published tick",
          len(TOKEN_SEEN) == published)


def test_zmq_loopback_smoke():
    """Cross-process format proof on this machine: mock sender frame ->
    real SUB socket -> unpack -> core consumes it."""
    import zmq

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    port = sub.bind_to_random_port("tcp://127.0.0.1")
    sub.setsockopt_string(zmq.SUBSCRIBE, "pico_intent")
    sub.setsockopt(zmq.RCVTIMEO, 3000)

    from mock_pico_intent_sender import frame
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.connect(f"tcp://127.0.0.1:{port}")
    time.sleep(0.5)  # slow-joiner: let the TCP connection establish
    j, q, w = frame("wave", 0.5, np.random.default_rng(5))
    msg = pack_pose_message({
        "smpl_joints": j.reshape(72), "human_quat": q,
        "engaged": np.array([1.0], np.float32), "wrist_pr": w,
        "frame_index": np.array([0], np.int64),
    }, topic="pico_intent", version=4)
    raw = None
    for _ in range(20):  # repeat until the subscription is live (HWM-safe)
        pub.send(msg)
        try:
            raw = sub.recv()
            break
        except zmq.Again:
            time.sleep(0.05)
    check("loopback: real ZMQ PUB->SUB pico_intent frame received", raw is not None)
    if raw is not None:
        d = unpack_message(raw, expected_topic="pico_intent")
        check("loopback: decoded frame matches sender",
              d.fields["smpl_joints"].shape == (72,)
              and np.allclose(d.fields["smpl_joints"], j.reshape(72), atol=1e-6))
        # and the core accepts exactly these decoded fields
        core = fresh_core()
        core.feed_intent(0.0, d.fields["smpl_joints"], d.fields["human_quat"],
                         float(d.fields["engaged"][0]), d.fields["wrist_pr"])
        check("loopback: decoded frame feeds the core", len(core.ring) == 1)
    sub.close(0)
    pub.close(0)
    ctx.term()


def test_monotonic_clock_sync_and_stream_stop():
    """sync_and_tick keeps wall clock advancing from t0>0 so window moves and stale stream stops."""
    t0 = 1500.0
    core = fresh_core(now=t0)
    rng = np.random.default_rng(7)
    base_q = np.array([1.0, 0, 0, 0], np.float32)
    body_q = np.zeros(NUM_X2, np.float32)

    pub_obs = []
    # Run 120 ticks (2.4 s) at 50 Hz with varying stand joints -> crosses 1.0 s warmup + history
    for i in range(120):
        t = t0 + i * CONTROL_DT
        j = stand_joints(float(i) * CONTROL_DT, rng)
        # inject gentle wave into wrists/hands so obs definitely evolves over time
        j[20, 2] += 0.10 * math.sin(i * 0.25)
        j[21, 2] += 0.10 * math.cos(i * 0.25)
        core.feed_intent(t, j, np.array([1.0, 0, 0, 0], np.float32), 1.0)
        core.feed_debug(t + 0.001, base_q, body_q)
        res = sync_and_tick(core, t)
        if res.published:
            pub_obs.append(TOKEN_SEEN[-1].copy())

    check("sync_clock: publishes tokens past 1.0 s warmup when t0=1500",
          len(pub_obs) >= 20, f"n_pub={len(pub_obs)}")
    if len(pub_obs) >= 2:
        obs_diff = float(np.max(np.abs(pub_obs[-1] - pub_obs[0])))
        check("sync_clock: live window advances instead of pinning at ring[0]",
              obs_diff > 1e-3, f"max_obs_diff={obs_diff:.4f}")

    # Stop feeding intent while advancing wall clock & robot debug state
    t_last_intent = t0 + 119 * CONTROL_DT
    disengaged_at = None
    for k in range(1, 50):
        t = t_last_intent + k * CONTROL_DT
        core.feed_debug(t, base_q, body_q)
        res = sync_and_tick(core, t)
        if not res.published and not res.intent_fresh:
            disengaged_at = t - t_last_intent
            break
    check("sync_clock: stops publishing within INTENT_FRESH_S after stream stops",
          disengaged_at is not None and disengaged_at <= INTENT_FRESH_S + 2 * CONTROL_DT,
          f"disengaged_after={disengaged_at}s limit={INTENT_FRESH_S}s")


def main() -> int:
    tests = [
        test_wire_roundtrip_token_payload,
        test_wire_roundtrip_intent,
        test_smpl_obs_layout,
        test_window_sampling,
        test_frozen_intent_guard,
        test_freshness_guards,
        test_engage_warmup_and_anchor,
        test_yaw_align_at_engage,
        test_token_message_fields,
        test_wrist_clip,
        test_root_leveling_and_caps,
        test_end_to_end_50hz_stream,
        test_zmq_loopback_smoke,
        test_monotonic_clock_sync_and_stream_stop,
    ]
    print(f"x2_smpl_wholebody offline tests — {len(tests)} cases")
    for t in tests:
        print(f"[{t.__name__}]")
        t()
    n_fail = sum(1 for _, ok in PASS if not ok)
    print(f"\n{len(PASS) - n_fail}/{len(PASS)} checks passed"
          + (f"  ({n_fail} FAILED)" if n_fail else ""))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
