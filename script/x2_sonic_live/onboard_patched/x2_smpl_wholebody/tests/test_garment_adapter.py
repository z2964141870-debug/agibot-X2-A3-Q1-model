"""Offline tests for the garment -> pico_intent adapter (stage 2a).

No BLE, no LFP, no torch: frames are synthesized the way SmpleXConverter's
FK emits them — a name-keyed dict of (world-frame joint position, global
joint quat WXYZ) — with a canonical A-pose rigidly placed in the world.

Covers:
  1. root-local math: world-placed+rotated pose -> canonical local joints,
     human_quat = operator global orientation
  2. wire contract: field set/shapes match the token service decoder
  3. engage gate: no file -> never; file + moving wearer -> never;
     file + stationary wearer -> engages after --stationary-s; file
     removed -> disengages
  4. frame_index only counts published frames
  5. chain interop: adapter frames drive TokenServiceCore to ENGAGED with
     tokens flowing; turning the operator (root yaw) keeps the rel root
     rotation ~identity (heading anchored), and the local skeleton is
     invariant to world yaw by construction

Run:  python tests/test_garment_adapter.py
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))

from garment_intent_adapter import (  # noqa: E402
    _SMPL_FK_WORLD_ALIGN, _SMPL_TO_ZUP,
    GarmentIntentAdapter, joints_from_human_data, root_local_joints)
from token_service_core import TokenServiceCore, NUM_X2, _stand_plausible  # noqa: E402
from wire_format import unpack_message  # noqa: E402

PASS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    PASS.append((name, bool(cond)))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


# ---------------------------------------------------------------- fixtures
def a_pose() -> np.ndarray:
    """Canonical standing pose, pelvis at origin, facing +x (from the mock
    sender's rest template — plausibility suffices for math tests)."""
    from mock_pico_intent_sender import _REST
    return _REST.copy()


def human_data_world(pos_local: np.ndarray, root_yaw: float = 0.0,
                     root_pitch: float = 0.0, pelvis_world=(0.3, -0.2, 0.95),
                     source_frame: str = "canonical"):
    """Place a local pose in the world the way FK would:
    - source_frame='canonical': R_wire @ p + t, quats = R_wire
    - source_frame='smpl_fk': mirrors SmpleXConverter's T-rotated Y-up body FK."""
    R = Rot.from_euler("ZYX", [root_yaw, root_pitch, 0.0])  # yaw then pitch
    if source_frame == "smpl_fk":
        r_mat = _SMPL_FK_WORLD_ALIGN.T @ R.as_matrix() @ _SMPL_TO_ZUP
        pos_body = pos_local.astype(np.float64) @ _SMPL_TO_ZUP
        R_emit = Rot.from_matrix(r_mat)
    else:
        pos_body = pos_local.astype(np.float64)
        R_emit = R
    names = [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
        "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
        "neck", "left_collar", "right_collar", "head", "left_shoulder",
        "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
        "right_wrist", "left_hand", "right_hand",
    ]
    t = np.asarray(pelvis_world, np.float64)
    out = {}
    q_wxyz = np.array([*R_emit.as_quat()[3:4], *R_emit.as_quat()[:3]], np.float64)
    for i, n in enumerate(names):
        p = R_emit.apply(pos_body[i]) + t
        out[n] = (p.astype(np.float32), q_wxyz.astype(np.float32))
    return out


def wiggle(pos_local: np.ndarray, amp: float, t: float) -> np.ndarray:
    """Non-stationary, NON-periodic micro-motion (two incommensurate
    frequencies on different joints): a real wearer's IMU stream is never
    byte-identical, and a pure sinusoid repeats against the 25-frame
    FROZEN-INTENT window at certain phases."""
    p = pos_local.copy()
    p[21] = p[21] + np.array([0.0, 0.0, amp * math.sin(2 * math.pi * 1.3 * t)])
    p[20] = p[20] + np.array([0.0, amp * 0.6 * math.sin(2 * math.pi * 1.7 * t + 1.0), 0.0])
    return p


# ------------------------------------------------------------------ tests
def test_root_local_math():
    P = a_pose()
    hd = human_data_world(P, root_yaw=math.radians(120),
                          pelvis_world=(2.0, 5.0, 1.1))
    pos, quat = joints_from_human_data(hd)
    j_local, root_q = root_local_joints(pos, quat[0])
    check("root-local: pelvis at origin",
          np.allclose(j_local[0], 0.0, atol=1e-6))
    check("root-local: skeleton invariant to world placement/rotation",
          np.allclose(j_local, P, atol=1e-4))
    q_xyzw = [root_q[1], root_q[2], root_q[3], root_q[0]]
    yaw = Rot.from_quat(q_xyzw).as_euler("ZYX")[0]
    check("human_quat keeps operator world yaw (120 deg)",
          abs(np.degrees(yaw) - 120.0) < 1e-3)
    # pitch survives in the quat too (token service DIAG/leveling rely on it)
    hd2 = human_data_world(P, root_pitch=math.radians(15))
    _, q2 = joints_from_human_data(hd2)
    pitch = Rot.from_quat([q2[0][1], q2[0][2], q2[0][3], q2[0][0]]).as_euler("ZYX")[1]
    check("human_quat keeps operator pitch (15 deg)",
          abs(np.degrees(pitch) - 15.0) < 1e-3)


def test_wire_contract():
    P = a_pose()
    with tempfile.TemporaryDirectory() as td:
        engage = os.path.join(td, "engage")
        ad = GarmentIntentAdapter(engage_file=engage, stationary_s=0.2, source_frame="canonical")
        open(engage, "w").close()
        msg = None
        for i in range(30):
            hd = human_data_world(wiggle(P, 0.001, i * 0.02))
            r = ad.publish(hd, i * 0.02)
            if r is not None:
                msg = r
        check("gate: stationary wearer + file -> publishes", msg is not None)
        d = unpack_message(msg, expected_topic="pico_intent")
        want = {"smpl_joints", "human_quat", "engaged", "wrist_pr", "frame_index"}
        check("wire: exact pico_intent field set", set(d.fields) == want)
        check("wire: shapes", d.fields["smpl_joints"].shape == (72,)
              and d.fields["human_quat"].shape == (4,)
              and d.fields["engaged"].shape == (1,)
              and d.fields["wrist_pr"].shape == (4,))
        check("wire: engaged=1, wrist zeros",
              float(d.fields["engaged"][0]) == 1.0
              and np.allclose(d.fields["wrist_pr"], 0.0))


def test_engage_gate(tmp=None):
    P = a_pose()
    with tempfile.TemporaryDirectory() as td:
        engage = os.path.join(td, "engage")
        # fresh adapter per sub-case: production clocks are monotonic, so
        # the gate may assume ever-advancing t
        ad = GarmentIntentAdapter(engage_file=engage, stationary_s=0.5,
                                  stationary_speed_mps=0.2, source_frame="canonical")
        # no file: never publish even when still
        out = [ad.publish(human_data_world(P), t) for t in np.arange(0, 2.0, 0.02)]
        check("gate: no engage file -> nothing published", all(o is None for o in out))
        check("gate: reason no-engage-file", ad.last_reason == "no-engage-file")

        # file + continuous motion: still blocked
        ad = GarmentIntentAdapter(engage_file=engage, stationary_s=0.5,
                                  stationary_speed_mps=0.2, source_frame="canonical")
        open(engage, "w").close()
        out = [ad.publish(human_data_world(wiggle(P, 0.30, t)), t)
               for t in np.arange(0, 2.0, 0.02)]
        check("gate: moving wearer blocked", all(o is None for o in out))
        check("gate: reason wearer-moving", ad.last_reason == "wearer-moving")

        # file + still: publishes after the stationary window
        ad = GarmentIntentAdapter(engage_file=engage, stationary_s=0.5,
                                  stationary_speed_mps=0.2, source_frame="canonical")
        out = [ad.publish(human_data_world(P), t) for t in np.arange(0, 2.0, 0.02)]
        check("gate: engages after stationary window", any(o is not None for o in out))
        check("gate: reason ok-or-engaged (latch)",
              ad.last_reason in ("ok", "engaged"))

        # file removed: immediate disengage
        os.remove(engage)
        r = ad.publish(human_data_world(P), 10.0)
        check("gate: file removed -> disengage", r is None)


def test_frame_index_counts_published():
    P = a_pose()
    with tempfile.TemporaryDirectory() as td:
        engage = os.path.join(td, "engage")
        ad = GarmentIntentAdapter(engage_file=engage, stationary_s=0.1, source_frame="canonical")
        open(engage, "w").close()
        idx = []
        for i, t in enumerate(np.arange(0, 1.0, 0.02)):
            r = ad.publish(human_data_world(P), t)
            if r is not None:
                idx.append(unpack_message(r, expected_topic="pico_intent")
                           .fields["frame_index"][0])
        check("frame_index: strictly increasing over published frames",
              all(b > a for a, b in zip(idx, idx[1:])))


def test_chain_interop_with_token_service():
    """Adapter -> token service core: ENGAGED + tokens + yaw-turn invariance."""
    P = a_pose()
    with tempfile.TemporaryDirectory() as td:
        engage = os.path.join(td, "engage")
        open(engage, "w").close()
        ad = GarmentIntentAdapter(engage_file=engage, stationary_s=0.1)

        seen_obs = []

        def fake_tok(obs):
            seen_obs.append(np.asarray(obs, np.float32).copy())
            return np.ones(64, np.float32)

        core = TokenServiceCore(fake_tok, ori_mode="heading")
        base = np.array([1.0, 0, 0, 0], np.float32)
        body = np.zeros(NUM_X2, np.float32)
        published = 0
        yawed_published = 0
        t = 0.0
        for i in range(240):
            # breathing micro-motion (a real wearer is never byte-identical —
            # the FROZEN-INTENT guard would rightly block a perfectly static
            # human), operator faces 0 deg until i=120, then turns 90 deg
            yaw = math.radians(90) if i >= 120 else 0.0
            hd = human_data_world(wiggle(P, 0.004, t), root_yaw=yaw, source_frame="smpl_fk")
            t = i * 0.02
            msg = ad.publish(hd, t)
            if msg is not None:
                f = unpack_message(msg, expected_topic="pico_intent").fields
                core.feed_intent(t, f["smpl_joints"], f["human_quat"],
                                 float(f["engaged"][0]), f["wrist_pr"])
            core.feed_debug(t, base, body)
            r = core.tick()
            published += int(r.published)
            if r.published and i >= 120:
                yawed_published += 1
        check("chain: core reached ENGAGED and tokens flowed (>=80)",
              published >= 80)
        check("chain: tokenizer saw [1,840] obs", seen_obs
              and seen_obs[0].shape == (1, 840))
        # world yaw must not leak into the skeleton (root-local): compare
        # both a pre-turn and a post-turn obs against the canonical pose —
        # deviation must stay within the micro-motion amplitude; a yaw leak
        # would be orders of magnitude larger
        if len(seen_obs) > 130:
            P72 = P.reshape(72)
            early_dev = float(np.abs(seen_obs[5][0][:72] - P72).max())
            late_dev = float(np.abs(seen_obs[-1][0][:72] - P72).max())
            check("chain: skeleton stays root-local canonical across yaw turn",
                  early_dev < 0.03 and late_dev < 0.03,
                  f"dev early={early_dev:.4f} late={late_dev:.4f}")
        check("chain: tokens published after the turn too",
              yawed_published > 0)


def test_jittered_clock_still_engages():
    """Regression for the 2026-10-29 field failure: with jittered frame
    timestamps the sliding-window span check was unsatisfiable — the
    rate-adaptive frame count must engage."""
    P = a_pose()
    with tempfile.TemporaryDirectory() as td:
        engage = os.path.join(td, "engage")
        open(engage, "w").close()
        ad = GarmentIntentAdapter(engage_file=engage, stationary_s=0.5,
                                  stationary_speed_mps=0.2, source_frame="canonical")
        rng = np.random.default_rng(7)
        t = 0.0
        published = 0
        for i in range(60):  # ~2 s of jittered frames (28-35 Hz)
            t += 0.032 + float(rng.uniform(-0.006, 0.006))
            r = ad.publish(human_data_world(P), t)
            published += int(r is not None)
        check("jitter: engages despite uneven frame clock", published > 0)


def test_engage_latches_through_motion():
    """2026-10-29 field fix: once engaged, normal wearer motion must NOT
    disengage — only removing the engage file releases. Motion IS the
    whole-body command; only the engage moment is gated."""
    P = a_pose()
    with tempfile.TemporaryDirectory() as td:
        engage = os.path.join(td, "engage")
        ad = GarmentIntentAdapter(engage_file=engage, stationary_s=0.3,
                                  stationary_speed_mps=0.2, source_frame="canonical")
        open(engage, "w").close()
        # stand still -> engage
        for i in range(40):
            ad.publish(human_data_world(P), i * 0.02)
        check("latch: engaged after stationary window", ad.engaged)
        # wearer now moves freely (large motion) -> MUST keep publishing
        out = [ad.publish(human_data_world(wiggle(P, 0.30, i * 0.02)), 1.0 + i * 0.02)
               for i in range(40)]
        check("latch: motion while engaged keeps publishing",
              all(o is not None for o in out))
        check("latch: still engaged", ad.engaged)
        # file removed -> release
        os.remove(engage)
        r = ad.publish(human_data_world(P), 5.0)
        check("latch: file removed -> release", r is None and not ad.engaged)
        # re-engage requires stillness again
        open(engage, "w").close()
        out = [ad.publish(human_data_world(wiggle(P, 0.30, t)), 6.0 + t)
               for t in np.arange(0, 1.0, 0.02)]
        check("latch: re-engage re-gated by motion", all(o is None for o in out))


def test_smpl_fk_frame_transform():
    """Real SmpleXConverter FK puts root quat in T-world and joints in Y-up;
    source_frame='smpl_fk' (default) must recover SONIC Z-up + stand_plausible."""
    P = a_pose()
    hd = human_data_world(P, root_yaw=math.radians(35), root_pitch=math.radians(12),
                          pelvis_world=(1.0, -0.5, 0.9), source_frame="smpl_fk")
    pos, quat = joints_from_human_data(hd)
    # Without the fix (canonical), SMPL Y-up fails stand plausibility.
    j_bad, _ = root_local_joints(pos, quat[0], source_frame="canonical")
    ok_bad, _ = _stand_plausible(j_bad)
    check("smpl_fk regression: canonical frame on SMPL-FK fails stand check", not ok_bad)

    j_zup, q_wire = root_local_joints(pos, quat[0], source_frame="smpl_fk")
    ok_good, why = _stand_plausible(j_zup)
    check("smpl_fk fix: converted frame passes _stand_plausible", ok_good, why)
    check("smpl_fk fix: matches canonical local Z-up joints", np.allclose(j_zup, P, atol=1e-4))
    eul = Rot.from_quat([q_wire[1], q_wire[2], q_wire[3], q_wire[0]]).as_euler("ZYX", degrees=True)
    check("smpl_fk fix: preserves yaw (35 deg) and pitch (12 deg)",
          abs(eul[0] - 35.0) < 1e-2 and abs(eul[1] - 12.0) < 1e-2, f"eul={np.round(eul, 2)}")


def test_smplx_converter_fast_interop():
    """On X2 (where smartwear_v2 is installed), run the real SMPLXConverter fast path."""
    smartwear = Path("/home/agi/projects/smartwear_v2")
    if not (smartwear / "general_motion_retargeting" / "SmpleXConverter.py").is_file():
        print("  SKIP  real SMPLXConverter not present on this host")
        return
    if str(smartwear) not in sys.path:
        sys.path.insert(0, str(smartwear))
    from general_motion_retargeting.SmpleXConverter import SMPLXConverter

    conv = SMPLXConverter()
    angles_rest = np.zeros((24, 3), dtype=np.float32)
    trans = np.zeros(3, dtype=np.float32)
    hd_rest = conv.convert_axis_angle_to_smpl_bundle_fast(angles_rest, trans)["human_data"]
    pos_r, q_r = joints_from_human_data(hd_rest)
    j_rest, rq_rest = root_local_joints(pos_r, q_r[0], source_frame="smpl_fk")
    ok_r, why_r = _stand_plausible(j_rest)
    check("real SMPLXConverter rest: stand_plausible", ok_r, why_r)
    check("real SMPLXConverter rest: +y left hip, -y right hip, +z head, +x toes",
          j_rest[1, 1] > 0.04 and j_rest[2, 1] < -0.04 and j_rest[15, 2] > 0.4 and j_rest[10, 0] > j_rest[7, 0])
    eul = Rot.from_quat([rq_rest[1], rq_rest[2], rq_rest[3], rq_rest[0]]).as_euler("ZYX", degrees=True)
    check("real SMPLXConverter rest: upright human_quat (|pitch|,|roll| < 5 deg)",
          abs(eul[1]) < 5.0 and abs(eul[2]) < 5.0, f"eul={np.round(eul, 2)}")

    # Raise left shoulder (SMPL joint 16, T-pose arm points +X; +Z rotation lifts +X toward +Y up)
    angles_raise = angles_rest.copy()
    angles_raise[16] = np.array([0.0, 0.0, 0.8], dtype=np.float32)
    hd_up = conv.convert_axis_angle_to_smpl_bundle_fast(angles_raise, trans)["human_data"]
    with tempfile.TemporaryDirectory() as td:
        engage = os.path.join(td, "engage")
        open(engage, "w").close()
        ad = GarmentIntentAdapter(engage_file=engage, stationary_s=0.05)
        for k in range(5):
            ad.publish(hd_rest, k * 0.03)
        msg = ad.publish(hd_up, 0.20)
        check("real SMPLXConverter: GarmentIntentAdapter default publishes", msg is not None)
        if msg is not None:
            j_up = unpack_message(msg, expected_topic="pico_intent").fields["smpl_joints"].reshape(24, 3)
            check("real SMPLXConverter left raise: left wrist z rises above rest",
                  j_up[20, 2] > j_rest[20, 2] + 0.15 and abs(j_up[21, 2] - j_rest[21, 2]) < 0.05,
                  f"lw_z {j_rest[20,2]:.3f}->{j_up[20,2]:.3f}")


def main() -> int:
    tests = [
        test_root_local_math,
        test_smpl_fk_frame_transform,
        test_smplx_converter_fast_interop,
        test_wire_contract,
        test_engage_gate,
        test_frame_index_counts_published,
        test_chain_interop_with_token_service,
        test_jittered_clock_still_engages,
        test_engage_latches_through_motion,
    ]
    print(f"x2_smpl_wholebody garment-adapter tests — {len(tests)} cases")
    for t in tests:
        print(f"[{t.__name__}]")
        t()
    n_fail = sum(1 for _, ok in PASS if not ok)
    print(f"\n{len(PASS) - n_fail}/{len(PASS)} checks passed"
          + (f"  ({n_fail} FAILED)" if n_fail else ""))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
