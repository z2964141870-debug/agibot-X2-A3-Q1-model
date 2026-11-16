"""Garment -> pico_intent adapter: FGP SMPL output to the token chain.

Bridges the smartwear (FGP) per-frame output to the exact wire contract
the token service consumes (topic ``pico_intent``; see mock_pico_intent_sender
and pc2_pico_token_service). Semantics follow the upstream Pico manager
(pico_manager_thread_server.py::process_smpl_joints at c0036a4):

  wire smpl_joints  = R_root^-1 . (joints_world - pelvis)   (24x3, root
                      frame: pelvis at origin, root rotation removed from
                      the skeleton -> operator heading lives ONLY in
                      human_quat, where the token service's yaw anchor
                      needs it)
  wire human_quat   = operator global root orientation, WXYZ
  wire engaged      = engage file present AND wearer stationary for
                      --stationary-s (mirrors the 2026-08 garment proxy's
                      STANDSTILL_READY rule: never engage into a moving
                      wearer; motion DURING whole-body is the point, only
                      the engage moment is gated)
  wire wrist_pr     = zeros (no hand controllers on the garment; the
                      token policy drives the wrists)

No resampling: the token service samples its 10-frame window on its own
0.02 s grid (nearest-at-or-before), so any source cadence <= 50 Hz works.
"""
from __future__ import annotations

import os
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from wire_format import pack_pose_message

# Motion slower than this (m/s, max over all 24 joints) counts as
# "stationary" for the engage gate.
STATIONARY_SPEED_THRESH_MPS = 0.20

# SmpleXConverter applies T = Rx(+90 deg) on the left of root_orient, leaving
# the skeleton in SMPL body axes (x=left, y=up, z=fwd). Wire protocol expects
# SONIC Z-up (x=fwd, y=left, z=up): M = Rz(+90 deg) @ T maps SMPL body -> Z-up.
_SMPL_FK_T = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
_SMPL_TO_ZUP = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
_SMPL_FK_WORLD_ALIGN = _SMPL_TO_ZUP @ _SMPL_FK_T.T  # = Rz(+90 deg)


def joints_from_human_data(human_data: dict) -> tuple[np.ndarray, np.ndarray]:
    """{(name): (pos(3,), quat_wxyz(4,))} -> (pos24x3, quat_wxyz24x4)
    in SMPL joint order. Raises on missing joints."""
    names = [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
        "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
        "neck", "left_collar", "right_collar", "head", "left_shoulder",
        "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
        "right_wrist", "left_hand", "right_hand",
    ]
    pos = np.empty((24, 3), np.float32)
    quat = np.empty((24, 4), np.float32)
    for i, n in enumerate(names):
        if n not in human_data:
            raise KeyError(f"human_data missing joint {n!r}")
        p, q = human_data[n]
        pos[i] = np.asarray(p, np.float32).reshape(3)
        quat[i] = np.asarray(q, np.float32).reshape(4)
    return pos, quat


def root_local_joints(pos24x3: np.ndarray, root_quat_wxyz: np.ndarray,
                      source_frame: str = "canonical",
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Pico process_smpl_joints semantics on world-frame input:
    (R_root^-1 (pos - pelvis), root_quat_wxyz) for canonical Z-up FK, or
    convert SmpleXConverter's T-rotated Y-up body frame into SONIC Z-up
    (x=fwd, y=left, z=up) when source_frame='smpl_fk'."""
    if source_frame not in ("canonical", "smpl_fk"):
        raise ValueError(f"source_frame must be 'canonical' or 'smpl_fk', got {source_frame!r}")
    p = np.asarray(pos24x3, np.float32)
    q = np.asarray(root_quat_wxyz, np.float32).reshape(4)
    r = Rot.from_quat([q[1], q[2], q[3], q[0]])
    local = r.inv().apply((p - p[0:1]).astype(np.float64))
    if source_frame == "canonical":
        return local.astype(np.float32), q
    local_zup = local @ _SMPL_TO_ZUP.T
    r_wire = _SMPL_FK_WORLD_ALIGN @ r.as_matrix() @ _SMPL_TO_ZUP.T
    qx = Rot.from_matrix(r_wire).as_quat()
    q_wire = np.array([qx[3], qx[0], qx[1], qx[2]], dtype=np.float32)
    return local_zup.astype(np.float32), q_wire


class GarmentIntentAdapter:
    """Per-frame FGP -> pico_intent converter. Not thread-safe."""

    def __init__(self,
                 engage_file: str | None = None,
                 stationary_s: float = 2.0,
                 stationary_speed_mps: float = STATIONARY_SPEED_THRESH_MPS,
                 source_frame: str = "smpl_fk"):
        if source_frame not in ("canonical", "smpl_fk"):
            raise ValueError(f"source_frame must be 'canonical' or 'smpl_fk', got {source_frame!r}")
        self.engage_file = engage_file
        self.stationary_s = float(stationary_s)
        self.stationary_v = float(stationary_speed_mps)
        self.source_frame = source_frame
        self.history: deque = deque()          # (t, joints_local) for the gate
        self.engaged = False
        self.frame_index = 0
        self.last_reason = "init"

    # ---- engage gate ----------------------------------------------------
    def _engage_file_present(self) -> bool:
        return bool(self.engage_file) and os.path.exists(self.engage_file)

    def _stationary(self, t: float, joints_local: np.ndarray) -> bool:
        ok, reason = self._stationary_detail(t, joints_local)
        return ok

    def _stationary_detail(self, t: float, joints_local: np.ndarray) -> tuple[bool, str]:
        self.history.append((t, joints_local))
        while self.history and t - self.history[0][0] > max(self.stationary_s, 1.0) + 1.0:
            self.history.popleft()
        # Rate-adaptive frame counting (2026-10-29): a span-vs-threshold check
        # is UNSATISFIABLE under jitter — the sliding window's oldest frame is
        # always (threshold - dt, threshold] old, so "span >= threshold" almost
        # never fires. Count frames instead: need = ceil(stationary_s / est_dt).
        if len(self.history) < 2:
            return False, "warming-window(2 frames)"
        est_dt = (self.history[-1][0] - self.history[0][0]) / (len(self.history) - 1)
        need = max(3, int(round(self.stationary_s / max(est_dt, 1e-3))) + 1)
        if len(self.history) < need:
            return False, f"warming-window({len(self.history)}/{need} frames)"
        window = list(self.history)[-need:]
        worst = 0.0
        for (ta, ja), (tb, jb) in zip(window, window[1:]):
            dt = max(tb - ta, 1e-3)
            speed = float(np.abs(jb - ja).max() / dt)
            worst = max(worst, speed)
            if speed > self.stationary_v:
                if int(t) != int(t - 0.05):  # ~1 Hz diagnostic
                    print(f"[pico-adapter] gate: moving {speed:.3f} m/s "
                          f"(thresh {self.stationary_v})", flush=True)
                return False, f"moving({speed:.3f}m/s)"
        return True, f"ok({worst:.3f}m/s)"

    def engage_allowed(self, t: float, joints_local: np.ndarray) -> tuple[bool, str]:
        if not self._engage_file_present():
            return False, "no-engage-file"
        if not self._stationary(t, joints_local):
            return False, "wearer-moving"
        return True, "ok"

    # ---- per-frame -------------------------------------------------------
    def publish(self, human_data: dict, t: float) -> bytes | None:
        """One FGP frame (name-keyed FK dict) -> packed pico_intent bytes,
        or None while disengaged. t = monotonic seconds (same clock across
        frames; feeds the stationary gate).

        Engage LATCH semantics (2026-10-29): the stationary gate guards ONLY
        the disengaged->engaged transition. Once engaged, the wearer moves
        freely — that motion IS the whole-body command — and only the engage
        file disappearing releases."""
        pos, quat = joints_from_human_data(human_data)
        j_local, root_q = root_local_joints(pos, quat[0], source_frame=self.source_frame)

        if self.engaged:
            allowed = self._engage_file_present()
            reason = "engaged" if allowed else "released"
            if not allowed:
                self.engaged = False
                self.last_reason = reason
                self.history.clear()   # re-arm the stationary gate
                return None
        else:
            allowed, reason = self.engage_allowed(t, j_local)
            self.engaged = allowed
            if not allowed:
                self.last_reason = reason
                return None

        self.last_reason = reason
        msg = pack_pose_message({
            "smpl_joints": j_local.reshape(72),
            "human_quat": root_q.astype(np.float32),
            "engaged": np.array([1.0], np.float32),
            "wrist_pr": np.zeros(4, np.float32),
            "frame_index": np.array([self.frame_index], np.int64),
        }, topic="pico_intent", version=4)
        self.frame_index += 1
        return msg
