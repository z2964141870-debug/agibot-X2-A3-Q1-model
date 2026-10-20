"""P1.5a pilot driver (v9, parameterized): preview SMPL-H GT -> A3 joint angles
via vendor ChainA3Retargeter. argv: [session] [limit] [out_base]. Zero vendor writes."""
import json, sys, types
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

OPS = Path("/opt/data/private/yuanqi_ops")
ROOT = Path("/opt/data/private/hechengzhu/ZHY")
sys.path.insert(0, str(ROOT))

SESSION = sys.argv[1] if len(sys.argv) > 1 else "session_20260901_144355"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 0
BASE_ARG = Path(sys.argv[3]) if len(sys.argv) > 3 else None
OUT = ((BASE_ARG / SESSION) if BASE_ARG else
       OPS / "data/experiments/p15a_preview_retarget_pilot_20261020" / SESSION)
H5 = OPS / "data/humanplus_preview" / SESSION / "annotation.hdf5"
OUT.mkdir(parents=True, exist_ok=True)

def heading_rotation(matrix):
    m = np.asarray(matrix, dtype=float)[:3, :3]
    axis = m[:3, 0].copy(); axis[2] = 0.0
    if np.linalg.norm(axis) < 1e-9:
        axis = m[:3, 1].copy(); axis[2] = 0.0
    n = np.linalg.norm(axis)
    x = axis / max(n, 1e-12)
    y = np.cross([0.0, 0.0, 1.0], x)
    return np.column_stack([x, y, [0.0, 0.0, 1.0]])

class BodyReference:
    def __init__(self, times, names, positions, orientations, calibration_positions,
                 calibration_orientations, names_check=None, label=""):
        if names_check is not None and list(names_check) != list(names):
            raise ValueError("BodyReference name mismatch")
        self.times = np.asarray(times, dtype=float)
        self.names = list(names)
        self.positions = np.asarray(positions, dtype=float)
        self.orientations = np.asarray(orientations, dtype=float)
        self.calibration_positions = np.asarray(calibration_positions, dtype=float)
        self.calibration_orientations = np.asarray(calibration_orientations, dtype=float)
        self.label = label
    @classmethod
    def load(cls, path, names):
        with np.load(path, allow_pickle=False) as d:
            return cls(d["times"], d["names"].tolist(), d["positions"], d["orientations"],
                       d["calibration_positions"], d["calibration_orientations"], names,
                       str(d["label"]))

shim_q1 = types.ModuleType("q1_retarget")
shim_q1.heading_rotation = heading_rotation
shim_q1.BodyReference = BodyReference
sys.modules["q1_retarget"] = shim_q1
shim_ego = types.ModuleType("egolocate_input")
def _unused(*a, **k):
    raise NotImplementedError("shim: not used in P1.5a")
shim_ego.convert_bundle = _unused
shim_ego.save_test_only = _unused
sys.modules["egolocate_input"] = shim_ego

from script.a3.mocap_adapter import A3Retargeter
from script.a3.smpl_chain_retarget import ChainA3Retargeter

SMPL_IDX = {"pelvis": 0, "spine3": 9, "neck": 12,
            "left_hip": 1, "left_knee": 4, "left_ankle": 7, "left_toe": 10,
            "right_hip": 2, "right_knee": 5, "right_ankle": 8, "right_toe": 11,
            "left_shoulder": 16, "left_elbow": 18, "left_wrist": 20,
            "right_shoulder": 17, "right_elbow": 19, "right_wrist": 21}
BONE_CHILD = {"left_hip": "left_knee", "right_hip": "right_knee",
              "left_knee": "left_ankle", "right_knee": "right_ankle",
              "left_ankle": "left_toe", "right_ankle": "right_toe",
              "left_shoulder": "left_elbow", "right_shoulder": "right_elbow",
              "left_elbow": "left_wrist", "right_elbow": "right_wrist",
              "left_wrist": None, "right_wrist": None, "spine3": None}
REST_FWD = np.array([1.0, 0.0, 0.0])

def align_many(d_rest, d_obs):
    axis = np.cross(d_rest, d_obs)
    s = np.linalg.norm(axis, axis=-1)
    c = np.clip(d_obs @ np.asarray(d_rest), -1.0, 1.0)
    ang = np.arctan2(s, c)
    ok = s > 1e-6
    rotvec = np.zeros((len(ang), 3))
    rotvec[ok] = (axis[ok] / s[ok][:, None]) * ang[ok][:, None]
    return Rotation.from_rotvec(rotvec).as_matrix()

import h5py
with h5py.File(H5, "r") as f:
    kp = f["body_motion/body_keypoints"][:]
    rootm = f["body_motion/T_mocapworld_root"][:]
T = len(kp)
used = LIMIT if LIMIT and LIMIT < T else T
kp = kp[:used].astype(float)
root_rot = rootm[:used, :3, :3].astype(float)
A = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
kp = (A @ kp.reshape(-1, 3).T).T.reshape(kp.shape)
root_rot = np.einsum("ij,tjk->tik", A, root_rot)
kp -= kp[0, 0]
W = 5
pad = W // 2
kp_pad = np.pad(kp, ((pad, pad), (0, 0), (0, 0)), mode="edge")
kp = np.stack([kp_pad[i:i + used] for i in range(W)]).mean(0)
Mr = np.stack([np.pad(root_rot, ((pad, pad), (0, 0), (0, 0)), mode="edge")[i:i + used] for i in range(W)]).mean(0)
U, _, Vt = np.linalg.svd(Mr)
root_rot = np.einsum("tij,tjk->tik", U, Vt)
times = np.arange(used) / 30.0
names = list(A3Retargeter().names)
kp13 = kp[:, [SMPL_IDX[n] for n in names]]

solver = ChainA3Retargeter()
rp = solver.rest_position
idx = {n: i for i, n in enumerate(names)}
def _u(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)
rel = np.tile(np.eye(3), (used, len(names), 1, 1))
root_rel = root_rot @ root_rot[0].T
rel[:, idx["pelvis"]] = root_rel
for n, child in BONE_CHILD.items():
    i = idx[n]
    if child is None:
        continue
    if child.endswith("toe"):
        d_rest = REST_FWD
    else:
        d_rest = _u(rp[idx[child]] - rp[i])
    d_obs = _u(kp[:, SMPL_IDX[child]] - kp[:, SMPL_IDX[n]])
    rel[:, i] = align_many(d_rest, d_obs)
rel[:, idx["spine3"]] = align_many(_u(rp[idx["spine3"]] - rp[idx["pelvis"]]),
                                   _u(kp[:, SMPL_IDX["spine3"]] - kp[:, SMPL_IDX["pelvis"]]))
cal_quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(names), 1))
orientations = Rotation.from_matrix(rel.reshape(-1, 3, 3)).as_quat(scalar_first=True).reshape(used, len(names), 4)
source = BodyReference(times, names, kp13, orientations, kp13[0], cal_quats, names,
                       "P15A_PREVIEW_" + SESSION)

lengths, robot_lengths = [], []
for side in ("left", "right"):
    ids = [names.index(side + "_" + p) for p in ("hip", "knee", "ankle")]
    lengths.append(np.median(np.linalg.norm(np.diff(source.positions[:, ids], axis=1), axis=-1).sum(axis=-1)))
    robot_lengths.append(np.linalg.norm(np.diff(rp[ids], axis=0), axis=-1).sum())
scale = float(np.mean(robot_lengths) / np.mean(lengths))
print(json.dumps({"frames": used, "scale": round(scale, 4),
                  "human_leg_m": [round(float(v), 4) for v in lengths],
                  "robot_leg_m": [round(float(v), 4) for v in robot_lengths]}), flush=True)

status = "passed_kinematics_only"
try:
    report = solver.solve(source, scale, OUT)
except Exception as error:
    import traceback
    status = type(error).__name__ + ": " + str(error)[:200]
    traceback.print_exc()
    report = {}
print(json.dumps({"status": status, "summary": {k: report[k] for k in report
      if k in ("frames", "joint_limit_violations", "ik_position_p95_m", "ik_position_max_m",
               "ik_orientation_p95_rad", "joint_velocity_max_rad_s", "foot_speed_p95_m_s",
               "foot_geometry_penetration_frames", "duration_s")}}, ensure_ascii=False), flush=True)
