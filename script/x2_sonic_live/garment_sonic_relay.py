"""LEGACY-ONLY WORKAROUND: only needed when on-board garment_intent_adapter.py
is unpatched (``LEGACY=1 bash script/x2_sonic_live/run_shadow_live.sh``).

Do NOT place this relay in front of a patched ``garment_intent_adapter.py``
(which already converts SMPL Y-up -> SONIC Z-up).
"""
import sys, time, zmq, numpy as np
from scipy.spatial.transform import Rotation as Rot
sys.path[:0] = ["/home/agi/projects/project_tools/x2_smpl_wholebody"]
from wire_format import pack_pose_message, unpack_message

M = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
T = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
R_MT = M @ T.T

ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.connect("tcp://127.0.0.1:15573")
sub.setsockopt(zmq.SUBSCRIBE, b"pico_intent")

pub = ctx.socket(zmq.PUB)
pub.setsockopt(zmq.LINGER, 0)
pub.bind("tcp://127.0.0.1:15575")

count = 0
while True:
    raw = sub.recv()
    f = unpack_message(raw, expected_topic="pico_intent").fields
    sj = np.asarray(f["smpl_joints"], np.float32).reshape(24, 3)
    hq = np.asarray(f["human_quat"], np.float32).ravel()[:4]
    eg = np.asarray(f.get("engaged", [0.0]), np.float32).ravel()[:1]
    wp = np.asarray(f.get("wrist_pr", [0.0]*4), np.float32).ravel()[:4]
    fi = np.asarray(f.get("frame_index", [count]), np.int64).ravel()[:1]

    j_sonic = (sj @ M.T).astype(np.float32)
    R_hq = Rot.from_quat([hq[1], hq[2], hq[3], hq[0]]).as_matrix()
    R_sonic = R_MT @ R_hq @ M.T
    qx = Rot.from_matrix(R_sonic).as_quat()
    q_sonic = np.array([qx[3], qx[0], qx[1], qx[2]], dtype=np.float32)

    out = pack_pose_message({
        "smpl_joints": j_sonic.reshape(72),
        "human_quat": q_sonic,
        "engaged": eg,
        "wrist_pr": wp,
        "frame_index": fi,
    }, topic="pico_intent", version=4)
    pub.send(out)
    count += 1
    if count == 1 or count % 60 == 0:
        print(f"RELAY_TX count={count} eng={float(eg[0]):.1f} q={np.round(q_sonic,3).tolist()}", flush=True)
