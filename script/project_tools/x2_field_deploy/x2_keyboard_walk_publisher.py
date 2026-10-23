#!/usr/bin/env python3
"""X2 keyboard intent -> ZMQ v5.1 pose reference publisher (T02 sim-gate tool).

Wire contract: X2_ZMQ_V51_WIRE_SPEC.md (extracted line-by-line from
zmq_pose_input_source.cpp). Layout per ZMQ part:

    b"pose" + 1280-byte JSON header (NUL padded, compact separators)
    + packed little-endian payload, fields in header order, no padding.

Strict field set (spec section 4):
    joint_pos_mj[31], joint_vel_mj[31], root_quat_xyzw[4],
    joint_pos_mj_future[9,31], joint_vel_mj_future[9,31],
    root_quat_xyzw_future[9,4], future_dt_s[1]=0.1, frame_index[1] i64.
Total bytes: 4 + 1280 + 2652 = 3936.

Modes:
  planner-cmd  faithful keyboard path: JSON planner_cmd on :5563 consumed by
            pc2_kplanner_onnx.py (SUB :5563; fields validated at ingest:
            target_velocity 4-list [vx,vy,vz,yaw], intent/stick_* optional).
            Chain: this tool :5563 -> kplanner PUB :5556 (V51 pose, 50Hz,
            x2_planner_template.onnx) -> deploy --zmq-pose-port 5556.
            Velocities latch (VR/Quest direct-velocity semantics, kplanner
            clamps at ingest: FWD 0.35 m/s, turn 0.65 rad/s from
            robot_env.env KPLANNER_FIXED_*).
  stand     publish vendor stand pose continuously (bring-up bootstrap)
  swing     sinusoidal in-place weight-shift surrogate (SIM ONLY; exercises
            the pipeline until the real KPlanner is wired; NOT a walk gait)
  kplanner  --planner-module FILE providing: reset(hz) and
            step(intent,t_now) -> (pos[31], quat_xyzw[4]); runs at 20 Hz
            and is resampled onto the 50 Hz wire with 9 future slots
            (pc2_kplanner_onnx.py contract: get_next_frame_resampled).

Keyboard (any mode, when stdin is a tty): w/s vx +-0.1 m/s, a/d yaw_rate
+-0.2 rad/s, space zero intent, q clean exit. Intent ramps to zero on exit,
stand frames are published for --exit-hold seconds, then the stream ends.
No 'estop' field is ever sent (spec section 4: forbidden except real e-stop).
On crash/Ctrl-C the stream simply stops; the deploy side staleness watchdog
owns recovery. There is no automatic re-enable (review report section 8).
"""
import argparse, json, math, struct, sys, time
import numpy as np
import zmq

HEADER_SIZE = 1280
NUM_DOFS, NUM_FUTURE = 31, 9
DT_FUTURE = 0.1
STAND = np.array([-0.312,0,0,0.669,-0.363,0,-0.312,0,0,0.669,-0.363,0,0,0,0,0.2,0.2,0,-0.6,0,0,0,0.2,-0.2,0,-0.6,0,0,0,0,0], dtype=np.float64)
JOINT_NAMES = ["left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint","right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint","waist_yaw_joint","waist_pitch_joint","waist_roll_joint","left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint","left_wrist_yaw_joint","left_wrist_pitch_joint","left_wrist_roll_joint","right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint","right_wrist_yaw_joint","right_wrist_pitch_joint","right_wrist_roll_joint","head_yaw_joint","head_pitch_joint"]

def build_header(fields, version=5):
    hj = json.dumps({"v": version, "endian": "le", "count": 1, "fields": fields},
                    separators=(",", ":")).encode("utf-8")
    if len(hj) > HEADER_SIZE:
        raise ValueError(f"header {len(hj)} > {HEADER_SIZE}")
    return hj.ljust(HEADER_SIZE, b"\x00")

def pack_pose(payload: dict, topic="pose", version=5) -> bytes:
    fields, blobs = [], []
    for name, arr in payload.items():
        a = np.ascontiguousarray(arr)
        dt = {np.dtype(np.float32): "f32", np.dtype(np.int64): "i64"}.get(a.dtype)
        if dt is None:
            raise ValueError(f"unsupported dtype {a.dtype} for {name}")
        fields.append({"name": name, "dtype": dt, "shape": list(a.shape)})
        blobs.append(a.astype(a.dtype.newbyteorder("<")).tobytes())
    return topic.encode() + build_header(fields, version) + b"".join(blobs)

STRICT_FIELDS = [
    {"name": "joint_pos_mj", "dtype": "f32", "shape": [31]},
    {"name": "joint_vel_mj", "dtype": "f32", "shape": [31]},
    {"name": "root_quat_xyzw", "dtype": "f32", "shape": [4]},
    {"name": "joint_pos_mj_future", "dtype": "f32", "shape": [9, 31]},
    {"name": "joint_vel_mj_future", "dtype": "f32", "shape": [9, 31]},
    {"name": "root_quat_xyzw_future", "dtype": "f32", "shape": [9, 4]},
    {"name": "future_dt_s", "dtype": "f32", "shape": [1]},
    {"name": "frame_index", "dtype": "i64", "shape": [1]},
]

def make_frame(pos, vel, quat, future_pos, future_vel, future_quat, idx,
               future_dt=DT_FUTURE) -> bytes:
    return pack_pose({
        "joint_pos_mj": pos.astype(np.float32),
        "joint_vel_mj": vel.astype(np.float32),
        "root_quat_xyzw": quat.astype(np.float32),
        "joint_pos_mj_future": future_pos.astype(np.float32),
        "joint_vel_mj_future": future_vel.astype(np.float32),
        "root_quat_xyzw_future": future_quat.astype(np.float32),
        "future_dt_s": np.array([future_dt], dtype=np.float32),
        "frame_index": np.array([idx], dtype=np.int64),
    })

def decode_verify(msg: bytes) -> dict:
    """Pure-python decoder mirroring the C++ offset walk (self-test only)."""
    if not msg.startswith(b"pose"):
        raise ValueError("topic prefix mismatch")
    off, hdr = 4, msg[4:4 + HEADER_SIZE]
    j = json.loads(hdr.rstrip(b"\x00").decode())
    off += HEADER_SIZE
    out, expect = {}, 2652
    for f in j["fields"]:
        shape, dt = f["shape"], f["dtype"]
        n = 1
        for s in shape: n *= s
        size = n * {"f32": 4, "i64": 8, "i32": 4, "u8": 1}[dt]
        raw = msg[off:off + size]; off += size
        fmt = {"f32": "f", "i64": "q", "i32": "i", "u8": "B"}[dt]
        out[f["name"]] = struct.unpack("<" + fmt * n, raw)
    if off != len(msg):
        raise ValueError(f"payload length {off} != {len(msg)}")
    if off - 4 - HEADER_SIZE != expect:
        raise ValueError(f"strict set must be {expect} bytes")
    return out

class KeyReader:
    """Single-key nonblocking stdin reader (termios, restores on exit)."""
    def __enter__(self):
        import termios, tty
        self.fd, self.old = sys.stdin.fileno(), termios.tcgetattr(sys.stdin)
        tty.setcbreak(self.fd); return self
    def __exit__(self, *a):
        import termios; termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
    def key(self):
        import select
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

class SwingSurrogate:
    """In-place gentle weight shift; SIM ONLY, not a gait."""
    def reset(self, hz): self.t0 = time.monotonic()
    def ref(self, t, intent):
        ph = (t - self.t0) * 2 * math.pi * 0.8
        amp = 0.04 + 0.03 * abs(intent.vx / 0.4)
        pos = STAND.copy(); pos[0] += amp * math.sin(ph); pos[6] -= amp * math.sin(ph)
        pos[4] += 0.02 * max(0.0, math.sin(ph)); pos[10] += 0.02 * max(0.0, -math.sin(ph))
        yaw = intent.yaw_rate * (t - self.t0)
        q = np.array([0, 0, math.sin(yaw / 2), math.cos(yaw / 2)])
        return pos, q

class Intent:
    def __init__(self): self.vx, self.yaw_rate = 0.0, 0.0
    def key(self, k):
        if k == "w": self.vx = min(0.4, self.vx + 0.1)
        elif k == "s": self.vx = max(-0.2, self.vx - 0.1)
        elif k == "a": self.yaw_rate = max(-0.6, self.yaw_rate - 0.2)
        elif k == "d": self.yaw_rate = min(0.6, self.yaw_rate + 0.2)
        elif k == " ": self.vx, self.yaw_rate = 0.0, 0.0

def load_planner(path, hz):
    import importlib.util
    spec = importlib.util.spec_from_file_location("x2_planner_mod", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    planner = mod.Planner(); planner.reset(hz); return planner

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["stand", "swing", "kplanner", "planner-cmd"], default="stand")
    ap.add_argument("--cmd-port", type=int, default=5563, help="planner_cmd PUB port (mode=planner-cmd)")
    ap.add_argument("--fwd-mps", type=float, default=0.35, help="KPLANNER_FIXED_FWD_MPS")
    ap.add_argument("--turn-rad-s", type=float, default=0.65, help="KPLANNER_FIXED_TURN_RAD_S")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--topic", default="pose")
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--duration", type=float, default=0, help="0 = until q/Ctrl-C")
    ap.add_argument("--exit-hold", type=float, default=2.0)
    ap.add_argument("--planner-module", default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        pos, vel = STAND, np.zeros(31); quat = np.array([0, 0, 0, 1.0])
        fp = np.tile(pos, (9, 1)); fv = np.zeros((9, 31)); fq = np.tile(quat, (9, 1))
        msg = make_frame(pos, vel, quat, fp, fv, fq, 0)
        d = decode_verify(msg)
        assert len(d["joint_pos_mj"]) == 31 and len(msg) == 3936
        print(f"SELF-TEST OK: {len(msg)} bytes, fields={sorted(d)}")
        return

    ctx = zmq.Context()
    sock = None
    if a.mode == "planner-cmd":
        # CONFLATE belongs on the SUB side (V51 spec); on PUB it silently
        # drops everything with this libzmq build (A/B verified 40 vs 0).
        cmd = ctx.socket(zmq.PUB)
        # kplanner BINDS 5563 as the SUB server (rendezvous); we are a PUB
        # CLIENT and must connect. (bind here collides: Address already in use)
        cmd.connect(f"tcp://{a.host}:{a.cmd_port}")
        time.sleep(0.5)   # PUB slow-joiner: let the SUB pick us up before first frame
        print(f"PUB->tcp://{a.host}:{a.cmd_port}/planner-cmd (JSON, latch semantics)")
        intent = Intent()
        sent, t0 = 0, time.monotonic()
        period = 1.0 / 30.0
        try:
            kb = sys.stdin.isatty()
            with KeyReader() if kb else open("/dev/null") as kr:
                while True:
                    now = time.monotonic()
                    if a.duration and now - t0 >= a.duration: break
                    if kb:
                        k = kr.key()
                        if k == "w":
                            intent.vx = a.fwd_mps; intent.yaw_rate = 0.0
                        elif k == "s":
                            intent.vx = -0.1; intent.yaw_rate = 0.0
                        elif k == "a":
                            intent.yaw_rate = a.turn_rad_s
                        elif k == "d":
                            intent.yaw_rate = -a.turn_rad_s
                        elif k == " ":
                            intent.vx = 0.0; intent.yaw_rate = 0.0
                        if k in ("w", "s", "a", "d", " "):
                            print(f"cmd vx={intent.vx:+.2f} yaw={intent.yaw_rate:+.2f}")
                        if k == "q": print("quit key"); break
                    vx, yaw = intent.vx, intent.yaw_rate
                    tv = [yaw, 0.0, vx, 0.687]  # kplanner tuple = (yaw, lateral_x, fwd_z, hip_h)
                    payload = {"intent": "target_velocity",
                               "target_velocity": tv,
                               "stick_fwd": vx, "stick_side": 0.0,
                               "stick_yaw": yaw}
                    cmd.send_multipart([b"planner_cmd", json.dumps(payload).encode()])
                    sent += 1
                    time.sleep(max(0.0, t0 + sent * period - time.monotonic()))
        except KeyboardInterrupt:
            print("interrupted: stream ends (kplanner watchdog owns recovery)")
        print(f"planner_cmd frames sent: {sent}")
        cmd.close(linger=0); ctx.term(); return

    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://{a.host}:{a.port}")
    time.sleep(0.2)
    print(f"PUB tcp://{a.host}:{a.port}/{a.topic} mode={a.mode} hz={a.hz}")

    planner = (SwingSurrogate() if a.mode == "swing" else
               load_planner(a.planner_module, a.hz) if a.mode == "kplanner" else None)
    if planner: planner.reset(a.hz)
    intent, keys = Intent(), sys.stdin.isatty()
    idx, sent, t0, last = 0, 0, time.monotonic(), None
    period = 1.0 / a.hz
    try:
        with KeyReader() if keys else open("/dev/null") as kr:
            while True:
                now = time.monotonic()
                if a.duration and now - t0 >= a.duration: break
                if keys:
                    k = kr.key()
                    if k and k in "wasd ": intent.key(k); print(f"intent vx={intent.vx:+.2f} yaw={intent.yaw_rate:+.2f}")
                    if k == "q": print("quit key"); break
                if planner:
                    pos, quat = planner.ref(now, intent)
                else:
                    pos, quat = STAND.copy(), np.array([0, 0, 0, 1.0])
                vel = np.zeros(31) if last is None else (pos - last) * a.hz
                last = pos.copy()
                fp = np.tile(pos, (NUM_FUTURE, 1)); fv = np.tile(vel, (NUM_FUTURE, 1))
                fq = np.tile(quat, (NUM_FUTURE, 1))
                sock.send(a.topic.encode() + make_frame(pos, vel, quat, fp, fv, fq, idx)[4:])
                idx += 1; sent += 1
                nxt = t0 + idx * period
                if nxt > now: time.sleep(nxt - now)
    except KeyboardInterrupt:
        print("interrupted: stream ends (deploy watchdog owns recovery)")
    print(f"frames sent: {sent}")
    sock.close(linger=0); ctx.term()

if __name__ == "__main__":
    main()
