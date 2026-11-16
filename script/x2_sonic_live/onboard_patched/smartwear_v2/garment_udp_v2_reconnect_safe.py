"""BNO085 衣服 -> SMPL-X -> GMR -> G1 qpos -> Humanoid-GPT (UDP HMCP 流) —— V2 版。

基于 ``[DEMO] BNO085接收器版 上衣+裤子_GMR_HumanoidGPT_UDP.py`` 重构：

* 输入层改用 ``hardware_boot.receive_085_linux_v2``（UB 封装 + 137 字节帧
  + Q8/Q14 定点解码 + UTC 时间戳；对外接口与 V1 兼容）；
* V2 帧每设备固定 6 IMU 槽，裤子取槽 ``[0,1,2,4,5]``（与 FGP README 一致）；
* V2 时间同步（UQ/UR 授时）由 FGP 的 ``uart-time-bridge``（chrony）负责，
  本脚本只消费 IMU 数据流；未同步时帧内 UTC 为 None，数据仍可用。

下游（SMPL/GMR/UDP）与 V1 完全相同。接收端二选一:
  * G1 Jetson 板载: ``python -m deploy.onboard_deploy_wo_GMR.play_track_onboard_wo_GMR
    --policy-type transformer --onnx-track storage/ckpts/pns_wo_priv216_trans.onnx
    --history-len 16``
  * 本地仿真验证(无需机器人): ``python -m tools.play_track_local_sim --source udp``
"""

import os
import sys
import time
import json
import atexit
import signal
import subprocess
import argparse
import logging
import importlib.util
from threading import Event, Thread

sys.coinit_flags = 0  # 0 means MTA

import onnxruntime as rt
import torch
import numpy as np

from Socket.UDP import *
import config
from Aplus.tools.pd_controll import FpsController

from general_motion_retargeting import SmpleXConverter
from general_motion_retargeting.SmpleXConverter import *
from general_motion_retargeting import GeneralMotionRetargeting as GMR

# 输入层按 --input 切换:
#   uart: receive_085_linux_v2 (串口 dongle, UB 封装/137B)
#   ble : receive_bleak_085_v2 (蓝牙直连两台 SmartWear, 137B V2 Notify)
_INPUT_BLE = False


def _import_input_layer(input_mode: str):
    global receive_085_linux, _INPUT_BLE
    if input_mode == "ble":
        from hardware_boot import receive_bleak_085_v2 as _mod
        _INPUT_BLE = True
    else:
        from hardware_boot import receive_085_linux_v2 as _mod
        _INPUT_BLE = False
    receive_085_linux = _mod
    return _mod


def _ble_input_status(max_age_s: float = 0.5):
    """Return whether both V2 garments have a fresh frame after reconnects."""
    if not _INPUT_BLE:
        return True, "uart"

    state_fn = getattr(receive_085_linux, "get_connection_state", None)
    if state_fn is not None:
        state = state_fn()
        if not all(state["active"]):
            return False, f"connection={state['active']}"
        if min(state["buffer_lengths"]) < 1:
            return False, f"buffers={state['buffer_lengths']}"

    ages_fn = getattr(receive_085_linux, "get_frame_ages", None)
    if ages_fn is not None:
        ages = ages_fn()
        if not all(np.isfinite(age) and age <= max_age_s for age in ages):
            return False, f"ages={ages[0]:.3f}/{ages[1]:.3f}s"

    if not receive_085_linux.data_up_buffer or not receive_085_linux.data_down_buffer:
        return False, "empty buffer"
    return True, "fresh"


def _snapshot_input_frames(history: int = 1):
    """Copy matching buffer tails while the receiver holds its mutation lock."""
    lock = getattr(receive_085_linux, "_buffer_lock", None)

    def snapshot():
        up = list(receive_085_linux.data_up_buffer)[-history:]
        down = list(receive_085_linux.data_down_buffer)[-history:]
        if not up or not down:
            raise IndexError("garment buffer cleared during BLE reconnect")
        return np.asarray(up).copy(), np.asarray(down).copy()

    if lock is None:
        return snapshot()
    with lock:
        return snapshot()

# V2 帧每设备固定 6 槽，裤子取槽 0,1,2,4,5（FGP README）
PANT_SLOTS_V2 = np.array([0, 1, 2, 4, 5], dtype=np.int32)

# Humanoid-GPT 仓库的 HMCP 协议(单协议来源, 避免复制漂移)。
# 默认指向 hp3090 上的仓库; 衣服机与仓库机不同时用 --hgpt-root 指定。
_HGPT_DEFAULT_ROOT = os.path.expanduser("~/projects/humanoid-gpt-humanplus")


def _import_hgpt_protocol(hgpt_root: str):
    if not hgpt_root or not os.path.isdir(hgpt_root):
        raise FileNotFoundError(
            f"Humanoid-GPT repo not found at {hgpt_root!r}; pass --hgpt-root"
        )
    if hgpt_root not in sys.path:
        sys.path.insert(0, hgpt_root)
    from deploy.onboard_deploy_wo_GMR.protocol import (  # noqa: F401
        DEFAULT_PORT,
        G1_DOF_FULL,
        encode_frame,
    )
    return DEFAULT_PORT, G1_DOF_FULL, encode_frame


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CALIBRATION_DONE = False
RUNNING = True
_STOP_EVENT = Event()

clock = Clock()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GMR 输出的是 MuJoCo qpos，GR00T v1 要求 joint_pos/joint_vel 为 IsaacLab 29 关节顺序。
G1_MUJOCO_TO_ISAACLAB_DOF = np.array(
    [
        0, 6, 12, 1, 7, 13, 2, 8, 14, 3,
        9, 15, 22, 4, 10, 16, 23, 5, 11, 17,
        24, 18, 25, 19, 26, 20, 27, 21, 28,
    ],
    dtype=np.int32,
)


def build_args():
    parser = argparse.ArgumentParser(description="BNO085 mocap retarget to G1 and stream via GR00T ZMQ")

    parser.add_argument("--mode", default="single", type=str, choices=["single", "twins"])
    parser.add_argument("--skeleton", default="smpl", type=str, choices=["smpl", "h36m"])
    parser.add_argument(
        "--rotation_type",
        default="AXIS_ANGLE",
        type=str,
        choices=["AXIS_ANGLE", "DCM", "QUATERNION", "R6D", "EULER_ANGLE"],
    )
    parser.add_argument(
        "--part",
        default="body",
        type=str,
        choices=[
            "body",
            "upper_body",
            "lower_body",
            "head",
            "spine",
            "left_hand",
            "right_hand",
            "left_leg",
            "right_leg",
            "hands",
        ],
    )
    parser.add_argument("--fps", default=35, type=int, help="mocap inference + send fps")
    parser.add_argument(
        "--input",
        default="ble",
        type=str,
        choices=["ble", "uart"],
        help="输入层: ble=蓝牙直连两台 SmartWear(V2 137B); uart=串口 dongle",
    )
    parser.add_argument(
        "--addr-up",
        default="D5:F4:A2:41:93:4B",
        type=str,
        help="上衣 SmartWear BLE 地址 (--input ble)",
    )
    parser.add_argument(
        "--addr-down",
        default="F3:FB:AD:FC:7D:82",
        type=str,
        help="裤子 SmartWear BLE 地址 (--input ble)",
    )
    parser.add_argument(
        "--auto-calibrate",
        action="store_true",
        help="免交互: 数据就绪后自动倒计时开始 T-Pose 校准(SSH 无终端时自动启用)",
    )
    parser.add_argument(
        "--calibration-trigger-file",
        default="",
        help="数据就绪后等待该文件出现，再开始 T-Pose 校准倒计时",
    )
    parser.add_argument(
        "--stream-mode",
        default="udp",
        type=str,
        choices=["udp", "zmq", "both", "none"],
        help="udp=Humanoid-GPT HMCP 流(默认); zmq=旧 GR00T/SONIC 流; both=同时发; "
             "none=不发下游(X2 SMPL 链专用: 只出 --pico-intent-port)",
    )
    parser.add_argument("--robot-ip", default="127.0.0.1", type=str, help="Humanoid-GPT UDP 接收端 IP (G1 Jetson 或本地仿真)")
    parser.add_argument("--udp-port", default=51234, type=int, help="Humanoid-GPT HMCP UDP 端口 (protocol.DEFAULT_PORT)")
    parser.add_argument(
        "--hgpt-root",
        default=_HGPT_DEFAULT_ROOT,
        type=str,
        help="Humanoid-GPT 仓库根目录(提供 HMCP protocol.py)",
    )
    parser.add_argument(
        "--smplx-folder",
        default="",
        type=str,
        help="SMPL-X 模型目录(含 smplx/ 子目录); 默认自动探测本脚本所在目录的 assets/body_models",
    )
    parser.add_argument(
        "--smpl-converter",
        default="fast",
        choices=["fast", "smplx"],
        help="fast=lightweight SMPL FK for HMCP; smplx=full model reference path",
    )
    parser.add_argument(
        "--smpl-fast-device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="device for lightweight SMPL FK; cpu avoids Orin GPU synchronization",
    )
    parser.add_argument("--zmq-host", default="*", type=str, help="ZMQ PUB bind host")
    parser.add_argument("--zmq-port", default=5556, type=int, help="ZMQ PUB bind port")
    parser.add_argument(
        "--zmq-port-search-count",
        default=20,
        type=int,
        help="when the preferred ZMQ port is busy, try this many consecutive ports",
    )
    parser.add_argument("--zmq-topic", default="pose", type=str, help="GR00T deployment subscribed topic")
    parser.add_argument(
        "--protocol-version",
        default=1,
        type=int,
        choices=[1, 3],
        help="GR00T ZMQ protocol version: 1=joint-based, 3=joint+SMPL combined",
    )
    parser.add_argument(
        "--gr00t-root",
        default=os.path.expanduser("~/GR00T-WholeBodyControl"),
        type=str,
        help="preferred GR00T-WholeBodyControl root; also auto-detects $GR00T_ROOT and ~/sonic_lty/GR00T-WholeBodyControl",
    )
    parser.add_argument(
        "--catch-up",
        action="store_true",
        help="append catch_up=true field; default false for realtime streaming",
    )
    parser.add_argument(
        "--vel-alpha",
        default=0.2,
        type=float,
        help="EMA factor for finite-difference joint velocity smoothing",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="disable local MuJoCo viewer and only stream retargeted data",
    )
    parser.add_argument(
        "--log-interval",
        default=1.0,
        type=float,
        help="seconds between runtime status logs",
    )
    parser.add_argument(
        "--gmr-max-iter",
        default=3,
        type=int,
        help="maximum IK refinement iterations for GMR retargeting",
    )
    parser.add_argument(
        "--gmr-damping",
        default=1.0,
        type=float,
        help="IK damping for GMR retargeting",
    )
    parser.add_argument(
        "--gmr-verbose",
        action="store_true",
        help="enable verbose GMR initialization logs",
    )
    parser.add_argument(
        "--gmr-single-stage",
        action="store_true",
        help="only run the first IK stage in GMR for lower latency",
    )
    parser.add_argument(
        "--gmr-backend",
        default="baseline",
        choices=["baseline", "fixed", "native"],
        help="fixed caches Mink limits; native also uses C++ target preprocessing",
    )
    parser.add_argument(
        "--lfp-backend",
        default="local-cpu",
        choices=["local-cpu", "local-cuda", "soc2"],
        help="local-cpu avoids GPU contention; local-cuda is reference; soc2 is optional offload",
    )
    parser.add_argument("--lfp-soc2-host", default="10.0.1.42")
    parser.add_argument("--lfp-soc2-port", default=51236, type=int)
    parser.add_argument("--lfp-timeout-ms", default=80.0, type=float)
    # --- X2 SMPL whole-body (OKR 衣服→FGP→SMPL→X2): pico_intent 直出 ---
    parser.add_argument(
        "--pico-intent-port", default=0, type=int,
        help=">0 时每帧把 FGP SMPL 输出映射成 pico_intent 并 PUB 到该端口"
             "(token service 的 intent SUB, 机器默认 5573); 0 = 关闭",
    )
    parser.add_argument(
        "--pico-intent-engage-file", default="~/pico_intent.engage",
        help="engage 哨兵文件: 文件存在才允许 engage",
    )
    parser.add_argument(
        "--pico-intent-stationary-s", default=2.0, type=float,
        help="首次 engage 要求穿衣者静止的秒数 (2026-08 代理 STANDSTILL 规则)",
    )
    return parser.parse_args()


def _import_garment_intent_adapter():
    """X2 token 链的 pico_intent 适配器 (script/project_tools/x2_smpl_wholebody)."""
    _here = os.path.dirname(os.path.abspath(__file__))
    _mod_root = os.path.abspath(os.path.join(_here, "..", "project_tools", "x2_smpl_wholebody"))
    if _mod_root not in sys.path:
        sys.path.insert(0, _mod_root)
    import garment_intent_adapter as _gia
    return _gia


args = build_args()


def _import_gr00t_pack_pose_message(gr00t_root: str):
    candidate_roots = []
    env_root = os.environ.get("GR00T_ROOT")
    if env_root:
        candidate_roots.append(env_root)
    if gr00t_root:
        candidate_roots.append(gr00t_root)
    candidate_roots.append(os.path.expanduser("~/GR00T-WholeBodyControl"))
    candidate_roots.append(os.path.expanduser("~/sonic_lty/GR00T-WholeBodyControl"))

    checked_roots = []
    for candidate_root in candidate_roots:
        normalized_root = os.path.abspath(os.path.expanduser(candidate_root))
        if normalized_root in checked_roots:
            continue
        checked_roots.append(normalized_root)

        module_path = os.path.join(
            normalized_root, "gear_sonic", "utils", "teleop", "zmq", "zmq_planner_sender.py"
        )
        if not os.path.isfile(module_path):
            continue

        spec = importlib.util.spec_from_file_location("gr00t_zmq_planner_sender", module_path)
        if spec is None or spec.loader is None:
            continue

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logging.info("Loaded GR00T ZMQ packer from %s", module_path)
        return module.pack_pose_message

    logging.warning(
        "GR00T repository not found from candidates %s; using built-in ZMQ pose packer",
        checked_roots,
    )

    def _build_header(fields: list, version: int = 1, count: int = 1) -> bytes:
        header = {
            "v": version,
            "endian": "le",
            "count": count,
            "fields": fields,
        }
        header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
        header_size = 1280
        if len(header_json) > header_size:
            raise ValueError(f"Header too large: {len(header_json)} > {header_size}")
        return header_json.ljust(header_size, b"\x00")

    def _fallback_pack_pose_message(pose_data: dict, topic: str = "pose", version: int = 3) -> bytes:
        fields = []
        binary_data = []

        for key, value in pose_data.items():
            if not isinstance(value, np.ndarray):
                continue

            if value.dtype == np.float32:
                dtype_str = "f32"
            elif value.dtype == np.float64:
                dtype_str = "f64"
            elif value.dtype == np.int32:
                dtype_str = "i32"
            elif value.dtype == np.int64:
                dtype_str = "i64"
            elif value.dtype == np.uint8:
                dtype_str = "u8"
            elif value.dtype == bool:
                dtype_str = "bool"
            else:
                dtype_str = "f32"
                value = value.astype(np.float32)

            fields.append({"name": key, "dtype": dtype_str, "shape": list(value.shape)})

            if not value.flags["C_CONTIGUOUS"]:
                value = np.ascontiguousarray(value)
            if value.dtype.byteorder == ">":
                value = value.astype(value.dtype.newbyteorder("<"))

            binary_data.append(value.tobytes())

        header_bytes = _build_header(fields, version=version, count=1)
        return topic.encode("utf-8") + header_bytes + b"".join(binary_data)

    return _fallback_pack_pose_message


class _Gr00TZmqPosePublisher:
    """旧链路: 向 GR00T/SONIC 控制器发 ZMQ pose 流(原脚本逻辑, 未改动)。"""
    def __init__(
        self,
        host: str,
        port: int,
        port_search_count: int,
        topic: str,
        catch_up: bool,
        vel_alpha: float,
        gr00t_root: str,
        protocol_version: int,
    ):
        import zmq  # 仅在需要 ZMQ 模式时引入

        self.topic = topic
        self.catch_up = catch_up
        self.vel_alpha = float(np.clip(vel_alpha, 0.0, 1.0))
        self.protocol_version = protocol_version
        self.frame_index = 0
        self.prev_joint_pos_il = None
        self.prev_joint_vel_il = None
        self.prev_time = None

        self.pack_pose_message = _import_gr00t_pack_pose_message(gr00t_root)

        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUB)
        self.port = self._bind_socket(host=host, port=port, search_count=port_search_count)
        self.endpoint = f"tcp://{host}:{self.port}"
        time.sleep(0.5)
        logging.info("GR00T ZMQ publisher bound to %s, topic=%s", self.endpoint, self.topic)

    def _bind_socket(self, host: str, port: int, search_count: int) -> int:
        search_count = max(1, int(search_count))
        last_error = None
        for offset in range(search_count):
            candidate_port = port + offset
            candidate_endpoint = f"tcp://{host}:{candidate_port}"
            try:
                self.socket.bind(candidate_endpoint)
                if offset > 0:
                    logging.warning(
                        "Preferred ZMQ endpoint tcp://%s:%d was busy; using %s instead",
                        host,
                        port,
                        candidate_endpoint,
                    )
                return candidate_port
            except zmq.ZMQError as exc:
                last_error = exc
                if exc.errno != zmq.EADDRINUSE:
                    raise

        raise zmq.ZMQError(
            f"Failed to bind any ZMQ endpoint from tcp://{host}:{port} "
            f"to tcp://{host}:{port + search_count - 1}: {last_error}"
        )

    def _compute_joint_vel(self, joint_pos_il: np.ndarray, now: float) -> np.ndarray:
        if self.prev_joint_pos_il is None or self.prev_time is None:
            joint_vel_il = np.zeros_like(joint_pos_il, dtype=np.float32)
        else:
            dt = max(now - self.prev_time, 1e-3)
            raw_vel = ((joint_pos_il - self.prev_joint_pos_il) / dt).astype(np.float32)
            if self.prev_joint_vel_il is None:
                joint_vel_il = raw_vel
            else:
                joint_vel_il = (
                    self.vel_alpha * raw_vel + (1.0 - self.vel_alpha) * self.prev_joint_vel_il
                ).astype(np.float32)

        self.prev_joint_pos_il = joint_pos_il.copy()
        self.prev_joint_vel_il = joint_vel_il.copy()
        self.prev_time = now
        return joint_vel_il

    def publish_motion(
        self,
        qpos_mj: np.ndarray | None,
        body_quat_w: np.ndarray | None = None,
        smpl_joints: np.ndarray | None = None,
        smpl_pose: np.ndarray | None = None,
    ):
        if qpos_mj is not None:
            qpos_mj = np.asarray(qpos_mj, dtype=np.float32).reshape(-1)
            if qpos_mj.shape[0] < 36:
                raise ValueError(f"Expected qpos with at least 36 dims (3+4+29), got {qpos_mj.shape[0]}")
            root_quat_wxyz = qpos_mj[3:7].astype(np.float32)
            joint_pos_mj = qpos_mj[7:36].astype(np.float32)
            joint_pos_il = joint_pos_mj[G1_MUJOCO_TO_ISAACLAB_DOF]
        else:
            if body_quat_w is None:
                raise ValueError("body_quat_w is required when qpos_mj is None")
            root_quat_wxyz = np.asarray(body_quat_w, dtype=np.float32).reshape(4)
            joint_pos_il = np.zeros(29, dtype=np.float32)

        joint_vel_il = self._compute_joint_vel(joint_pos_il, time.perf_counter())

        numpy_data = {
            "body_quat_w": root_quat_wxyz.reshape(1, 4),
            "joint_pos": joint_pos_il.reshape(1, 29),
            "joint_vel": joint_vel_il.reshape(1, 29),
            "frame_index": np.array([self.frame_index], dtype=np.int64),
        }
        if self.protocol_version == 3:
            if smpl_joints is None or smpl_pose is None:
                raise ValueError("Protocol v3 requires smpl_joints and smpl_pose")
            numpy_data["smpl_joints"] = np.asarray(smpl_joints, dtype=np.float32).reshape(1, 24, 3)
            numpy_data["smpl_pose"] = np.asarray(smpl_pose, dtype=np.float32).reshape(1, 21, 3)
        if self.catch_up:
            numpy_data["catch_up"] = np.array([1], dtype=np.uint8)

        packed_message = self.pack_pose_message(
            numpy_data, topic=self.topic, version=self.protocol_version
        )
        self.socket.send(packed_message)
        self.frame_index += 1
        return joint_pos_il, joint_vel_il


class _HgptUdpPublisher:
    """Humanoid-GPT 链路: 每帧 G1 qpos(36) 用 HMCP 协议 UDP 发给接收端。

    接收端 (deploy.onboard_deploy_wo_GMR) 只消费 qpos, 参考特征与关节速度
    由接收端 LiveRefConverter 自行计算, 这里不做二次加工。
    """

    def __init__(self, robot_ip: str, udp_port: int, hgpt_root: str):
        import socket

        default_port, _, encode_frame = _import_hgpt_protocol(hgpt_root)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._dest = (robot_ip, udp_port if udp_port else default_port)
        self._encode = encode_frame
        self.frame_index = 0
        logging.info("HMCP UDP publisher -> %s:%d", *self._dest)

    def publish_motion(
        self,
        qpos_mj: np.ndarray | None,
        body_quat_w: np.ndarray | None = None,
        smpl_joints: np.ndarray | None = None,
        smpl_pose: np.ndarray | None = None,
    ):
        if qpos_mj is None:
            raise ValueError("HMCP UDP 需要 GMR 输出的 qpos_mj (36 维)")
        qpos_mj = np.asarray(qpos_mj, dtype=np.float32).reshape(-1)
        if qpos_mj.shape[0] < 36:
            raise ValueError(f"Expected qpos with at least 36 dims, got {qpos_mj.shape[0]}")
        packet = self._encode(
            seq=self.frame_index,
            send_ts=time.time(),
            qpos=qpos_mj[:36],
        )
        self._socket.sendto(packet, self._dest)
        self.frame_index += 1
        return qpos_mj[7:36], np.zeros(29, dtype=np.float32)  # (占位返回值, 与旧接口一致)


class MotionPublisher:
    """按 --stream-mode 同时/分别向 Humanoid-GPT (UDP) 与 GR00T (ZMQ) 发布。"""

    def __init__(self, args):
        self.args = args
        self.streamers = []
        if args.stream_mode in ("udp", "both"):
            self.streamers.append(_HgptUdpPublisher(args.robot_ip, args.udp_port, args.hgpt_root))
        if args.stream_mode in ("zmq", "both"):
            self.streamers.append(
                _Gr00TZmqPosePublisher(
                    host=args.zmq_host,
                    port=args.zmq_port,
                    port_search_count=args.zmq_port_search_count,
                    topic=args.zmq_topic,
                    catch_up=args.catch_up,
                    vel_alpha=args.vel_alpha,
                    gr00t_root=args.gr00t_root,
                    protocol_version=args.protocol_version,
                )
            )
        if not self.streamers and args.stream_mode != "none":
            raise ValueError(f"未知 stream-mode: {args.stream_mode}")

    def publish_motion(self, qpos_mj=None, body_quat_w=None, smpl_joints=None, smpl_pose=None):
        if not self.streamers:
            # stream-mode=none: pico_intent-only run; keep the loop unpack happy
            return np.zeros(29, np.float32), np.zeros(29, np.float32)
        out = None
        for s in self.streamers:
            out = s.publish_motion(
                qpos_mj=qpos_mj, body_quat_w=body_quat_w,
                smpl_joints=smpl_joints, smpl_pose=smpl_pose,
            )
        return out

    @property
    def frame_index(self):
        return max((s.frame_index for s in self.streamers), default=0)


def build_onnx_providers():
    available = rt.get_available_providers()
    preferred = []
    if "CUDAExecutionProvider" in available:
        preferred.append("CUDAExecutionProvider")
    preferred.append("CPUExecutionProvider")
    logging.info("ONNXRuntime available providers: %s", available)
    logging.info("ONNXRuntime selected providers: %s", preferred)
    return preferred


def data_transmit(fps=30):
    global CALIBRATION_DONE
    global i_session
    global imu_order_clothes
    global imu_order_pant
    global streamer

    fc = FpsController(set_fps=fps)

    body_model = None
    if args.smpl_converter == "smplx" and args.smplx_folder:
        smplx_folder = args.smplx_folder
    elif args.smpl_converter == "smplx":
        # 自动探测: 脚本所在目录的 assets/body_models
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _candidate = os.path.join(_script_dir, "assets", "body_models")
        if os.path.isdir(os.path.join(_candidate, "smplx")):
            smplx_folder = _candidate
        else:
            smplx_folder = "/home/humanplus/projects/Human+智能服装动作捕捉系统/assets/body_models"
            logging.warning(
                "未找到 %s, 回退到硬编码路径 %s; 可用 --smplx-folder 指定",
                _candidate, smplx_folder,
            )
    if args.smpl_converter == "smplx":
        import smplx

        body_model = smplx.create(
            smplx_folder,
            "smplx",
            gender="neutral",
            use_pca=False,
        )
        if torch.cuda.is_available():
            body_model = body_model.to(device)
        logging.info("SMPL converter=smplx device=%s", device)
    else:
        if args.stream_mode in ("zmq", "both") and args.protocol_version == 3:
            raise ValueError("--smpl-converter fast does not provide protocol-v3 SMPL payloads")
        logging.info(
            "SMPL converter=fast device=%s (full SMPL-X model is not loaded)",
            args.smpl_fast_device,
        )
    smplexconverter = SMPLXConverter()
    retarget_class = GMR
    if args.gmr_backend == "fixed":
        from native_gmr import LimitsFixedGMR

        class FixedGMR(LimitsFixedGMR, GMR):
            pass

        retarget_class = FixedGMR
    elif args.gmr_backend == "native":
        from native_gmr import NativePreprocessedGMR

        class NativeGMR(NativePreprocessedGMR, GMR):
            pass

        retarget_class = NativeGMR
    logging.info("GMR backend=%s", args.gmr_backend)
    logging.info("[pico-diag] pub=%s adapter=%s engage_file=%s exists=%s",
                 pico_pub is not None, pico_adapter is not None,
                 args.pico_intent_engage_file,
                 os.path.exists(os.path.expanduser(args.pico_intent_engage_file)))
    pico_diag_n = 0
    if args.stream_mode == "none":
        # X2 SMPL 链: 不做任何下游流, GMR(旧路线的 MuJoCo 重定向)整段跳过
        logging.info("stream-mode=none: GMR skipped (pico_intent only)")
        retarget = None
    else:
        retarget = retarget_class(
            src_human="smplx",
            tgt_robot="unitree_g1",
            actual_human_height=1.8,
            damping=args.gmr_damping,
            verbose=args.gmr_verbose,
        )
        retarget.max_iter = max(0, args.gmr_max_iter)
        if args.gmr_single_stage:
            retarget.use_ik_match_table2 = False
            retarget.tasks2 = []
        retarget.human_body_to_task2 = {}
        retarget.pos_offsets2 = {}
        retarget.rot_offsets2 = {}
        retarget.task_errors2 = {}
        logging.info("GMR configured in single-stage mode: ik_match_table2 disabled")
    viewer = None
    if not args.no_viewer:
        try:
            from general_motion_retargeting import RobotMotionViewer
        except ImportError:
            # Older target bundles keep the viewer out of the package exports.
            from general_motion_retargeting.robot_motion_viewer import RobotMotionViewer
        viewer = RobotMotionViewer(robot_type="unitree_g1")

    while RUNNING:
        time.sleep(2)
        if len(receive_085_linux.data_up_buffer) < 2:
            print("\r 等待接收衣服数据...", end="")
            continue
        if len(receive_085_linux.data_down_buffer) < 2:
            print("\r 等待接收裤子数据...", end="")
            continue
        break

    if args.calibration_trigger_file:
        trigger_path = os.path.abspath(os.path.expanduser(args.calibration_trigger_file))
        logging.info("TPOSE_GATE_READY trigger=%s", trigger_path)
        while RUNNING and not os.path.exists(trigger_path):
            time.sleep(0.1)
        if not RUNNING:
            return
        # 边沿触发语义: 消费即删除。残留文件不会在数据就绪瞬间被误吞
        # (2026-10-28 教训: 触发文件早于闸门就绪存在 → 闸门一开瞬间标定错误姿势)
        try:
            os.remove(trigger_path)
        except OSError:
            pass
        logging.info("TPOSE_TRIGGER_ACCEPTED: starting 3-second calibration countdown")
    elif args.auto_calibrate:
        logging.info("--auto-calibrate: 5 秒后自动开始 T-Pose 校准, 请保持 T-Pose 姿势")
        for i in range(5):
            if not RUNNING or _STOP_EVENT.wait(1):
                return
            print(5 - i, end=" ", flush=True)
        print()
    else:
        try:
            input("数据接收成功, 接下来进行校准数据采集, 请输入任意字符开始")
        except (EOFError, KeyboardInterrupt):
            if not RUNNING:
                return
            logging.warning("无终端输入, 5 秒后自动开始校准, 请保持 T-Pose")
            for i in range(5):
                if not RUNNING or _STOP_EVENT.wait(1):
                    return
                print(5 - i, end=" ", flush=True)
            print()

    for i in range(3):
        if not RUNNING or _STOP_EVENT.wait(1):
            return
        print(3 - i)

    stream_started = False
    start_time = None
    last_status_time = time.perf_counter()
    perf_acc = {
        "preprocess": 0.0,
        "lfp": 0.0,
        "postprocess": 0.0,
        "smplx": 0.0,
        "retarget": 0.0,
        "viewer": 0.0,
        "publish": 0.0,
    }
    perf_count = 0
    input_paused = False

    while RUNNING:
        fc.sleep()

        ready, reason = _ble_input_status()
        if not ready:
            if not input_paused:
                logging.warning("V2 garment stream paused: %s; HMCP publish is held", reason)
                input_paused = True
            continue
        if input_paused:
            logging.info("V2 garment stream recovered; resuming HMCP publish")
            input_paused = False

        if CALIBRATION_DONE is False:
            print("请保持 T-Pose")

            tpose_up, tpose_down = _snapshot_input_frames(history=60)
            tpose_data_clothes = tpose_up.reshape(-1, 6, 7)[
                :, imu_order_clothes, :
            ].mean(axis=0)
            # V2: 每帧 6 槽，裤子取 [0,1,2,4,5] 再按 imu_order 排列
            tpose_data_pant = tpose_down.reshape(-1, 6, 7)[
                :, PANT_SLOTS_V2, :
            ][:, imu_order_pant, :].mean(axis=0)
            tpose_data = np.concatenate([tpose_data_clothes, tpose_data_pant], axis=0)
            tpose_acc = torch.FloatTensor(tpose_data[:, 0:3])
            tpose_q = torch.FloatTensor(tpose_data[:, 3:7])
            tpose_oris = quaternion_to_rotation_matrix(tpose_q)

            tpose_payload = np.array(torch.cat([tpose_acc.reshape(-1), tpose_oris.reshape(-1)], dim=0)).tolist()
            print("校准数据设置中")
            my_server.set_calibrate_data(tpose_payload)

            start_time = time.time()
            CALIBRATION_DONE = True
            continue

        try:
            latest_up, latest_down = _snapshot_input_frames(history=1)
        except (IndexError, RuntimeError):
            if not input_paused:
                logging.warning("V2 garment buffers changed during reconnect; HMCP publish is held")
                input_paused = True
            continue

        data_clothes = latest_up[-1].reshape(6, 7)[imu_order_clothes]
        # V2: 每帧 6 槽，裤子取 [0,1,2,4,5] 再按 imu_order 排列
        data_pant = latest_down[-1].reshape(6, 7)[PANT_SLOTS_V2][imu_order_pant]
        data = np.concatenate([data_clothes, data_pant], axis=0)
        accs = torch.FloatTensor(data[:, 0:3])
        q = torch.FloatTensor(data[:, 3:7])
        oris = quaternion_to_rotation_matrix(q)

        t0 = time.perf_counter()
        payload = np.array(torch.cat([accs.reshape(-1), oris.reshape(-1)], dim=0)).tolist()
        payload = my_server.calibrate(payload)
        my_server.operator(payload)
        data_feed = my_server.to_predict_data()
        t1 = time.perf_counter()

        result = i_session.run(output_names=None, input_feed=data_feed)
        t_lfp = time.perf_counter()
        result = my_server.predict_result(result)
        t2 = time.perf_counter()

        if args.smpl_converter == "fast":
            fast_human = getattr(
                smplexconverter, "convert_axis_angle_to_human_data_fast", None
            )
            if fast_human is not None:
                human_data = fast_human(
                    result["axis_angles"],
                    result["root_translation"],
                    device=args.smpl_fast_device,
                )
                smpl_joints = None
                smpl_pose = None
            else:
                # Target GMR bundles expose the same lightweight FK through
                # a bundle-returning API. Keep both package generations usable.
                fast_bundle = smplexconverter.convert_axis_angle_to_smpl_bundle_fast(
                    result["axis_angles"], result["root_translation"]
                )
                human_data = fast_bundle["human_data"]
                smpl_joints = fast_bundle.get("smpl_joints")
                smpl_pose = fast_bundle.get("smpl_pose")
        else:
            smpl_bundle = smplexconverter.convert_axis_angle_to_smplx_bundle(
                result["axis_angles"], result["root_translation"], body_model
            )
            human_data = smpl_bundle["human_data"]
            smpl_joints = smpl_bundle["smpl_joints"]
            smpl_pose = smpl_bundle["smpl_pose"]
        t3 = time.perf_counter()
        if retarget is not None:
            qpos = retarget.retarget(human_data)
        else:
            qpos = None
        t4 = time.perf_counter()

        if viewer is not None:
            viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retarget.scaled_human_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                rate_limit=False,
            )
        t5 = time.perf_counter()

        current_time = time.time()
        if current_time - start_time > 3:
            joint_pos_il, joint_vel_il = streamer.publish_motion(
                qpos_mj=qpos,
                smpl_joints=smpl_joints,
                smpl_pose=smpl_pose,
            )
            t6 = time.perf_counter()
            # X2 SMPL whole-body: FGP human_data -> pico_intent (OKR 新链)
            if pico_pub is not None and pico_adapter is not None:
                _pico_msg = pico_adapter.publish(human_data, time.perf_counter())
                if _pico_msg is not None:
                    pico_pub.send(_pico_msg)
                if pico_diag_n % 31 == 0:  # ~1 Hz 无条件心跳
                    logging.info("[pico-diag] engaged=%s reason=%s msg=%s human_data=%s",
                                 pico_adapter.engaged, pico_adapter.last_reason,
                                 _pico_msg is not None,
                                 type(human_data).__name__ if human_data is not None else None)
                pico_diag_n += 1
            if not stream_started:
                logging.info(
                    "开始向 %s 持续发送 G1 参考帧 (stream-mode=%s)",
                    "Humanoid-GPT/GR00T" if args.stream_mode == "both" else args.stream_mode,
                    args.stream_mode,
                )
                stream_started = True
            perf_acc["preprocess"] += t1 - t0
            perf_acc["lfp"] += t_lfp - t1
            perf_acc["postprocess"] += t2 - t_lfp
            perf_acc["smplx"] += t3 - t2
            perf_acc["retarget"] += t4 - t3
            perf_acc["viewer"] += t5 - t4
            perf_acc["publish"] += t6 - t5
            perf_count += 1

            now_perf = time.perf_counter()
            if now_perf - last_status_time >= args.log_interval:
                denom = max(perf_count, 1)
                logging.info(
                    "fps=%.2f frame=%d q0=%s v0=%s avg_ms(pre=%.1f lfp=%.1f post=%.1f smpl=%.1f retarget=%.1f viewer=%.1f pub=%.1f)",
                    fc.get_fps(),
                    streamer.frame_index - 1,
                    np.round(joint_pos_il[:3], 3).tolist(),
                    np.round(joint_vel_il[:3], 3).tolist(),
                    1000.0 * perf_acc["preprocess"] / denom,
                    1000.0 * perf_acc["lfp"] / denom,
                    1000.0 * perf_acc["postprocess"] / denom,
                    1000.0 * perf_acc["smplx"] / denom,
                    1000.0 * perf_acc["retarget"] / denom,
                    1000.0 * perf_acc["viewer"] / denom,
                    1000.0 * perf_acc["publish"] / denom,
                )
                last_status_time = now_perf
                perf_count = 0
                for key in perf_acc:
                    perf_acc[key] = 0.0
        else:
            print("校准完成，请保持直立")


def dynamic_calibration(t_gap=1):
    while RUNNING and not _STOP_EVENT.wait(t_gap):
        if CALIBRATION_DONE:
            my_server.auto_calibrate()


def _disconnect_ble_fallback(addresses: tuple[str, ...]) -> None:
    """Best-effort BlueZ disconnect fallback if Bleak client teardown missed a link."""
    for addr in addresses:
        if not addr:
            continue
        try:
            subprocess.run(
                ["bluetoothctl", "disconnect", addr],
                check=False,
                capture_output=True,
                timeout=3.0,
            )
        except Exception:
            pass


def shutdown_garment_session(manager, pico_pub=None, threads=(),
                             ble_addresses: tuple[str, ...] = ()) -> None:
    """Stop worker loops, close BLE clients via Bleak, and release BlueZ links."""
    global RUNNING
    RUNNING = False
    _STOP_EVENT.set()
    if manager is not None and hasattr(manager, "close_all"):
        try:
            manager.close_all()
        except Exception as exc:
            logging.warning("BLE manager close_all error: %s", exc)
    for thread in threads:
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
    if pico_pub is not None:
        try:
            pico_pub.close(0)
        except Exception:
            pass
    if ble_addresses:
        _disconnect_ble_fallback(ble_addresses)


if __name__ == "__main__":
    providers = build_onnx_providers()
    lfp_model = os.path.abspath("./onnx_models/LFP_dense_taichi_ft.onnx")
    lfp_cpu_options = rt.SessionOptions()
    lfp_cpu_options.intra_op_num_threads = 4
    lfp_cpu_options.inter_op_num_threads = 1
    if args.lfp_backend == "soc2":
        from lfp_remote import RemoteLFPSession

        i_session = RemoteLFPSession(
            host=args.lfp_soc2_host,
            port=args.lfp_soc2_port,
            timeout_s=args.lfp_timeout_ms / 1000.0,
            local_model=lfp_model,
            local_providers=["CPUExecutionProvider"],
        )
    elif args.lfp_backend == "local-cpu":
        logging.info("LFP backend=local CPU ONNX Runtime threads=4")
        i_session = rt.InferenceSession(
            lfp_model,
            sess_options=lfp_cpu_options,
            providers=["CPUExecutionProvider"],
        )
    else:
        logging.info("LFP backend=local CUDA ONNX Runtime")
        i_session = rt.InferenceSession(lfp_model, providers=providers)
    calibration_session = rt.InferenceSession("./onnx_models/TIC4Clothes_dense.onnx", providers=providers)

    device_config_clothes = config.device_config.jacket_6IMU
    imu_order_clothes = device_config_clothes["imu_order"]

    device_config_pant = config.device_config.pants_5IMU
    imu_order_pant = device_config_pant["imu_order"]

    my_server = DataProcessServer_FullBody(
        rotation_type=args.rotation_type,
        part=args.part,
        config=[device_config_clothes, device_config_pant],
        mode=demo_mode.FULL,
        track_trans=True,
        calibration_session=calibration_session,
        run_unity_package=False,
        physics_optim=True,
        cali_pose="T",
        beta=None,
    )

    streamer = MotionPublisher(args)

    # X2 SMPL whole-body: pico_intent 直出 (默认关闭, --pico-intent-port 开启)
    pico_adapter = None
    pico_pub = None
    if getattr(args, "pico_intent_port", 0) > 0:
        _gia = _import_garment_intent_adapter()
        pico_adapter = _gia.GarmentIntentAdapter(
            engage_file=os.path.expanduser(args.pico_intent_engage_file),
            stationary_s=args.pico_intent_stationary_s,
        )
        import zmq as _zmq
        pico_pub = _zmq.Context.instance().socket(_zmq.PUB)
        pico_pub.setsockopt(_zmq.LINGER, 0)
        pico_pub.bind(f"tcp://*:{args.pico_intent_port}")
        time.sleep(0.5)
        logging.info(
            "pico_intent PUB tcp://*:%d (engage file: %s, stationary %.1fs)"
            " -> X2 token service",
            args.pico_intent_port, args.pico_intent_engage_file,
            args.pico_intent_stationary_s,
        )

    input_mod = _import_input_layer(args.input)
    if _INPUT_BLE:
        # BLE 直连: 串行连接上衣+裤子并订阅 V2 Notify
        manager = input_mod.BleV2Manager(addr_up=args.addr_up, addr_down=args.addr_down)
        manager.connect_devices(max_devices=2)
        input_mod.start_data_threads(manager, input_mod.data_up_buffer, input_mod.data_down_buffer)
    else:
        manager = input_mod.MultiPortManagerV2()
        manager.connect_devices(max_devices=2)
        input_mod.start_data_threads(manager, input_mod.data_up_buffer, input_mod.data_down_buffer)

    ble_addrs = (args.addr_up, args.addr_down) if _INPUT_BLE else ()
    _cleaned = False

    def _cleanup_once():
        global _cleaned
        if _cleaned:
            return
        _cleaned = True
        shutdown_garment_session(manager, pico_pub=pico_pub, threads=threads, ble_addresses=ble_addrs)

    def _handle_signal(signum, _frame):
        global RUNNING
        logging.info("收到退出信号 (%s), 正在关闭衣服采集与蓝牙连接...", signum)
        RUNNING = False
        _STOP_EVENT.set()

    threads = [
        Thread(target=data_transmit, kwargs={"fps": args.fps}, daemon=True),
        Thread(target=dynamic_calibration, kwargs={"t_gap": 2}, daemon=True),
    ]
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    atexit.register(_cleanup_once)
    try:
        for thread in threads:
            thread.start()
        while RUNNING and not _STOP_EVENT.is_set():
            if not any(t.is_alive() for t in threads):
                break
            _STOP_EVENT.wait(0.2)
    except KeyboardInterrupt:
        RUNNING = False
        _STOP_EVENT.set()
    finally:
        _cleanup_once()
