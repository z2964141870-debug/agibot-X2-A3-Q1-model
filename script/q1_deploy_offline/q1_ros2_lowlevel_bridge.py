"""Q1 external 3090 ROS 2 low-level bridge.

The policy/recovery inference cadence is 50 Hz.  HAL commands are repeated at
500 Hz with a monotonic onboard ROS timestamp, as required by the Q1 SDK.
The stand-start path reads hardware feedback and checks commands in memory.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import numpy as np
import rclpy
from aimdk_msgs.msg import JointCommand, JointCommandArray, JointStateArray
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import Imu

from q1_cgfa_obs_np import ContactGatedFootAnchorNP, LivePlannerControllerNP, build_obs_80_np
from q1_fgp_client import FGPHttpSource
from q1_kinematics_np import Q1KinematicsNP, quat_xyzw_to_mat, yaw_angle_from_mat, yaw_mat_np
from q1_recovery_runtime import Q1RecoveryBranch
from q1_soft_recovery_v2_owner import Q1SoftRecoveryV2Owner
from q1_stand_start import Q1StandStart
from q1_v2b_runtime import Q1V2bOnnxController
from replay_q1_walk_start import load_standing_feedback


def build_joint_command(
    joint_state: JointStateArray,
    stamp,
    sequence: int,
    policy_index: dict[str, int],
    target: np.ndarray,
    stiffness: np.ndarray,
    damping: np.ndarray,
) -> JointCommandArray:
    """Keep policy values keyed by name, and serialize in the HAL feedback order."""
    names = [joint.name for joint in joint_state.joints]
    if len(names) != len(policy_index) or set(names) != set(policy_index):
        raise ValueError("Command requires a complete, unique set of HAL joint names")
    msg = JointCommandArray()
    for name in names:
        joint = JointCommand()
        joint.name = name
        msg.joints.append(joint)
    update_joint_command(
        msg, joint_state, stamp, sequence, policy_index, target, stiffness, damping,
    )
    return msg


def update_joint_command(
    command: JointCommandArray,
    joint_state: JointStateArray,
    stamp,
    sequence: int,
    policy_index: dict[str, int],
    target: np.ndarray,
    stiffness: np.ndarray,
    damping: np.ndarray,
) -> None:
    """Refresh one command in place, preserving the HAL order from its feedback."""
    if len(command.joints) != len(joint_state.joints) or any(
        item.name != feedback.name for item, feedback in zip(command.joints, joint_state.joints)
    ):
        raise ValueError("HAL feedback order changed")
    command.header.stamp = stamp
    command.header.frame_id = joint_state.header.frame_id
    command.header.meas_stamp = joint_state.header.meas_stamp
    command.header.sequence = sequence
    for item in command.joints:
        i = policy_index[item.name]
        item.position = float(target[i])
        item.stiffness = float(stiffness[i])
        item.damping = float(damping[i])


class Q1LowLevelBridge(Node):
    def __init__(
        self, bundle_root: Path, *, enable_recovery: bool, publish_hz: float = 500.0,
        stand_readonly: bool = False, stand_start: bool = False,
        stand_reference: Path | None = None, stand_seconds: float = 0.6,
        run_duration: float | None = None, planner_command=(0.0, 0.0, 0.0, 1.0, 0.04),
        human_smpl: bool = False, fgp_url: str | None = None,
        fgp_standing_height: float | None = None, fgp_readonly: bool = False,
        recovery_offline: bool = False, switchable_teleop: bool = False,
    ):
        super().__init__("q1_external_lowlevel_bridge")
        self.root = bundle_root.resolve()
        self.cfg = json.loads((self.root / "config/q1_deploy_config.json").read_text(encoding="utf-8"))
        self.names = list(self.cfg["joint_names"])
        self.index = {name: i for i, name in enumerate(self.names)}
        self.default_q = np.asarray(self.cfg["default_q"], dtype=np.float32)
        self.kp = np.asarray(self.cfg["kp"], dtype=np.float32)
        self.kd = np.asarray(self.cfg["kd"], dtype=np.float32)
        self.eff = np.asarray(self.cfg["effort_limit_nm"], dtype=np.float32)
        self.action_scale = np.asarray(self.cfg["action_scale"], dtype=np.float32)
        self.publish_hz = float(publish_hz)
        self.period = 1.0 / self.publish_hz
        self.infer_divider = max(1, round(self.publish_hz / float(self.cfg["ctrl_hz"])))
        self.enable_recovery = bool(enable_recovery)
        recovery_cfg = self.cfg["recovery"]
        self.recovery_v2 = self.enable_recovery and recovery_cfg.get("owner_class") == "Q1SoftRecoveryV2Owner"
        self.recovery_offline = recovery_offline
        if recovery_offline and not self.recovery_v2:
            raise ValueError("Offline recovery requires a Recovery V2 bundle")
        if self.recovery_v2 and self.publish_hz != recovery_cfg["sender_ramp_hz"]:
            raise ValueError("Recovery V2 requires the configured sender cadence")
        self.human_smpl = bool(human_smpl)
        self.switchable_teleop = bool(switchable_teleop)
        self.fgp_readonly = bool(fgp_readonly)
        self.readonly = bool(stand_readonly or fgp_readonly)
        if self.human_smpl and not fgp_url:
            raise ValueError("human_smpl mode requires --fgp-url")
        if self.switchable_teleop and not self.human_smpl:
            raise ValueError("Switchable teleop requires the human_smpl input")

        if self.human_smpl:
            modalities = ("human_smpl", "planner_cmd")
        elif stand_readonly or stand_start:
            modalities = ("robot_pose",)
        else:
            modalities = ("planner_cmd",)
        self.tracker = Q1V2bOnnxController(self.root, modalities=modalities)
        self.stand_start = bool(stand_start)
        self.stand_seconds = float(stand_seconds)
        self.run_duration = run_duration
        self.planner_command_lock = threading.Lock()
        self.planner_command = tuple(float(value) for value in planner_command)
        self.stand_reference = self._load_stand_reference(stand_reference) if stand_start else None
        self.kin = Q1KinematicsNP(self.root / "robot_model/supermimic.xml")
        self.cgfa = ContactGatedFootAnchorNP(self.kin, dt=float(self.cfg["dt_s"]))
        self.planner_stream = LivePlannerControllerNP(
            dt=float(self.cfg["dt_s"]),
            num_frames=int(self.cfg["hist_len"]),
            smooth_slew=True,
        )
        self.fgp = (
            FGPHttpSource(
                fgp_url,
                dt=float(self.cfg["dt_s"]),
                standing_height_m=fgp_standing_height,
            )
            if self.human_smpl else None
        )
        self.fgp_segment = None
        self.fgp_heading = None
        self.fgp_origin = None
        self.recovery = None
        if self.enable_recovery:
            expert = self.root / "onnx" / recovery_cfg["expert_file"]
            xml = self.root / "robot_model/supermimic.xml"
            if self.recovery_v2:
                self.recovery = Q1SoftRecoveryV2Owner(
                    self.tracker, expert, xml,
                    blend_ticks=recovery_cfg["blend_ticks"],
                    leg_speed_cap=recovery_cfg["leg_speed_cap"],
                    leg_brake_gain=recovery_cfg["leg_brake_gain"],
                )
                self.sender_increment = recovery_cfg["sender_max_increment_rad_per_2ms"]
            else:
                self.recovery = Q1RecoveryBranch(self.tracker, expert, xml)
        if self.human_smpl and self.enable_recovery and not self.recovery_v2:
            raise ValueError("FGP recovery requires a Recovery V2 bundle")

        qos_state = QoSProfile(depth=20)
        qos_state.reliability = ReliabilityPolicy.BEST_EFFORT
        qos_state.durability = DurabilityPolicy.VOLATILE
        qos_state.history = HistoryPolicy.KEEP_LAST
        qos_cmd = QoSProfile(depth=10)
        qos_cmd.reliability = ReliabilityPolicy.BEST_EFFORT
        qos_cmd.durability = DurabilityPolicy.VOLATILE
        qos_cmd.history = HistoryPolicy.KEEP_LAST
        self.create_subscription(Imu, "/aima/hal/imu/state", self._on_imu, qos_state)
        self.create_subscription(JointStateArray, "/aima/hal/joint/state", self._on_joint, qos_state)
        self.publisher = None
        if not self.readonly:
            self.publisher = self.create_publisher(JointCommandArray, "/aima/hal/joint/command", qos_cmd)

        self.lock = threading.Lock()
        self.mode_lock = threading.Lock()
        self.active_mode = "planner" if self.switchable_teleop else (
            "human_smpl" if self.human_smpl else "planner"
        )
        self.pending_mode: str | None = None
        self.imu = None
        self.joint = None
        self.last_imu_wall = 0.0
        self.last_joint_wall = 0.0
        self.command_state = None
        self.sent_active_command = False
        self.target = self.default_q.copy()
        self.last_target = self.default_q.copy()
        self.desired_target = self.default_q.copy()
        self.prev_action = np.zeros(22, dtype=np.float32)
        self.R_ref = np.eye(3, dtype=np.float32)
        self.initialized = False
        self.sequence = 0
        self.stop_event = threading.Event()
        self.control_thread = threading.Thread(target=self._control_loop, name="q1_500hz", daemon=True)
        self.stand_controller = None
        self.started_at = None

    def set_planner_command(self, command) -> None:
        """Replace the planner command atomically while the 500 Hz loop is running."""
        values = tuple(float(value) for value in command)
        if len(values) != 5 or not np.isfinite(values).all():
            raise ValueError("Planner command must contain five finite values")
        with self.planner_command_lock:
            self.planner_command = values

    def get_planner_command(self):
        with self.planner_command_lock:
            return self.planner_command

    def get_control_mode(self) -> str:
        with self.mode_lock:
            return self.active_mode

    def request_control_mode(self, mode: str) -> bool:
        if mode not in ("planner", "human_smpl"):
            raise ValueError(f"Unsupported control mode: {mode}")
        if mode == "human_smpl":
            if self.fgp is None:
                self.get_logger().error("Teleop mode requires an FGP source")
                return False
            status = self.fgp.status()
            if not status["fresh"]:
                self.get_logger().warning(
                    "Teleop mode refused: FGP input is stale or unavailable"
                )
                return False
        with self.mode_lock:
            self.pending_mode = mode
        self.get_logger().info(f"Requested control mode: {mode}")
        return True

    def _take_pending_mode(self) -> str | None:
        with self.mode_lock:
            mode = self.pending_mode
            self.pending_mode = None
            return mode

    def _apply_pending_mode(self, R, omega, q, qd, queued, live_fgp: bool) -> None:
        requested = self._take_pending_mode()
        if requested is None or requested == self.get_control_mode():
            return
        if requested == "human_smpl":
            if queued is None or not live_fgp:
                self.get_logger().warning("Teleop mode deferred: no fresh FGP frame available")
                with self.mode_lock:
                    self.pending_mode = requested
                return
            self._reset_fgp_segment(R, omega, q, qd, queued.tick)
            self.tracker.reset(q)
        else:
            self._reanchor_reference(R, omega, q, qd)
            self.tracker.reset(q)
        with self.mode_lock:
            self.active_mode = requested
        self.get_logger().info(f"Control mode active: {requested}")

    def _load_stand_reference(self, path: Path | None) -> np.ndarray:
        source = path or self.root / "real_logs/q1_official_standup_faceup_20261120.npz"
        with np.load(source, allow_pickle=False) as data:
            names = data["joint_names"].tolist()
            q = np.asarray(data["q"], dtype=np.float32)
        if len(names) != len(self.names) or set(names) != set(self.names):
            raise ValueError("Standing reference has an unexpected joint-name set")
        order = [names.index(name) for name in self.names]
        return np.median(q[:, order], axis=0).astype(np.float32)

    def start(self) -> None:
        if self.fgp is not None:
            self.fgp.start()
        self.control_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)
        if self.fgp is not None:
            self.fgp.stop()
        if self.sent_active_command and self.publisher is not None and rclpy.ok():
            for _ in range(5):
                self._publish_damping()
                time.sleep(self.period)

    def _on_imu(self, msg: Imu) -> None:
        with self.lock:
            self.imu = msg
            self.last_imu_wall = time.monotonic()

    def _on_joint(self, msg: JointStateArray) -> None:
        with self.lock:
            self.joint = msg
            self.last_joint_wall = time.monotonic()

    def _snapshot(self):
        with self.lock:
            return self.imu, self.joint, min(self.last_imu_wall, self.last_joint_wall)

    def _decode_state(self, imu_msg: Imu, joint_msg: JointStateArray):
        q = np.full(22, np.nan, dtype=np.float32)
        qd = np.full(22, np.nan, dtype=np.float32)
        effort = np.full(22, np.nan, dtype=np.float32)
        seen = set()
        for item in joint_msg.joints:
            if item.name not in self.index or item.name in seen:
                return None
            i = self.index[item.name]
            q[i] = float(item.position)
            qd[i] = float(item.velocity)
            effort[i] = float(item.effort)
            seen.add(item.name)
        if len(seen) != 22 or not np.isfinite(q).all() or not np.isfinite(qd).all():
            return None
        quat = np.array(
            [imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w],
            dtype=np.float32,
        )
        omega = np.array([imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z], dtype=np.float32)
        if not np.isfinite(quat).all() or not np.isfinite(omega).all():
            return None
        if abs(np.linalg.norm(quat) - 1.0) > 0.02:
            return None
        return quat_xyzw_to_mat(quat), omega, q, qd, effort

    def run_stand_readonly(self, recording: Path, transition_s: float, duration: float) -> dict:
        q_reference, _, _, _, _ = load_standing_feedback(recording, self.tracker)
        standing_q = np.median(q_reference, axis=0)
        dt = self.tracker.cfg["dt_s"]
        cmd_topic = "/aima/hal/joint/command"
        ready_count = duplicate_reads = 0
        max_publishers = 0
        previous_stamps = (0, 0)
        controller = None
        report = {"steps": 0, "serialized_frames": 0, "ready_ticks": 0}
        deadline = time.monotonic() + duration
        next_tick = time.monotonic()
        last_advance = next_tick
        command = None
        while rclpy.ok() and time.monotonic() < deadline:
            now = time.monotonic()
            if now < next_tick:
                rclpy.spin_once(self, timeout_sec=next_tick - now)
                continue
            next_tick = now + dt
            max_publishers = max(max_publishers, self.count_publishers(cmd_topic))
            imu, joint, last_received = self._snapshot()
            age = now - last_received
            state = self._decode_state(imu, joint) if imu is not None and joint is not None and age <= 0.04 else None
            if state is None:
                ready_count = 0
                if controller is not None:
                    report["stopped"] = "Stale or invalid feedback"
                    break
                continue

            stamps = tuple(
                msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
                for msg in (imu, joint)
            )
            if any(current < previous for current, previous in zip(stamps, previous_stamps)):
                report["stopped"] = "Feedback timestamp moved backwards"
                break
            if any(current == previous for current, previous in zip(stamps, previous_stamps)):
                duplicate_reads += 1
                ready_count = 0
                if controller is not None and now - last_advance > 0.04:
                    report["stopped"] = "Feedback source timestamps stopped advancing"
                    break
                continue
            previous_stamps = stamps
            last_advance = now
            ready_count += 1
            if ready_count < 5 and controller is None:
                continue

            rotation, omega, q, qd, _ = state
            try:
                if controller is None:
                    controller = Q1StandStart(
                        self.tracker, self.kin, q, rotation, standing_q, transition_s=transition_s,
                    )
                target, info = controller.step(rotation, omega, q, qd)
            except ValueError as error:
                report["stopped"] = str(error)
                break
            stamp = self.get_clock().now().to_msg()
            if command is None:
                command = build_joint_command(
                    joint, stamp, report["steps"], self.index, target, self.kp, self.kd,
                )
            else:
                try:
                    update_joint_command(
                        command, joint, stamp, report["steps"], self.index, target, self.kp, self.kd,
                    )
                except ValueError as error:
                    report["stopped"] = str(error)
                    break
            decoded = deserialize_message(serialize_message(command), JointCommandArray)
            if (
                decoded.header.frame_id != joint.header.frame_id
                or decoded.header.meas_stamp != joint.header.meas_stamp
                or decoded.header.sequence != report["steps"]
                or len(decoded.joints) != len(self.names)
                or any(
                    actual.name != feedback.name
                    or actual.position != float(target[self.index[feedback.name]])
                    or actual.stiffness != float(self.kp[self.index[feedback.name]])
                    or actual.damping != float(self.kd[self.index[feedback.name]])
                    for actual, feedback in zip(decoded.joints, joint.joints)
                )
            ):
                report["stopped"] = "Command layout verification failed"
                break
            if not report["steps"]:
                report["initial_tilt_deg"] = round(info["tilt_deg"], 3)
                report["first_target_error_rad"] = round(info["target_error_rad"], 5)
                report["hal_order_differs_from_policy"] = [j.name for j in joint.joints] != self.names
            report["steps"] += 1
            report["serialized_frames"] += 1
            report["controller_ready_ticks"] = report.get("controller_ready_ticks", 0) + int(info["controller_ready"])
            report["load_release_ready_ticks"] = report.get("load_release_ready_ticks", 0) + int(info["load_release_ready"])
            for key in ("target_step_rad", "target_projection_rad", "target_error_rad", "peak_estimated_pd_nm"):
                report["max_" + key] = round(max(report.get("max_" + key, 0.0), info[key]), 5)
            report["last_tilt_deg"] = round(info["tilt_deg"], 3)
            report["last_stand_reference_error_rad"] = round(info["stand_reference_error_rad"], 5)
            report["max_feedback_age_ms"] = round(max(report.get("max_feedback_age_ms", 0.0), age * 1e3), 3)
        report["command_publishers"] = max(max_publishers, self.count_publishers(cmd_topic))
        report["command_subscribers"] = self.count_subscribers(cmd_topic)
        report["duplicate_feedback_reads"] = duplicate_reads
        return report

    def _reanchor_reference(self, R, omega, q, qd) -> None:
        self.R_ref = R.copy()
        self.cgfa.reset(R, omega, q, qd, np.zeros(2, dtype=np.float32), R_ref=self.R_ref)
        self.planner_stream.reset(yaw0_rad=yaw_angle_from_mat(R))
        self.fgp_segment = None

    def _reset_anchor(self, R: np.ndarray, omega: np.ndarray, q: np.ndarray, qd: np.ndarray) -> None:
        self._reanchor_reference(R, omega, q, qd)
        self.tracker.reset(q)
        if self.recovery is not None:
            self.recovery.reset()
        self.prev_action.fill(0.0)
        self.last_target = q.copy()
        self.target = q.copy()
        self.desired_target = q.copy()
        self.initialized = True
        if self.stand_start:
            self.stand_controller = Q1StandStart(
                self.tracker, self.kin, q, R, self.stand_reference,
                transition_s=self.stand_seconds,
            )

    def _reset_fgp_segment(
        self, R: np.ndarray, omega: np.ndarray, q: np.ndarray, qd: np.ndarray, tick,
        *, reset_tracker: bool = True,
    ) -> None:
        self.fgp_segment = int(tick.segment)
        self.fgp_heading = yaw_mat_np(R) @ yaw_mat_np(tick.root_rotation).T
        self.fgp_origin = np.asarray(tick.root_xy, dtype=np.float32).copy()
        p_ref_xy = self.fgp_heading[:2, :2] @ (tick.root_xy - self.fgp_origin)
        r_ref = self.fgp_heading @ tick.root_rotation
        self.R_ref = r_ref.astype(np.float32)
        self.cgfa.reset(R, omega, q, qd, p_ref_xy, R_ref=self.R_ref)
        if reset_tracker:
            self.tracker.reset(q)

    def _fgp_reference(self, tick):
        p_ref_xy = self.fgp_heading[:2, :2] @ (tick.root_xy - self.fgp_origin)
        r_ref = self.fgp_heading @ tick.root_rotation
        return p_ref_xy.astype(np.float32), r_ref.astype(np.float32)

    def _base_step(self, R: np.ndarray, omega: np.ndarray, q: np.ndarray, qd: np.ndarray, *, neutral=False):
        command = (0.0, 0.0, 0.0, 1.0, 0.04) if neutral else self.get_planner_command()
        planner, p_ref_xy, planner_R_ref = self.planner_stream.push_and_build(
            *command,
            cgfa=self.cgfa,
            R_imu_aligned=R,
        )
        self.R_ref = planner_R_ref
        exy, _ = self.cgfa.step(R, omega, q, qd, p_ref_xy, R_ref=planner_R_ref)
        obs = build_obs_80_np(
            q=q,
            qd=qd,
            default_q=self.default_q,
            R_imu=R,
            w_body=omega,
            a_prev=self.prev_action,
            R_ref=planner_R_ref,
            cgfa_exy=exy,
        )
        return self.tracker.step(obs, planner, q, qd, modality="planner_cmd")

    def _human_smpl_step(self, R, omega, q, qd, tick):
        if tick.segment != self.fgp_segment:
            self._reset_fgp_segment(R, omega, q, qd, tick)
        p_ref_xy, r_ref = self._fgp_reference(tick)
        self.R_ref = r_ref
        exy, _ = self.cgfa.step(R, omega, q, qd, p_ref_xy, R_ref=r_ref)
        obs = build_obs_80_np(
            q=q,
            qd=qd,
            default_q=self.default_q,
            R_imu=R,
            w_body=omega,
            a_prev=self.prev_action,
            R_ref=r_ref,
            cgfa_exy=exy,
        )
        return self.tracker.step(obs, tick.feature, q, qd, modality="human_smpl")

    def _infer_once(self, decoded) -> None:
        R, omega, q, qd, _ = decoded
        if not self.initialized:
            self._reset_anchor(R, omega, q, qd)

        if self.stand_start:
            target, info = self.stand_controller.step(R, omega, q, qd)
            self.target = np.asarray(target, dtype=np.float32)
            self.last_target = self.target.copy()
            if self.sequence % 50 == 0:
                self.get_logger().info(
                    f"stand_start tilt={info['tilt_deg']:.2f} "
                    f"target_error={info['target_error_rad']:.3f} "
                    f"release_ready={info['load_release_ready']}"
                )
            return

        queued, _ = self.fgp.get_latest() if self.fgp is not None else (None, 0)
        now_ns = time.monotonic_ns()
        age_limit = 100_000_000 if self.recovery_v2 else 150_000_000
        live_fgp = queued is not None and 0 <= now_ns - queued.arrival_ns <= age_limit
        self._apply_pending_mode(R, omega, q, qd, queued, live_fgp)
        active_mode = self.get_control_mode()

        def base_step():
            if active_mode == "human_smpl" and live_fgp:
                return self._human_smpl_step(R, omega, q, qd, queued.tick)
            if active_mode == "planner" and self.fgp_segment is not None:
                self._reanchor_reference(R, omega, q, qd)
                self.tracker.reset(q)
            return self._base_step(
                R,
                omega,
                q,
                qd,
                neutral=(active_mode == "human_smpl" or self.recovery_offline),
            )

        if self.recovery_v2:
            def handback_anchor():
                if live_fgp:
                    self._reset_fgp_segment(R, omega, q, qd, queued.tick, reset_tracker=False)
                else:
                    self._reanchor_reference(R, omega, q, qd)

            stamp_ns = queued.arrival_ns if live_fgp else None if self.human_smpl else now_ns
            stamp = None if self.recovery_offline or stamp_ns is None else stamp_ns * 1e-9
            target, _, gate = self.recovery.step(
                R, omega, q, qd, self.prev_action, base_step, handback_anchor,
                sample_time_s=now_ns * 1e-9, teleop_stamp_s=stamp,
            )
        elif self.recovery is not None:
            target, _, gate = self.recovery.step(
                R, omega, q, qd, self.prev_action, base_step,
                lambda: self._reset_anchor(R, omega, q, qd),
            )
        elif self.fgp_readonly and not live_fgp:
            return
        else:
            target, _, _ = base_step()
            gate = 0.0

        target = np.asarray(target, dtype=np.float32)
        if self.recovery_v2:
            if not np.isfinite(target).all():
                raise ValueError("Nonfinite Recovery V2 target")
            self.desired_target = np.clip(target, self.default_q - 2.0, self.default_q + 2.0)
            return
        target = np.clip(target, self.last_target - 0.2, self.last_target + 0.2)
        target = np.clip(target, self.default_q - 2.0, self.default_q + 2.0)
        self.target = target
        self.last_target = target.copy()
        self.prev_action = (target - self.default_q) / self.action_scale
        if self.sequence % 50 == 0:
            position_error = float(np.max(np.abs(target - q)))
            self.get_logger().info(f"inference gate={gate:.2f} max_target_error_rad={position_error:.3f}")

    def _next_sender_target(self):
        increment = np.clip(self.desired_target - self.target, -self.sender_increment, self.sender_increment)
        return self.target + increment

    def _command(self, target: np.ndarray, stiffness: np.ndarray, damping: np.ndarray) -> JointCommandArray:
        return build_joint_command(
            self.command_state, self.get_clock().now().to_msg(), self.sequence,
            self.index, target, stiffness, damping,
        )

    def _publish_damping(self) -> None:
        msg = self._command(self.target, np.zeros(22, dtype=np.float32), np.full(22, 5.0, dtype=np.float32))
        self.publisher.publish(msg)
        self.sequence += 1

    def _control_loop(self) -> None:
        tick = 0
        self.started_at = time.monotonic()
        next_tick = time.monotonic()
        while not self.stop_event.is_set():
            if self.run_duration is not None and time.monotonic() - self.started_at >= self.run_duration:
                self.stop_event.set()
                return
            now = time.monotonic()
            if now < next_tick:
                time.sleep(next_tick - now)
            # Avoid emitting a burst of backlogged commands after an overrun.
            next_tick += self.period
            if self.recovery_v2 and next_tick <= time.monotonic():
                next_tick = time.monotonic() + self.period
            imu_msg, joint_msg, last_wall = self._snapshot()
            fresh = imu_msg is not None and joint_msg is not None and (time.monotonic() - last_wall) <= 0.08
            decoded = self._decode_state(imu_msg, joint_msg) if fresh else None
            if decoded is None:
                if self.sent_active_command:
                    print("q1_lowlevel_bridge: feedback became stale after engagement", flush=True)
                    self._publish_damping()
                    self.get_logger().error("State lost after engagement. Stopping control.")
                    self.stop_event.set()
                    return
            else:
                self.command_state = joint_msg
                if not self.initialized or tick % self.infer_divider == 0:
                    try:
                        self._infer_once(decoded)
                    except Exception as error:
                        if self.recovery_v2:
                            self.get_logger().error(f"Recovery V2 inference failed: {error}")
                            self.stop_event.set()
                            return
                        _, _, q, qd, _ = decoded
                        self.target = q.copy()
                        self.last_target = q.copy()
                        self.prev_action = (q - self.default_q) / self.action_scale
                        self.get_logger().error(
                            f"Inference exception, holding measured posture: {error}"
                        )
                        self.initialized = True
                if self.initialized and self.publisher is not None:
                    try:
                        target = self._next_sender_target() if self.recovery_v2 else self.target
                        msg = self._command(target, self.kp, self.kd)
                        self.publisher.publish(msg)
                        if self.recovery_v2:
                            self.target = target
                            self.prev_action = (target - self.default_q) / self.action_scale
                        self.sequence += 1
                        self.sent_active_command = True
                    except Exception as error:
                        print(f"q1_lowlevel_bridge: publish exception: {error!r}", flush=True)
                        self.stop_event.set()
                        return
            tick += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parents[1])
    mode = ap.add_mutually_exclusive_group()
    ap.add_argument("--enable-recovery", action="store_true")
    mode.add_argument("--recovery-offline", action="store_true")
    mode.add_argument("--stand-start-readonly", action="store_true")
    mode.add_argument("--stand-start", action="store_true")
    mode.add_argument("--planner", action="store_true")
    mode.add_argument("--keyboard", action="store_true", help="Interactive planner keyboard control")
    mode.add_argument(
        "--keyboard-switchable",
        action="store_true",
        help="Planner keyboard with s=human_smpl, p=planner, q=exit",
    )
    mode.add_argument("--human-smpl", action="store_true", help="Live FGP human_smpl policy input")
    mode.add_argument("--fgp-readonly", action="store_true", help="Live FGP human_smpl alignment without HAL commands")
    ap.add_argument("--stand-reference", type=Path)
    ap.add_argument("--stand-seconds", type=float, default=0.6)
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--planner-vx", type=float, default=0.0)
    ap.add_argument("--planner-vy", type=float, default=0.0)
    ap.add_argument("--planner-wz", type=float, default=0.0)
    ap.add_argument("--planner-z-ratio", type=float, default=1.0)
    ap.add_argument("--planner-reach-x", type=float, default=0.04)
    ap.add_argument("--keyboard-vx", type=float, default=0.25,
                    help="Keyboard forward/backward speed in m/s")
    ap.add_argument("--keyboard-vy", type=float, default=0.05,
                    help="Conservative keyboard left/right speed in m/s")
    ap.add_argument("--keyboard-wz", type=float, default=0.60,
                    help="Keyboard turn speed in rad/s")
    ap.add_argument("--keyboard-z-ratio", type=float, default=1.0)
    ap.add_argument("--keyboard-reach-x", type=float, default=0.04)
    ap.add_argument("--fgp-url", type=str)
    ap.add_argument("--fgp-standing-height", type=float)
    args = ap.parse_args()
    if not any((args.enable_recovery, args.recovery_offline, args.stand_start, args.stand_start_readonly,
                args.planner, args.keyboard, args.keyboard_switchable, args.human_smpl, args.fgp_readonly)):
        ap.error("Select a control or read-only mode")
    if args.enable_recovery and (args.stand_start or args.stand_start_readonly or args.fgp_readonly or args.keyboard_switchable):
        ap.error("--enable-recovery requires a control mode")
    if args.human_smpl or args.fgp_readonly or args.keyboard_switchable:
        if not args.fgp_url:
            ap.error("--human-smpl/--fgp-readonly/--keyboard-switchable requires --fgp-url")
    elif args.fgp_url:
        ap.error("--fgp-url requires --human-smpl or --fgp-readonly")
    if (args.stand_start_readonly or args.stand_start) and (
        not np.isfinite(args.stand_seconds) or args.stand_seconds <= 0
        or not np.isfinite(args.duration) or args.duration < 0
    ):
        ap.error("Stand transition must be positive and duration must be nonnegative")
    rclpy.init()
    node = None
    try:
        node = Q1LowLevelBridge(
            args.bundle, enable_recovery=(args.enable_recovery or args.recovery_offline),
            recovery_offline=args.recovery_offline,
            stand_readonly=args.stand_start_readonly,
            stand_start=args.stand_start,
            stand_reference=args.stand_reference,
            stand_seconds=args.stand_seconds,
            run_duration=(
                args.duration
                if (args.stand_start or args.fgp_readonly) and args.duration > 0
                else None
            ),
            human_smpl=(args.human_smpl or args.fgp_readonly or args.keyboard_switchable),
            fgp_url=args.fgp_url,
            fgp_standing_height=args.fgp_standing_height,
            fgp_readonly=args.fgp_readonly,
            switchable_teleop=args.keyboard_switchable,
            planner_command=(
                0.0 if args.keyboard or args.keyboard_switchable else args.planner_vx,
                0.0 if args.keyboard or args.keyboard_switchable else args.planner_vy,
                0.0 if args.keyboard or args.keyboard_switchable else args.planner_wz,
                args.keyboard_z_ratio if args.keyboard else args.planner_z_ratio,
                args.keyboard_reach_x if args.keyboard else args.planner_reach_x,
            ),
        )
        if args.stand_start_readonly:
            recording = args.stand_reference or args.bundle / "real_logs/q1_official_standup_faceup_20261120.npz"
            report = node.run_stand_readonly(recording, args.stand_seconds, args.duration)
            print(json.dumps(report, ensure_ascii=False))
            return 0 if report["steps"] and "stopped" not in report else 1
        node.start()
        if args.keyboard or args.keyboard_switchable:
            if not sys.stdin.isatty():
                raise RuntimeError("--keyboard requires an interactive terminal")
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            key_commands = {
                "w": (args.keyboard_vx, 0.0, 0.0),
                "a": (0.0, args.keyboard_vy, 0.0),
                "d": (0.0, -args.keyboard_vy, 0.0),
                "e": (0.0, 0.0, -args.keyboard_wz),
            } if args.keyboard_switchable else {
                "w": (args.keyboard_vx, 0.0, 0.0),
                "s": (-args.keyboard_vx, 0.0, 0.0),
                "a": (0.0, args.keyboard_vy, 0.0),
                "d": (0.0, -args.keyboard_vy, 0.0),
                "q": (0.0, 0.0, args.keyboard_wz),
                "e": (0.0, 0.0, -args.keyboard_wz),
            }
            if args.keyboard_switchable:
                print(
                    "Switchable planner active: w/a/d/e planner, s=human_smpl, "
                    "p=planner, space=stop, q=exit",
                    flush=True,
                )
            else:
                print(
                    "Keyboard planner active: w/s forward-back, a/d left-right, "
                    "q/e turn, space stop, x or Ctrl-C exit",
                    flush=True,
                )
            try:
                tty.setcbreak(fd)
                while rclpy.ok() and not node.stop_event.is_set():
                    rclpy.spin_once(node, timeout_sec=0.01)
                    readable, _, _ = select.select([fd], [], [], 0.0)
                    if not readable:
                        continue
                    key = os.read(fd, 1).decode("utf-8", errors="ignore").lower()
                    if args.keyboard_switchable and key == "s":
                        node.request_control_mode("human_smpl")
                        continue
                    if args.keyboard_switchable and key == "p":
                        node.request_control_mode("planner")
                        continue
                    if (args.keyboard_switchable and key == "q") or (not args.keyboard_switchable and key in ("x", "\x03")):
                        node.stop_event.set()
                        break
                    if key in (" ", "0"):
                        node.set_planner_command((0.0, 0.0, 0.0, args.keyboard_z_ratio, args.keyboard_reach_x))
                        print("planner_cmd = stop", flush=True)
                        continue
                    if key in key_commands:
                        vx, vy, wz = key_commands[key]
                        node.set_planner_command((vx, vy, wz, args.keyboard_z_ratio, args.keyboard_reach_x))
                        print(f"planner_cmd = vx={vx:.3f} vy={vy:.3f} wz={wz:.3f}", flush=True)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                print("Keyboard planner stopped", flush=True)
        else:
            while rclpy.ok() and not node.stop_event.is_set():
                rclpy.spin_once(node, timeout_sec=0.01)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.stop()
            node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
