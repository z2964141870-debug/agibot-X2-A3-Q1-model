"""Human policy wrapper using the unmodified pinned official PolicyRunner."""
from __future__ import annotations

import json
import time
import numpy as np

from .artifacts import activate, sha
from .core import DT_NS, HumanBuffer, array, pelvis_from_sensor, unit


class LiveReference:
    def __init__(self, robot):
        self.robot, self.frames, self.turn = robot, None, None

    def reset(self):
        self.turn = None

    def window(self, step_id, quat, q):
        from deploy.sim2sim.human import human_heading, turn_human
        if self.frames is None:
            raise ValueError("human reference not ready")
        if self.turn is None:
            w, x, y, z = self.robot.slot0_quat(quat, q)
            heading = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
            self.turn = heading-human_heading(self.frames[0])
        return turn_human(self.frames, self.turn, np.zeros(2))


class Runtime:
    """Produces advisory target messages only. No ROS/AimDK imports or HAL publisher."""
    def __init__(self, root=None, *, human_points="full", max_interp_gap_ns=2*DT_NS, max_age_ns=3_000_000_000):
        if human_points not in ("full", "five"):
            raise ValueError("human_points must be 'full' or 'five'")
        self.root = activate(root)
        self.human_points = human_points
        from deploy.sim2sim.obs import Robot
        from deploy.sim2sim.policy_node import PolicyRunner
        self.doc = json.loads((self.root/"assets/morphmimic_policy/x2.json").read_text())
        robot = Robot.from_dict(self.doc["robot"])
        # Robot dataclass is frozen; names are provided to the separate sensor adapter.
        self.robot = robot
        object.__setattr__(robot, "tree_names", tuple(b["name"] for b in self.doc["robot"]["fk"]["bodies"]))
        self.reference = LiveReference(robot)
        self.runner = PolicyRunner(self.root/"assets/morphmimic_policy", "x2", self.reference,
                                   interface="human", human_points=human_points)
        self.buffer = HumanBuffer(sha(self.root/"upstream/deploy/vr_teleop/human_skeleton.json"),
                                  max_age_ns=max_age_ns, max_interp_gap_ns=max_interp_gap_ns)
        self.step_id, self.last_state_sequence = 0, -1
        self.receiver_clock = None

    def reset(self):
        self.buffer.reset()
        self.runner.reset()
        self.reference.reset()
        self.runner.prev_step = -1
        self.step_id, self.last_state_sequence = 0, -1
        self.receiver_clock = None

    def push(self, packet):
        self.buffer.push(packet)

    def state(self, p, now_ns, receiver_clock):
        t0 = time.perf_counter_ns()
        if p.get("kind") != "x2_measurement" or p.get("version") != 1:
            raise ValueError("expected x2_measurement v1")
        if type(p.get("sequence")) is not int or p["sequence"] <= self.last_state_sequence:
            raise ValueError("state sequence must increase; reset explicitly after reconnect")
        if p.get("clock_id") != receiver_clock or type(p.get("received_ns")) is not int:
            raise ValueError("state receive timestamp must use the local receiver clock")
        if p["received_ns"] > now_ns or now_ns-p["received_ns"] > 3_000_000_000:
            raise ValueError("state receive timestamp unavailable/stale")
        if self.receiver_clock is not None and self.receiver_clock != receiver_clock:
            raise ValueError("receiver clock changed: reset required")
        names = p.get("joint_names", [])
        if len(names) != 31 or len(set(names)) != 31 or set(names) != set(self.robot.native_joint_names):
            raise ValueError("joint names must exactly cover the exported X2 Ultra 31 joints")
        order = [names.index(n) for n in self.robot.native_joint_names]
        q = array(p["q_rad"], (31,), "q_rad")[order]
        dq = array(p["dq_rad_s"], (31,), "dq_rad_s")[order]
        sq = unit(array(p["imu_quat_wxyz"], (4,), "imu_quat_wxyz"))
        sg = array(p["imu_gyro_rad_s"], (3,), "imu_gyro_rad_s")
        if p.get("imu_body") not in self.robot.tree_names or "body_from_sensor_wxyz" not in p:
            raise ValueError("explicit IMU body and sensor extrinsic required")
        extrinsic = unit(array(p["body_from_sensor_wxyz"], (4,), "body_from_sensor"))
        quat, gyro = pelvis_from_sensor(self.robot, q, dq, sq, sg, p["imu_body"], extrinsic)
        window, timing = self.buffer.window(now_ns, receiver_clock)
        self.reference.frames = window
        self.last_state_sequence, self.receiver_clock = p["sequence"], receiver_clock
        t1 = time.perf_counter_ns()
        targets = self.runner.step(self.step_id, quat, gyro, q, dq)
        self.step_id += 1
        if targets.shape != (31,) or not np.isfinite(targets).all():
            raise RuntimeError("non-finite or wrong shape target")
        t2 = time.perf_counter_ns()
        return dict(kind="x2_morph_advisory_target", version=1, ready=True,
                    publish_allowed=False, state_sequence=p["sequence"],
                    policy_sha256="cf95194793e4732447f66432fc2381d82dd7c3d82c94da15ad074f2cfe295512",
                    robot_profile="morphmimic_v01_x2_ultra",
                    human_points=self.human_points,
                    joint_names=list(self.robot.native_joint_names), q_target_rad=targets.tolist(),
                    canonical_action=self.runner.last_action.tolist(), timing=timing,
                    preprocessing_ms=(t1-t0)/1e6, policy_and_features_ms=(t2-t1)/1e6,
                    full_runtime_ms=(t2-t0)/1e6, imu_reconstructed_to=self.robot.imu_body)

    def handle(self, message, now_ns, receiver_clock):
        """Arrival timestamps are assigned here, not trusted from a remote sender."""
        from .core import NotReady
        kind = message.get("kind")
        if kind == "reset":
            self.reset()
            return dict(ready=False, reset=True, publish_allowed=False)
        if kind == "morph_human_frame":
            packet = dict(message, arrival_ns=now_ns, arrival_clock=receiver_clock)
            self.push(packet)
            return dict(ready=False, accepted_sequence=message["sequence"], publish_allowed=False)
        if kind == "x2_measurement":
            try:
                return self.state(dict(message, received_ns=now_ns, clock_id=receiver_clock), now_ns, receiver_clock)
            except NotReady as exc:
                return dict(ready=False, reason=str(exc), publish_allowed=False)
        raise ValueError("unsupported message kind")


def synthetic_state(runtime, sequence, now_ns, clock="replay_monotonic"):
    """Development fixture only: not measurements and never a robot startup pose."""
    return dict(kind="x2_measurement", version=1, sequence=sequence,
                received_ns=now_ns, clock_id=clock, joint_names=list(runtime.robot.native_joint_names),
                q_rad=runtime.robot.default_joint_pos.tolist(), dq_rad_s=[0.]*31,
                imu_quat_wxyz=[1., 0., 0., 0.], imu_gyro_rad_s=[0., 0., 0.],
                imu_body="pelvis", body_from_sensor_wxyz=[1., 0., 0., 0.])
