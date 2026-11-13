"""Validate feedback and stage native targets without changing policy scaling."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

from script.x2_morph_human.artifacts import MODEL_SHA, check, manifest


GROUPS = {"leg": (0, 12), "waist": (12, 15), "arm": (15, 29), "head": (29, 31)}
SOURCES = {*GROUPS, "imu"}


def vector(value, size, name):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must have {size} entries")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError(f"{name} must contain finite numbers")
    return [float(x) for x in value]


@dataclass(frozen=True)
class Contract:
    names: tuple
    kp: tuple
    kd: tuple
    default: tuple
    imu_bodies: tuple

    @classmethod
    def load(cls, path):
        path = Path(path)
        row = next(r for r in manifest()["files"] if r["path"] == "assets/morphmimic_policy/x2.json")
        check(path, row)
        robot = json.loads(path.read_text())["robot"]
        names = tuple(robot["native_joint_names"])
        if len(names) != 31 or len(set(names)) != 31:
            raise ValueError("export must cover 31 distinct native joints")
        kp = tuple(vector(robot["pd"]["kp"], 31, "model kp"))
        kd = tuple(vector(robot["pd"]["kd"], 31, "model kd"))
        if min(*kp, *kd) < 0:
            raise ValueError("negative model gains")
        return cls(names, kp, kd, tuple(vector(robot["default_joint_pos"], 31, "default")),
                   tuple(b["name"] for b in robot["fk"]["bodies"]))

    def measurement(self, message, max_age_ms):
        if message.get("kind") != "x2_measurement" or type(message.get("version")) is not int or message["version"] != 1:
            raise ValueError("expected x2_measurement v1")
        seq = message.get("sequence")
        if type(seq) is not int or seq < 0:
            raise ValueError("feedback sequence must be a nonnegative integer")
        names = message.get("joint_names")
        if not isinstance(names, list) or len(names) != 31 or any(type(n) is not str for n in names):
            raise ValueError("feedback must name all 31 native joints")
        if len(set(names)) != 31 or set(names) != set(self.names):
            raise ValueError("feedback joint names differ from model export")
        q = vector(message.get("q_rad"), 31, "feedback q")
        dq = vector(message.get("dq_rad_s"), 31, "feedback dq")
        ages = message.get("source_receive_ages_s")
        if not isinstance(ages, dict) or set(ages) != SOURCES:
            raise ValueError("feedback needs all five local receive ages")
        ages = dict(zip(sorted(SOURCES), vector([ages[k] for k in sorted(SOURCES)], 5, "receive ages")))
        if min(ages.values()) < 0 or max(ages.values()) * 1000 > max_age_ms:
            raise ValueError("feedback is stale or from a future receive clock")
        if message.get("imu_body") not in self.imu_bodies:
            raise ValueError("IMU body is absent from exported kinematic tree")
        for key in ("imu_quat_wxyz", "body_from_sensor_wxyz"):
            quat = vector(message.get(key), 4, key)
            if abs(math.sqrt(sum(x*x for x in quat)) - 1) > 1e-3:
                raise ValueError(f"{key} is not a unit quaternion")
        vector(message.get("imu_gyro_rad_s"), 3, "IMU gyro")
        order = [names.index(n) for n in self.names]
        return dict(message, joint_names=list(self.names), q_rad=[q[i] for i in order],
                    dq_rad_s=[dq[i] for i in order], source_receive_ages_s=ages)

    def candidate(self, reply, sequence):
        if not isinstance(reply, dict):
            raise ValueError("policy reply must be an object")
        if type(reply.get("ready")) is not bool or reply.get("publish_allowed") is not False:
            raise ValueError("policy did not declare advisory-only output")
        if not reply["ready"]:
            return None
        if (reply.get("kind") != "x2_morph_advisory_target" or type(reply.get("version")) is not int
                or reply["version"] != 1 or reply.get("policy_sha256") != MODEL_SHA
                or reply.get("robot_profile") != "morphmimic_v01_x2_ultra"):
            raise ValueError("policy model/profile identity mismatch")
        if type(reply.get("state_sequence")) is not int or reply["state_sequence"] != sequence:
            raise ValueError("reply is not for the requested feedback sequence")
        if reply.get("joint_names") != list(self.names):
            raise ValueError("policy target order differs from native X2 export")
        q = vector(reply.get("q_target_rad"), 31, "native target")
        vector(reply.get("canonical_action"), 29, "canonical action")
        # These positions already include the official default/scale/sign/perm.
        # Copy them exactly. PD gains remain model metadata, not hardware approval.
        groups = {g: dict(joint_names=list(self.names[a:b]), position_rad=q[a:b],
                          model_kp=list(self.kp[a:b]), model_kd=list(self.kd[a:b]))
                  for g, (a, b) in GROUPS.items()}
        return dict(kind="x2_morph_backend_candidate", version=1, ready=True,
                    state_sequence=sequence, joint_names=list(self.names), q_target_rad=q,
                    groups=groups, policy_sha256=MODEL_SHA, publish_allowed=False,
                    hardware_commands=0, hardware_gains_validated=False,
                    target_transform="native_copy_no_second_scaling")


class Backend:
    """One feedback owner; no target survives rejection or transport failure."""

    def __init__(self, contract, client, *, max_feedback_age_ms=120., max_rpc_ms=20.):
        if not all(math.isfinite(x) and x > 0 for x in (max_feedback_age_ms, max_rpc_ms)):
            raise ValueError("feedback and RPC budgets must be positive and finite")
        self.contract, self.client = contract, client
        self.max_age_ms, self.max_rpc_ms = max_feedback_age_ms, max_rpc_ms
        self.last_sequence, self.transport_failed = -1, False

    @staticmethod
    def reject(sequence, reason, **extra):
        return dict(kind="x2_morph_backend_candidate", version=1, ready=False,
                    state_sequence=sequence, reason=reason, publish_allowed=False,
                    hardware_commands=0, **extra)

    def step(self, measurement):
        seq = measurement.get("sequence")
        if self.transport_failed:
            return self.reject(seq, "transport_failed_restart_backend_and_explicitly_reset_policy")
        try:
            state = self.contract.measurement(measurement, self.max_age_ms)
            if seq <= self.last_sequence:
                raise ValueError("feedback sequence must strictly increase")
        except (ValueError, TypeError) as exc:
            return self.reject(seq, str(exc))
        self.last_sequence = seq
        started = time.perf_counter_ns()
        try:
            reply = self.client.request(state)
        except Exception as exc:
            # A timed-out REQ may still have advanced the remote policy history.
            # Latch the failure; do not reconnect silently or reset another owner.
            self.transport_failed = True
            return self.reject(seq, "policy_transport_failed", error=type(exc).__name__)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        try:
            result = self.contract.candidate(reply, seq)
        except (ValueError, TypeError, KeyError) as exc:
            return self.reject(seq, str(exc), local_rpc_ms=elapsed_ms)
        if result is None:
            return self.reject(seq, reply.get("reason", reply.get("error", "human_reference_not_ready")),
                               policy_reset=reply.get("reset", False), local_rpc_ms=elapsed_ms)
        if elapsed_ms > self.max_rpc_ms:
            return self.reject(seq, "policy_rpc_budget_exceeded", local_rpc_ms=elapsed_ms)
        max_age_ms = max(state["source_receive_ages_s"].values()) * 1000 + elapsed_ms
        if max_age_ms > self.max_age_ms:
            return self.reject(seq, "feedback_expired_during_rpc", local_rpc_ms=elapsed_ms)
        return dict(result, local_rpc_ms=elapsed_ms, max_feedback_age_at_reply_ms=max_age_ms,
                    policy_timing=reply.get("timing"), source_advisory=reply)
