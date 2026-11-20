"""Causal one-step target prediction around the public Q1 recovery controller (simulation candidate)."""

import numpy as np
import onnxruntime as ort
from q1_recovery_runtime import Q1RecoveryBranch
from q1_v2b_runtime import apply_torque_safe_pd_np


class CommandHistory:
    def __init__(self):
        self.previous = None
        self.seen = False

    def update(self, tracker, expert, target, gate, R, w, q, qd, previous_sent_action):
        if self.previous is None:
            self.previous = target.copy()
        self.seen |= gate > 0
        delta = target - self.previous
        self.previous = target.copy()
        state = expert.state[0]
        features = np.concatenate(
            [
                q - tracker.default_q,
                0.08 * qd,
                R[2],
                0.2 * w,
                previous_sent_action * 0.25,
                target - tracker.default_q,
                delta,
                [gate, self.seen],
                state[:4] / [300, 2, 3, 150],
                state[4:] - tracker.default_q,
            ]
        ).astype(np.float32)
        return features, delta


def corrected_target(tracker, target, q, qd, correction):
    proposed = target + np.clip(correction, -0.5, 0.5)
    issued = apply_torque_safe_pd_np(
        proposed,
        q,
        qd,
        tracker.q_lo,
        tracker.q_hi,
        tracker.kp,
        tracker.kd,
        tracker.eff,
        engage_steps=0,
    )
    tau = tracker.kp * (issued - q) - tracker.kd * qd
    return issued, tau, bool(np.any(abs(issued - proposed) > 1e-4))


class PredictiveRecoveryBranch(Q1RecoveryBranch):
    """Use the selected predictor only when configured for a one-tick command queue."""

    def __init__(self, tracker, expert_onnx, q1_xml, predictor_onnx, gain=1.5, *, delay_ticks):
        if delay_ticks not in (0, 1):
            raise ValueError("supported delay_ticks: 0 or 1")
        super().__init__(tracker, expert_onnx, q1_xml)
        self.history = CommandHistory()
        self.delay_ticks = delay_ticks
        self.gain = gain
        cfg = ort.SessionOptions()
        cfg.intra_op_num_threads = cfg.inter_op_num_threads = 1
        self.predictor = ort.InferenceSession(
            str(predictor_onnx), cfg, providers=["CPUExecutionProvider"]
        )

    def reset(self):
        super().reset()
        self.history = CommandHistory()

    def step(self, R, w, q, qd, previous_sent_action, base_step, reset_anchor):
        target, tau, g = super().step(R, w, q, qd, previous_sent_action, base_step, reset_anchor)
        if not self.delay_ticks:
            return target, tau, g
        features, _ = self.history.update(
            self.tracker, self.expert, target, g, R, w, q, qd, previous_sent_action
        )
        if self.history.seen:
            delta = self.predictor.run(None, {"features": features[None]})[0][0]
            target, tau, _ = corrected_target(self.tracker, target, q, qd, self.gain * delta)
        return target, tau, g
