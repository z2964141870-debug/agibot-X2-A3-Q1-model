"""Optional simulation branch for a controller with timestamped command-application feedback."""

from collections import deque
import numpy as np
import onnxruntime as ort
from predictor import CommandHistory, corrected_target
from q1_recovery_runtime import Q1RecoveryBranch


class AgeAwareRecoveryBranch(Q1RecoveryBranch):
    """Call acknowledge before step with an issued ID and its application time in 5 ms ticks."""

    def __init__(self, tracker, expert_onnx, q1_xml, model_path, gain):
        super().__init__(tracker, expert_onnx, q1_xml)
        self.gain = gain
        cfg = ort.SessionOptions()
        cfg.intra_op_num_threads = cfg.inter_op_num_threads = 1
        self.model = ort.InferenceSession(str(model_path), cfg, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        super().reset()
        self.history = CommandHistory()
        self.seq = 0
        self.recent = deque(maxlen=8)
        self.confirmed = None
        self.ages = deque(maxlen=4)
        self.last_ack_id = -1

    def acknowledge(self, seq, applied_tick):
        if seq <= self.last_ack_id or applied_tick > 4 * self.seq:
            return False
        for sid, cmd in self.recent:
            if sid != seq:
                continue
            transit = applied_tick - 4 * seq
            if not 0 <= transit <= 6 or (
                self.confirmed is not None and applied_tick < self.confirmed[0]
            ):
                return False
            self.last_ack_id = seq
            self.confirmed = (applied_tick, cmd)
            self.ages.append(transit / 4.0)
            return True
        return False

    def step(self, R, w, q, qd, previous_sent_action, base_step, reset_anchor):
        target, tau, g = super().step(R, w, q, qd, previous_sent_action, base_step, reset_anchor)
        f, _ = self.history.update(
            self.tracker, self.expert, target, g, R, w, q, qd, previous_sent_action
        )
        issued = target
        if self.history.seen and self.confirmed is not None:
            since = (4 * self.seq - self.confirmed[0]) / 4.0
            if since <= 3 and self.ages[-1] > 0:
                previous = self.recent[-1][1]
                features = np.r_[
                    f,
                    self.ages[-1],
                    np.mean(self.ages),
                    since,
                    1.0,
                    previous - target,
                    self.confirmed[1] - previous,
                ].astype(np.float32)
                correction = self.gain * self.model.run(None, {"features": features[None]})[0][0]
                issued, tau, _ = corrected_target(self.tracker, target, q, qd, correction)
        self.recent.append((self.seq, issued.copy()))
        self.seq += 1
        return issued, tau, g
