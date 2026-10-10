"""Causal pelvis orientation forecasting for the frozen A3-fast policy."""

import copy

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from script.a3.joint_forecast_replay import JointForecastReplay


def orientation_window(quaternions_wxyz, times, query, end_seen=False):
    q, times, query = np.asarray(quaternions_wxyz), np.asarray(times), np.asarray(query)
    if (q.shape != (len(times), 4) or not len(times) or not np.isfinite(q).all() or
            not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or
            not np.isfinite(query).all() or
            not np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-7, rtol=0)):
        raise ValueError("Invalid arrived pelvis rotations")
    if query.min() < times[0]-1e-9 or query.max() > times[-1]+.20+1e-9 and not end_seen:
        raise ValueError("Orientation requested outside the supported time interval")
    rotation = Rotation.from_quat(q[:, [1, 2, 3, 0]])
    current = rotation[-1]
    result = np.empty((len(query), 4))
    past = query <= times[-1] + 1e-10
    if len(times) == 1:
        result[:] = q[-1]
        return result
    if past.any():
        result[past] = Slerp(times, rotation)(np.clip(query[past], times[0], times[-1])).as_quat()[:, [3, 0, 1, 2]]
    if (~past).any():
        omega = (current * rotation[-2].inv()).as_rotvec() / (times[-1]-times[-2])
        delta = query[~past] - times[-1]
        future = current if end_seen else Rotation.from_rotvec(delta[:, None] * omega) * current
        result[~past] = future.as_quat()[..., [3, 0, 1, 2]]
    return result


class OrientationWindow:
    """Expose only a precomputed ten-slot forecast to the existing tokenizer."""

    def __init__(self, expected_indices, quaternions):
        self.expected_indices, self.quaternions = expected_indices, quaternions

    def __getitem__(self, indices):
        if not np.array_equal(indices, self.expected_indices):
            raise ValueError("Tokenizer requested a different orientation window")
        return self.quaternions


class FullCausalReplay(JointForecastReplay):
    def __init__(self, sim, mode, model_path=None, noise_std=0., seed=0):
        if mode == "oracle_aligned":
            raise ValueError("Oracle is not a causal mode")
        super().__init__(sim, mode, model_path, noise_std, seed)
        self.rotations, self.orientation_errors, self.robot_anchors = {}, [], []
        original_load = self.original_load

        def checked_load(model, runtime, solver, path, **kwargs):
            if runtime.mapping.anchor_body_name != "pelvis_link":
                raise ValueError("Raw root quaternion is only valid for the pelvis anchor")
            return original_load(model, runtime, solver, path, **kwargs)

        self.original_load = checked_load

    def register(self, reference, permutation, source_fps, stride):
        super().register(reference, permutation, source_fps, stride)
        _, quaternion, _, _ = self.sim.load_a3_flat_csv(reference.path, source_fps=source_fps, frame_stride=stride)
        if not np.isfinite(quaternion).all() or not np.allclose(np.linalg.norm(quaternion, axis=1), 1, atol=1e-7, rtol=0):
            raise ValueError("Nonfinite or nonunit native pelvis quaternion")
        self.rotations[str(reference.path)] = quaternion.copy()

    def build(self, reference, ref_frame, anchor, on_end="hold_last", frame_skip=1,
              history_frames=0, valid_future_frames=None, zero_pad_invalid_frames=False):
        anchor = np.asarray(anchor, dtype=float)
        if anchor.shape != (4,) or not np.isfinite(anchor).all() or not np.isclose(np.linalg.norm(anchor), 1, atol=1e-6, rtol=0):
            raise ValueError("Invalid actual robot anchor quaternion")
        times = self.sources[str(reference.path)][0]
        latest = min(int(np.searchsorted(times, ref_frame/50+1e-10, side="right")-1), len(times)-1)
        if latest < 0:
            raise ValueError("No root observation has arrived")
        end_seen = latest == len(times)-1
        base = min(ref_frame if end_seen else int(np.floor((times[latest]+1e-10)*50)), reference.num_frames-1)
        indices = np.minimum(base+np.arange(10), reference.num_frames-1)
        predicted = orientation_window(self.rotations[str(reference.path)][:latest+1], times[:latest+1],
                                       base/50+np.arange(10)/50, end_seen=end_seen)
        # The tokenizer can access only forecasts; clean future rotations remain offline labels.
        view = copy.copy(reference)
        view.anchor_quat_wxyz = OrientationWindow(indices, predicted)
        value = super().build(view, ref_frame, anchor, on_end, frame_skip,
                              history_frames, valid_future_frames, zero_pad_invalid_frames)
        true_rotation = Rotation.from_quat(reference.anchor_quat_wxyz[indices][:, [1, 2, 3, 0]])
        estimated = Rotation.from_quat(predicted[:, [1, 2, 3, 0]])
        angle = (estimated * true_rotation.inv()).magnitude()
        self.orientation_errors.append([float(np.sum(angle**2)), len(angle), float(angle.max())])
        self.robot_anchors.append(np.asarray(anchor).copy())
        return value

    def save(self, path):
        if not self.rows:
            raise ValueError("No actual causal policy inputs")
        np.savez_compressed(path, rows=np.asarray(self.rows), errors=np.asarray(self.errors),
            orientation_errors=np.asarray(self.orientation_errors),
            robot_anchor_quat_wxyz=np.asarray(self.robot_anchors),
            columns=np.asarray(["arrival_s", "target_s", "age_s", "latest_joint_and_root_read_s",
                                "latest_source_index", "eof_seen", "prediction_horizon_s"]),
            mode=np.asarray(self.mode), noise_std_rad=self.noise_std, seed=self.seed,
            anchor=np.asarray("pelvis_link"),
            scope=np.asarray("FULL_POLICY_REFERENCE_CAUSAL_SIMULATED_ARRIVALS_NOT_LIVE_CLOCKS"),
            root_translation=np.asarray("NOT_A_POLICY_REFERENCE_TOKEN; CLEAN_OFFLINE_METRICS_AND_FRAME0_RESET_ONLY"))
