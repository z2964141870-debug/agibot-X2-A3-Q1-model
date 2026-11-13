"""Sign-invariant IMU averaging and static-window diagnostics, not pose ground truth."""
import numpy as np


def quaternion_mean(values):
    q = np.asarray(values, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4 or not np.isfinite(q).all():
        raise ValueError("expected finite quaternions [N,4]")
    norms = np.linalg.norm(q, axis=1)
    if not len(q) or (norms < .5).any():
        raise ValueError("invalid quaternion samples")
    q = q / norms[:, None]
    _, vectors = np.linalg.eigh(q.T @ q)
    mean = vectors[:, -1]
    return mean if mean[0] >= 0 else -mean


def summarize_window(frames):
    values = np.asarray(frames, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (10, 7) or len(values) < 60 or not np.isfinite(values).all():
        raise ValueError("need at least 60 finite paired active-IMU frames")
    means = np.stack([quaternion_mean(values[:, j, 3:]) for j in range(10)])
    q = values[:, :, 3:]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    angles = np.rad2deg(2*np.arccos(np.clip(np.abs(np.einsum('tjc,jc->tj', q, means)), 0, 1)))
    jitter = np.percentile(angles, 95, axis=0)
    acc_std = np.linalg.norm(values[:, :, :3].std(axis=0), axis=1)
    from scipy.spatial.transform import Rotation
    rotations = Rotation.from_quat(means[:, [1, 2, 3, 0]]).as_matrix()
    calibration_data = np.concatenate((values[:, :, :3].mean(axis=0).ravel(), rotations.ravel()))
    stable = bool(jitter.max() <= 2.5 and acc_std.max() <= .8)
    return dict(samples=len(values), orientation_jitter_p95_deg=jitter.tolist(),
                acceleration_std_norm_m_s2=acc_std.tolist(), static_window_passed=stable,
                tpose_geometry_verified=False,
                calibration_data=calibration_data.tolist(), quaternion_mean_wxyz=means.tolist(),
                thresholds=dict(jitter_p95_deg=2.5, acceleration_std_m_s2=.8,
                                status="ENGINEERING_STATIC_WINDOW_THRESHOLDS_NOT_VALIDATED_QUALITY_SCORE"))
