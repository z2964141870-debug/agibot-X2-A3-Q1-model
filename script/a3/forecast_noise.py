"""Expected covariance of causal relative-history measurement noise."""

import numpy as np


def noise_covariances(feature_std, joints, horizons, sigma):
    std = np.asarray(feature_std)
    if len(std) % joints or sigma < 0 or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid relative-history normalization/noise")
    histories = len(std)//joints
    # Every history difference contains the same negative current-frame noise.
    covariance = np.kron(np.eye(histories)+np.ones((histories, histories)), np.eye(joints)) * sigma**2
    cross = np.tile(np.eye(joints), (histories, horizons)) * sigma**2
    return covariance / std[:, None] / std[None, :], cross / std[:, None]
