"""Differentiable coupling of extra foot scores to posture and relative height."""

import torch

from goal02_config import PAIR_HEIGHT_STD_M, REFERENCE_CLEARANCE_MIN_M, UP_VECTOR_STD


def coupling_factors(rotation, reference_rotation, sole, reference_sole, stance, swing):
    if rotation.shape[-2:] != (3, 3) or reference_rotation.shape != rotation.shape:
        raise ValueError("Aligned root rotation matrices required")
    expected = (*rotation.shape[:-2], 2)
    if any(value.shape != expected for value in (sole, reference_sole, stance, swing)):
        raise ValueError("Aligned two-foot heights and phases required")
    if stance.dtype != torch.bool or swing.dtype != torch.bool:
        raise ValueError("Boolean reference phases required")
    up_error = (rotation[..., 2, :] - reference_rotation[..., 2, :]).square().sum(-1)
    posture = torch.exp(-up_error / UP_VECTOR_STD**2)
    single_support = (stance.sum(-1) == 1) & (swing.sum(-1) == 1)
    clearance = ((reference_sole * swing).sum(-1)
                 - (reference_sole * stance).sum(-1))
    active = single_support & (clearance >= REFERENCE_CLEARANCE_MIN_M)
    difference_error = ((sole[..., 1] - sole[..., 0])
                        - (reference_sole[..., 1] - reference_sole[..., 0]))
    pair = torch.where(active, torch.exp(-(difference_error / PAIR_HEIGHT_STD_M).square()),
                       torch.ones_like(difference_error))
    return dict(factor=posture * pair, posture_factor=posture, pair_factor=pair,
                pair_active=active, up_error_squared=up_error,
                pair_height_error_m=difference_error)
