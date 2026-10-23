"""Pinned dimensions for the supplied Q1 v3 free-root model."""

NJ = ACT_DIM = 22
NQ, NV = 7 + NJ, 6 + NJ
STATE_DIM = FEAT_DIM = NQ + NV
DELTA_DIM = 12 + 2 * NJ
XI_DIM = 1 + ACT_DIM + 3
OBS_DIM = 3 * ACT_DIM + 6
GOAL_FEAT_DIM = 2 * ACT_DIM + 10
HISTORY = FUTURE_FRAMES = 10
GOAL_DIM = FUTURE_FRAMES * GOAL_FEAT_DIM
OBS_TERMS = (("joint_pos", ACT_DIM), ("joint_vel", ACT_DIM),
             ("base_ang_vel", 3), ("projected_gravity", 3), ("actions", ACT_DIM))
