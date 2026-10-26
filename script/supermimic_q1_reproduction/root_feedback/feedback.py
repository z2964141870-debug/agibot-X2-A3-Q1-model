"""Enable the existing differentiable anchor-pose goal without changing physics."""

from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parents[1]
for directory in (SCRIPT / "supermimic_for_q1", SCRIPT / "supermimic_q1_expanded"):
    sys.path.insert(0, str(directory))

from slotted_sim import SlottedQ1Sim
from supermimic.q1_sim import Q1Sim
from supermimic.robot_spec import GOAL_DIM
from supermimic.world_model.heads import ANCHOR_OBS_DIM, anchor_relative_obs
from supermimic.world_model.wm_trainer import reference_at

VARIANT = "Q1_REFERENCE_ANCHOR_POSE_GOAL_V1"
FEEDBACK_NOISE_HALF = torch.tensor([0.] * 3 + [.05] * 6)


def input_contract():
    return {"version": VARIANT, "base_goal_dim": GOAL_DIM,
        "feedback_dim": ANCHOR_OBS_DIM, "goal_dim": GOAL_DIM + ANCHOR_OBS_DIM,
        "fields": ["reference_minus_robot_anchor_position_body_xyz",
                   "reference_anchor_rotation_body_first_two_matrix_columns_row_major"],
        "anchor": "pelvis", "position_source": "simulator_or_predicted_world_model_state",
        "routing": "Current feedback appended to the motion-encoder goal, not observation history",
        "position_noise_half_width_m": 0., "rotation6d_noise_half_width": .05,
        "imagination": "Existing imagine(anchor_obs=True); feedback remains differentiable",
        "upstream_difference": "X2 uses heading XY plus relative rotation in actor observation history; Q1 reuses its existing body-frame XYZ/rotation goal interface",
        "hardware_position_estimator_verified": False}


def reference_feedback(heads, state, reference):
    bodies = heads.robot_bodies(state)
    index = heads.anchor_index
    return anchor_relative_obs(bodies["pos"][..., index, :], bodies["quat"][..., index, :],
                               reference.body_pos_w[..., index, :],
                               reference.body_quat_w[..., index, :])


class FeedbackInputs:
    def policy_inputs(self):
        observation, goal = super().policy_inputs()
        extra = reference_feedback(self.heads, self.state, reference_at(self.lib, self.frame))
        if self.cfg.observation_noise:
            extra = extra + (2. * torch.rand_like(extra) - 1.) * FEEDBACK_NOISE_HALF.to(extra)
        return observation, torch.cat((goal, extra), -1)


class FeedbackQ1Sim(FeedbackInputs, Q1Sim):
    pass


class FeedbackSlottedQ1Sim(FeedbackInputs, SlottedQ1Sim):
    pass


def configure_imagination():
    from supermimic.imagination import rollout
    rollout.ANCHOR_NOISE_HALF = FEEDBACK_NOISE_HALF


def augment_manifest(manifest, entry):
    from control import augment_training_manifest
    from supermimic.q1_assets import identity

    augment_training_manifest(manifest, entry)
    if (manifest["policy"]["goal_dim"] != GOAL_DIM + ANCHOR_OBS_DIM
            or not manifest["policy_trainer"]["anchor_obs"]):
        raise ValueError("Both collection and imagination must enable the feedback contract")
    manifest["reference_feedback"] = input_contract()
    manifest["main_variable"] = "CURRENT_REFERENCE_ANCHOR_POSITION_AND_ROTATION_FEEDBACK"
    manifest["paper_deviations"].append(input_contract()["upstream_difference"])
    manifest["reference_feedback_source"] = identity(__file__)
    manifest["user_training_budget"] = {"maximum_training_seconds": 3600.,
        "soft_target_seconds": 3000., "maximum_training_phase_seconds": 3500.,
        "cleanup_grace_seconds": 55., "automatic_extension": False}
    return manifest
