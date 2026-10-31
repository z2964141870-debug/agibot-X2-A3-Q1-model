"""Frozen first recipe for the second Q1 gait goal."""

RECIPE = "POSTURE_PAIR_COUPLED_V1"
UP_VECTOR_STD = .15
PAIR_HEIGHT_STD_M = .02
REFERENCE_CLEARANCE_MIN_M = .01


def contract():
    return dict(
        recipe=RECIPE,
        up_vector_std=UP_VECTOR_STD,
        pair_height_std_m=PAIR_HEIGHT_STD_M,
        reference_clearance_min_m=REFERENCE_CLEARANCE_MIN_M,
        coupling="Multiply existing extra foot bonuses by root-up and two-foot height-difference scores",
        pair_condition="Exactly one declared stance and one declared swing; reference clearance >=10mm",
        phases="Immutable reference intentions; not measured support loads",
        base_maximum_step_reward=.11,
        ppo_maximum_step_reward=.15,
        original_tracking_and_costs_preserved=True,
        control_and_residual_radius_preserved=True,
        training_upper_mixture_preserved=True,
        fixed_exploration_std_preserved=True,
        load_channel="PPO last native solve force is 5ms earlier than post-step pose; never label simultaneous",
    )
