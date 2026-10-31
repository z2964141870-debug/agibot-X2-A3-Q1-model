"""Plan explicit reductions of the pinned MorphMimic experiment without training."""

import argparse
import ast
from fractions import Fraction
import hashlib
import json
from pathlib import Path


UPSTREAM_COMMIT = "aa40fe46a7fa995ba7a7354ad065a2c8ad64a302"
TRAINER_SHA256 = "82519fcb992471eee910dcacf7f962a1e02174e4f7b32cd9d763105220993c67"
EXPERIMENT_ITERATIONS = 40_000
SCALED_BATCHES = ("wm_batch", "imag_batch", "kp_imag_batch", "latent_align_batch")
UNCHANGED = (
    "replay_capacity", "wm_lr", "policy_lr", "critic_lr", "wm_batches",
    "window_k", "horizon", "utd", "warmup_env_steps", "gamma", "lam",
    "action_noise", "actuator_delay", "wm_hidden", "policy_hidden", "critic_hidden",
    "kp_env_frac", "kp_masking", "kp_jitter", "kp_align_weight", "reset_norm",
    "wm_frame", "residual_scale", "termination", "wm_layer_norm", "critic_layer_norm",
    "critic_anchor_pos", "avoid_clip_end", "critic_target_clip", "bf16", "compile",
    "pool_norms", "actor_geometry",
)


def official_defaults(trainer):
    content = trainer.read_bytes()
    if hashlib.sha256(content).hexdigest() != TRAINER_SHA256:
        raise ValueError("Pinned upstream trainer identity differs; re-audit its contract")
    tree = ast.parse(content)
    cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "TrainCfg")
    result = {}
    for node in cfg.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                result[node.target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    required = {"num_envs", "wm_only_iters", "lr_decay_iters", "ckpt_every",
                *SCALED_BATCHES, *UNCHANGED}
    if not required <= result.keys():
        raise ValueError("Missing upstream scalar/tuple defaults")
    return result


def volumes(cfg):
    iterations = cfg["iterations"]
    policy_iters = max(0, iterations - cfg["wm_only_iters"])
    vector_steps = int(Fraction(iterations) / Fraction(str(cfg["utd"])))
    return dict(
        collection_transitions=cfg["num_envs"] * vector_steps,
        initialization_transitions=cfg["num_envs"] * cfg["warmup_env_steps"],
        replay_transition_capacity=cfg["num_envs"] * cfg["replay_capacity"],
        wm_optimizer_updates=iterations * cfg["wm_batches"],
        wm_training_windows=iterations * cfg["wm_batches"] * cfg["wm_batch"],
        wm_fitted_transition_presentations=iterations * cfg["wm_batches"]
            * cfg["wm_batch"] * cfg["window_k"],
        policy_optimizer_updates=policy_iters,
        motion_imagined_rollouts=policy_iters * cfg["imag_batch"],
        motion_imagined_transition_presentations=policy_iters
            * cfg["imag_batch"] * cfg["horizon"],
        keypoint_imagined_rollouts=policy_iters * cfg["kp_imag_batch"],
        latent_alignment_reference_draws=policy_iters * cfg["latent_align_batch"],
    )


def make_plan(defaults, envs):
    original_envs = defaults["num_envs"]
    if type(envs) is not int or envs <= 0 or original_envs % envs:
        raise ValueError("Environment count must be a positive divisor of4096")
    factor = original_envs // envs
    original = {key: defaults[key] for key in (*UNCHANGED, *SCALED_BATCHES)}
    original.update(num_envs=original_envs, iterations=EXPERIMENT_ITERATIONS,
                    wm_only_iters=defaults["wm_only_iters"],
                    lr_decay_iters=defaults["lr_decay_iters"],
                    ckpt_every=defaults["ckpt_every"], kp_interface=True,
                    kp_masking="full", library_dtype="float16")
    scaled = dict(original, num_envs=envs)
    for key in SCALED_BATCHES:
        if defaults[key] % factor:
            raise ValueError("Batch cannot be scaled exactly: " + key)
        scaled[key] = defaults[key] // factor
    for key in ("iterations", "wm_only_iters", "lr_decay_iters", "ckpt_every"):
        scaled[key] = original[key] * factor
    before, after = volumes(original), volumes(scaled)
    matched = ("collection_transitions", "wm_training_windows",
               "wm_fitted_transition_presentations", "motion_imagined_rollouts",
               "motion_imagined_transition_presentations", "keypoint_imagined_rollouts",
               "latent_alignment_reference_draws")
    if any(before[key] != after[key] for key in matched):
        raise AssertionError("Training exposure was not preserved")
    return dict(
        status="PLANNED_NOT_RESOURCE_TESTED_NOT_LAUNCHED",
        upstream_commit=UPSTREAM_COMMIT, upstream_trainer_sha256=TRAINER_SHA256,
        official_readme_experiment_iterations=EXPERIMENT_ITERATIONS,
        upstream_traincfg_default_iterations=defaults["iterations"],
        reduction_factor=factor, original=original, scaled=scaled,
        volumes_per_robot=dict(original=before, scaled=after),
        exactly_matched_counted_exposures=list(matched),
        accounting_scope={
            "collection": "Post-initialization collection; fractional UTD alternates vector-step counts.",
            "latent_alignment": "Reference-draw budget applies only to active multi-robot alignment; it is inactive in single G1.",
            "kp": "Enabled with the official full five-point interface.",
        },
        comparison_checkpoints=dict(iteration_aligned=EXPERIMENT_ITERATIONS,
                                    exposure_aligned=scaled["iterations"]),
        non_equivalences=[
            "Smaller batches produce more Adam steps and different gradient noise; equal exposure is not equal optimization.",
            "Warm-up collection keeps100 steps per environment; initial transitions and resident replay capacity shrink with environment count.",
            "The unchanged learning rate and loss weights are upstream values; warm-up iteration and cosine LR clocks scale with training exposure.",
            "A one-hour dataset is a declared reduced-data experiment, not the paper full-corpus result.",
            "Multi-robot shared training requires a separately verified execution method; three separate specialists do not constitute a unified model.",
            "The pinned source pools policy normalizers by default; the paper describes robot-specific statistics. This must be reported in the final recipe.",
        ],
        elapsed_time_policy=dict(user_accepts_hours=[8, 16],
            training_time_is_not_completion_criterion=True,
            old_1200_second_and_300_second_gait_stops_apply=False,
            resource_nonfinite_and_process_identity_guards_required=True),
        physics_steps=0, policy_calls=0, optimizer_updates=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(make_plan(official_defaults(args.trainer), args.envs),
                         indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
