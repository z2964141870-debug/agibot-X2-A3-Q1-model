"""Strict native-checkpoint inference through the pinned v0.1 policy math."""

from contracts import valid_commit
from common import require


CONTRACT = dict(state=269, physical_state=240, local_features=268,
                critic_features=297, context=62, proprio=93, morphology=154)


def validate_metadata(value, robot, commit):
    require(robot in ("g1", "q1") and valid_commit(commit), "Invalid evaluation identity")
    native, actual, cfg = (value.get(k, {}) for k in ("native_q1_contract", "actual_training", "cfg"))
    plan = native.get("plan", {})
    require(native.get("contracts") == CONTRACT and native.get("code_commit") == commit
            and plan.get("code_commit") == commit and plan.get("purpose") == "train"
            and plan.get("robots") == ["g1", "q1"], "Not an audited native shared training checkpoint")
    require(cfg.get("robot") == robot and cfg.get("morphology") is True
            and cfg.get("actor_geometry", True) is True
            and cfg.get("policy_goal", "motion") == "motion"
            and cfg.get("kp_interface") is True and cfg.get("kp_student") == "return",
            "Saved robot, morphology or policy interface differs")
    iteration = value.get("iteration")
    require(type(iteration) is int and iteration >= 0
            and value.get("schedule", {}).get("step") == iteration
            and value.get("schedule", {}).get("world_size") == 2
            and actual.get("iteration") == iteration and actual.get("code_commit") == commit
            and actual.get("fresh_initialization") is True
            and actual.get("profile_weights_reused") is False,
            "Missing actual iteration or fresh-training provenance")
    approved = plan.get("cfg", {})
    require(approved and all(cfg.get(key) == item for key, item in approved.items())
            and iteration <= approved.get("iterations", -1), "Saved configuration differs from its training plan")
    from controller import updates
    require(actual.get("optimizer_updates") == updates(cfg, iteration),
            "Saved optimizer counts disagree with the completed iteration")
    from contracts import expected_transitions
    exposure = 0 if iteration == 0 else expected_transitions(dict(cfg, iterations=iteration))
    require(actual.get("collected_environment_transitions") == exposure,
            "Saved native exposure disagrees with the completed iteration")
    require(value.get("goal_version") == 2 and value.get("kp_goal_version") == 10,
            "Incompatible motion or human goal version")
    require(cfg.get("hist_len") == 10 and cfg.get("goal_anchor", "reference") == "reference",
            "History length or goal frame differs from the approved recipe")
    return dict(robot=robot, iteration=iteration, training_transitions=exposure,
                code_commit=commit, hist_len=cfg["hist_len"], cfg=cfg)


def load_policy(path, spec, commit, device="cpu", expected_plan=None):
    """Call only after bootstrap.install(robot), before any Trainer is built."""
    import torch
    from morphmimic.constants import GOAL_DIM, HUMAN_FEAT_DIM, HUMAN_HEADING_DIM, PROPRIO_DIM
    from morphmimic.model.policy import Policy, PolicyCfg
    from morphmimic.train.normalizers import RunningNormalizer

    value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    meta = validate_metadata(value, spec.name, commit)
    cfg = meta["cfg"]
    if expected_plan is not None:
        require(value["native_q1_contract"]["plan"] == expected_plan, "Checkpoint belongs to another native plan")
    require(value.get("imu_body") == spec.imu_body, "Saved IMU body differs from the native robot")
    expected_presence = torch.tensor([v is not None for v in spec.translator.perm])
    morph = value["morph"]
    actor = morph["actor"]
    require(actor.shape == (154,) and torch.isfinite(actor).all().item()
            and torch.equal(actor[-29:].bool(), expected_presence)
            and torch.equal(actor, value["policy"]["decoder.morph_actor"])
            and morph["film"] == cfg.get("decoder_film"), "Saved actor morphology is not this robot's")
    from morphmimic.robots.morphology import morphology_vector
    expected_actor = morphology_vector(spec.name, actor=True, actor_geometry=cfg.get("actor_geometry", True))
    require(torch.equal(actor.cpu(), expected_actor.cpu()), "Saved morphology differs from the actual native assets")
    modules = ("policy", "obs_norm", "goal_norm", "human_norm", "heading_norm")
    require(all(torch.isfinite(t).all().item() for name in modules
                for t in value[name].values() if isinstance(t, torch.Tensor)),
            "Nonfinite saved inference tensor")
    policy = Policy(PolicyCfg(enc_hidden=tuple(cfg["policy_hidden"]),
        dec_hidden=tuple(cfg["policy_hidden"]), act=cfg["act"], hist_len=cfg["hist_len"],
        latent=cfg["latent"], morph_dim=154, film=bool(morph["film"]), human=True))
    policy.load_state_dict(value["policy"], strict=True)
    norms = {}
    for key, width in (("obs_norm", PROPRIO_DIM), ("goal_norm", GOAL_DIM),
                       ("human_norm", HUMAN_FEAT_DIM), ("heading_norm", HUMAN_HEADING_DIM)):
        norm = RunningNormalizer(width)
        norm.load_state_dict(value[key], strict=True)
        require((norm.var >= 0).all().item() and norm.count.item() >= 0, "Invalid saved normalizer statistics")
        norms[key] = norm.to(device).eval()
    policy = policy.to(device).eval()
    meta.update(inference_precision="float32; upstream evaluation default", normalizers_frozen=True,
                action_noise=False, human_points="full", kp_horizon=10, human_heading=True,
                human_drift=False)
    return NativePolicy(policy, norms, expected_presence.to(device)), meta


class NativePolicy:
    def __init__(self, policy, norms, present):
        self.policy, self.norms, self.present = policy, norms, present
        self.masks = None

    def __call__(self, history, goals, interface="motion"):
        import torch
        from morphmimic.model.human import fixed_human_masks
        from morphmimic.train.imagination import act, act_human
        require(interface in ("motion", "human"), "Unknown native policy interface")
        with torch.inference_mode():
            if interface == "motion":
                action = act(self.policy, self.norms["obs_norm"], self.norms["goal_norm"],
                             history, goals["goal"])
            else:
                if self.masks is None or self.masks[0].shape[0] != history.shape[0]:
                    self.masks = fixed_human_masks(history.shape[0], "full", 10, history.device, heading=True)
                action = act_human(self.policy, self.norms["obs_norm"], self.norms["human_norm"],
                    self.norms["heading_norm"], history, goals["goal_kp"], *self.masks)
        require(action.shape == (history.shape[0], 29) and torch.isfinite(action).all().item(),
                "Nonfinite or invalid native action")
        require(not torch.count_nonzero(action[..., ~self.present]).item(), "Absent joint action is not masked")
        return action
