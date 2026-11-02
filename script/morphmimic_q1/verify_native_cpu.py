"""Check real descriptors, recurrent memory, sparse losses and native config on CPU."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source / "src"))
    from bootstrap import install
    adapter = install("q1")
    import mujoco
    from morphmimic.constants import JOINT_NAMES
    from morphmimic.model.state import StateParts, StateNormalizer, join_state, split_state
    from morphmimic.model.world_model import WorldModel, WorldModelCfg
    from morphmimic.train.replay import ReplayBuffer, ReplayCfg, WindowBatch
    from morphmimic.model.policy import Policy, PolicyCfg, Critic, CriticCfg
    from morphmimic.env.env_cfg import morphmimic_env_cfg
    from morphmimic.train.imagination import noise_half_widths
    from morphmimic.constants import OBS_NOISE
    from assets import XML, presence

    if args.output.exists():
        raise FileExistsError("Preserve previous CPU result")
    spec = adapter["robot"]
    source_sha = hashlib.sha256(XML.read_bytes()).hexdigest()
    model = spec.entity_cfg_fn().spec_fn().compile()
    original_model = mujoco.MjModel.from_xml_path(str(XML))
    assert (model.nq, model.nv, model.nu) == (29, 28, 0)
    assert model.nbody == original_model.nbody
    data, original_data = mujoco.MjData(model), mujoco.MjData(original_model)
    mujoco.mj_forward(model, data)
    mujoco.mj_forward(original_model, original_data)
    import numpy as np
    assert np.array_equal(data.xpos, original_data.xpos)
    for side in ("left", "right"):
        for index in range(1, 6):
            assert model.geom(f"{side}_foot{index}_collision").id >= 0

    vectors, bounds, terms, geometry, masks = adapter["morphology_population"]()
    assert vectors.shape == (2, 154) and masks.sum(-1).tolist() == [29, 22]
    assert torch.isfinite(vectors).all() and (geometry > 0).all()
    assert not torch.equal(geometry[0], geometry[1])
    assert torch.equal(vectors[:, -29:].bool(), masks)
    mask = presence(spec, vectors)
    tr = spec.translator
    native_limits = torch.tensor(model.jnt_range[1:], dtype=torch.float32)
    wm = WorldModel(WorldModelCfg(hidden=(32, 32), morph_dim=154),
        StateNormalizer(), tr.limits_to_canonical(native_limits), torch.zeros(29))
    wm.set_morphology(vectors[1])
    tracked = [model.body(name).id for name in spec.tracked_bodies]
    batch = 4
    state = join_state(StateParts(torch.zeros(batch, 29), torch.zeros(batch, 29),
        torch.tensor(data.xpos[tracked], dtype=torch.float32).expand(batch, 14, 3),
        torch.tensor(data.xquat[tracked], dtype=torch.float32).expand(batch, 14, 4),
        torch.zeros(batch, 14, 3), torch.zeros(batch, 14, 3)))
    assert state.shape == (batch, 269)
    assert split_state(state).joint_pos.shape == (batch, 29)
    actions = torch.randn(batch, 3, 29, requires_grad=True)
    context = torch.zeros(batch, 62)
    context[:, 0] = .6
    prediction = wm.rollout(state, actions, context)
    projected = torch.where(mask, actions, torch.zeros_like(actions))
    assert torch.equal(prediction[..., 240:], projected)
    assert torch.count_nonzero(prediction[..., :29][..., ~mask]) == 0
    assert torch.count_nonzero(prediction[..., 29:58][..., ~mask]) == 0
    prediction[..., 240:].square().sum().backward()
    assert torch.count_nonzero(actions.grad[..., ~mask]) == 0
    assert torch.count_nonzero(actions.grad[..., mask]) > 0
    assert wm.local_features(state).shape == (batch, 297)
    half = noise_half_widths(OBS_NOISE, torch.device("cpu"))
    assert not half[:29][~mask].any() and not half[29:58][~mask].any()

    replay = ReplayBuffer(ReplayCfg(num_envs=batch, capacity=16))
    replay.add_step(state=state, obs=torch.zeros(batch, 93), action=torch.ones(batch, 29),
        reward=torch.zeros(batch), terminated=torch.zeros(batch, dtype=torch.bool),
        timeout=torch.zeros(batch, dtype=torch.bool), ctx=context,
        frame=torch.zeros(batch, dtype=torch.long), ep_step=torch.zeros(batch, dtype=torch.long))
    assert replay.state.shape[-1] == 269 and replay.ctx.shape[-1] == 62
    assert torch.count_nonzero(replay.action[:, 0, ~mask]) == 0
    policy = Policy(PolicyCfg(enc_hidden=(32,), dec_hidden=(32,), morph_dim=154))
    policy.set_morphology(vectors[1])
    action = policy(torch.zeros(batch, 930), torch.zeros(batch, 670))
    assert torch.count_nonzero(action[:, ~mask]) == 0
    explicit_action = policy(torch.zeros(batch, 930), torch.zeros(batch, 670), vectors[0])
    assert torch.isfinite(explicit_action).all()
    assert torch.count_nonzero(explicit_action[:, ~mask]) > 0
    critic = Critic(CriticCfg(hidden=(32,), morph_dim=154))
    critic.set_morphology(vectors[1])
    value = critic(wm.local_features(state), torch.zeros(batch, 670), wm.scale_ctx(context))
    assert torch.isfinite(value).all()

    class Library:
        def state(self, frames):
            return state[0].expand(*frames.shape, 269)

    logged = torch.cat((state[:, None], prediction.detach()), 1)
    window = WindowBatch(logged, actions.detach(), torch.zeros(batch, 3),
        torch.zeros(batch, 3, dtype=torch.bool), context, torch.zeros(batch, 4, dtype=torch.long))
    loss, info = wm.fit_loss(window, Library())
    assert torch.isfinite(loss) and info["state_loss"] == 0
    contaminated = logged.clone()
    contaminated[..., :29][..., ~mask] = float("nan")
    contaminated[..., 29:58][..., ~mask] = float("nan")
    clean_loss, clean_info = wm.fit_loss(window._replace(state=contaminated), Library())
    assert torch.isfinite(clean_loss) and clean_info["state_loss"] == 0

    cfg = morphmimic_env_cfg("q1", num_envs=16)
    assert cfg.sim.mujoco.timestep == .002 and cfg.decimation == 10
    assert all(a.delay_max_lag == 0 for a in cfg.scene.entities["robot"].articulation.actuators)
    checks = ["real_asset_fk_identity", "named_real_foot_geometry", "real_population_descriptors",
        "state_and_context_dimensions", "autoregressive_command_memory", "missing_action_gradients",
        "masked_sensor_noise", "masked_replay_action", "masked_native_policy", "critic_memory_input",
        "masked_fit_loss_and_nan_slots", "physics_and_actuator_delay_config", "immutable_source_and_cpu_only"]
    result = dict(status="PASSED", checks=checks, contracts=adapter["contracts"],
        new_simulation_steps=0, new_optimizer_updates=0, cuda_initialized=torch.cuda.is_initialized(),
        geometry=geometry.tolist(), presence=masks.tolist(), descriptors=vectors.tolist(),
        morphology_bounds={key: value.tolist() for key, value in bounds.items()},
        raw_actuator_terms=terms.tolist(), joints=list(JOINT_NAMES),
        source_xml_sha256_before=source_sha,
        source_xml_sha256_after=hashlib.sha256(XML.read_bytes()).hexdigest(),
        env_rewards={name: dict(function=term.func.__name__, params=str(term.params))
                     for name, term in cfg.rewards.items()},
        env_terminations={name: dict(function=term.func.__name__, params=str(term.params))
                          for name, term in cfg.terminations.items()})
    assert result["source_xml_sha256_before"] == result["source_xml_sha256_after"]
    assert not result["cuda_initialized"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status="PASSED", checks=result["checks"], geometry=result["geometry"])), flush=True)


if __name__ == "__main__":
    main()
