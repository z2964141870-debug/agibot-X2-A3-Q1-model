"""Bounded MuJoCo model storage with the preserved Q1 collection and reward logic."""

import copy

import mujoco
from mujoco.rollout import Rollout
import numpy as np
import torch

from supermimic.q1_sim import Q1Sim
from supermimic.robot_spec import ACT_DIM, OBS_DIM, STATE_DIM, XI_DIM


class SlottedRollout:
    def __init__(self, sim):
        self.sim = sim
        self.engine = Rollout(nthread=len(sim.data))

    def close(self):
        self.engine.close()

    def rollout(self, models, data, initial, control, *, nstep):
        sim = self.sim
        if models is not sim.models or data is not sim.data or len(initial) != sim.E:
            raise ValueError("Slotted rollout requires its own complete environment state")
        contexts = sim.context.cpu().numpy()
        states, sensors = [], []
        # A reset can touch a reused model. Reapply every environment's continuous DR before each wave.
        for start in range(0, sim.E, len(sim.slot_models)):
            stop = min(start + len(sim.slot_models), sim.E)
            wave = sim.slot_models[:stop - start]
            for model, context in zip(wave, contexts[start:stop]):
                model.geom_friction[sim.assets.foot_geoms, 0] = float(context[0])
                model.body_ipos[sim.torso_id] = sim.base_com + context[1 + ACT_DIM:]
                mujoco.mj_setConst(model, data[0])
            state, sensor = self.engine.rollout(wave, data, initial[start:stop], control[start:stop], nstep=nstep)
            states.append(state.copy())
            sensors.append(sensor.copy())
        return np.concatenate(states), np.concatenate(sensors)


class SlottedQ1Sim(Q1Sim):
    def __init__(self, assets, library, cfg, heads, seed=0, *, slots=4):
        if slots < 1 or cfg.num_envs < 1 or cfg.threads < 1:
            raise ValueError("Positive environment, model-slot and thread counts are required")
        self.assets, self.lib, self.cfg, self.heads = assets, library, cfg, heads
        self.device = library.device
        self.rng = np.random.default_rng(seed)
        self.E = cfg.num_envs
        self.ids = torch.arange(self.E, device=self.device)
        model = copy.copy(assets.model)
        for i, ai in enumerate(assets.aids):
            model.actuator_gainprm[ai, 0] = assets.kp[i]
            model.actuator_biasprm[ai, :3] = [0., -assets.kp[i], -assets.kd[i]]
            model.actuator_biastype[ai] = mujoco.mjtBias.mjBIAS_AFFINE
            model.actuator_forcelimited[ai] = True
            model.actuator_forcerange[ai] = [-assets.effort[i], assets.effort[i]]
        if model.na or mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS) != STATE_DIM + 1:
            raise ValueError("Q1 requires [time, qpos29, qvel28], with no actuator activation")
        self.slot_models = [copy.copy(model) for _ in range(min(slots, self.E))]
        self.models = [self.slot_models[index % len(self.slot_models)] for index in range(self.E)]
        self.data = [mujoco.MjData(model) for _ in range(min(cfg.threads, len(self.slot_models)))]
        self.pool = SlottedRollout(self)
        self.initial = np.zeros((self.E, STATE_DIM + 1), dtype=np.float64)
        self.torso_id = model.body("torso_link").id
        self.base_com = model.body_ipos[self.torso_id].copy()
        self.context = torch.zeros(self.E, XI_DIM, device=self.device)
        self.state = torch.zeros(self.E, STATE_DIM, device=self.device)
        self.frame = torch.zeros(self.E, device=self.device, dtype=torch.long)
        self.previous_action = torch.zeros(self.E, ACT_DIM, device=self.device)
        self.hist = torch.zeros(cfg.history, self.E, OBS_DIM, device=self.device)
        self.steps = torch.zeros(self.E, device=self.device, dtype=torch.long)
        self.ep_start = torch.ones(self.E, device=self.device, dtype=torch.bool)
        self.bin_return = torch.zeros(library.num_bins, device=self.device)
        self.bin_memory = torch.ones(library.num_bins, device=self.device)
        self.trajectories = [[] for _ in range(self.E)]
        self.noise_half = torch.tensor([.01] * ACT_DIM + [.5] * ACT_DIM + [.2] * 3
                                      + [.05] * 3 + [0.] * ACT_DIM, device=self.device)
        self.reset(self.ids)
