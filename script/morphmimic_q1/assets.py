"""Real Q1 V3 entity and sparse translator for the isolated v0.1 adapter."""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch

from canonical import SparseJointMap
from control import parameters, torque_bounds

ROOT = Path("/opt/data/private/ZHY")
XML = ROOT / "data/references/q1/model_20261023_P01/q1_v3/xml/q1_v3_mc.xml"
Q1_JOINTS = tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
) + ("waist_roll_joint", "waist_yaw_joint") + tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
)


@dataclass(frozen=True)
class SparseTranslator:
    mapping: SparseJointMap

    @property
    def num_native(self):
        return len(self.mapping.native_names)

    @property
    def perm(self):
        return self.mapping.native_for_slot

    @property
    def sign(self):
        return self.mapping.signs

    @property
    def offset(self):
        return self.mapping.offsets

    @property
    def untracked(self):
        return ()

    def to_canonical(self, value):
        return self.mapping.canonical(value)

    def to_canonical_delta(self, value):
        return self.mapping.canonical(value, delta=True)

    def to_native(self, value, default):
        return self.mapping.native(value, default)

    def to_native_delta(self, value):
        return self.mapping.native(value, delta=True)

    def limits_to_canonical(self, value):
        return self.mapping.limits(value)


def q1_entity():
    import mujoco
    from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab.utils.spec import create_motor_actuator

    class PrimebotActuator(IdealPdActuator):
        def edit_spec(self, spec, target_names):
            for name in target_names:
                limit = 36. if any(part in name for part in ("hip_", "knee_", "waist_")) else 22.
                actuator = create_motor_actuator(
                    spec, name, effort_limit=limit, armature=None,
                    frictionloss=None, viscous_damping=None,
                    transmission_type=self.cfg.transmission_type,
                )
                self._mjs_actuators.append(actuator)

        def initialize(self, *args, **kwargs):
            super().initialize(*args, **kwargs)
            kp, kd, self.peak, self.knee, self.cutoff = parameters(
                self.target_names, device=self.stiffness.device)
            self.stiffness.copy_(kp)
            self.damping.copy_(kd)
            self.force_limit.copy_(self.force_limit.new_tensor(
                [36. if any(part in name for part in ("hip_", "knee_", "waist_")) else 22.
                 for name in self.target_names]))
            self.default_stiffness.copy_(self.stiffness)
            self.default_damping.copy_(self.damping)
            self.default_force_limit.copy_(self.force_limit)

        def compute(self, cmd):
            # A custom compute keeps mjlab from fusing this as ordinary ideal PD.
            torque = self.stiffness * (cmd.position_target - cmd.pos) - self.damping * cmd.vel
            lower, upper = torque_bounds(cmd.vel, self.peak, self.knee, self.cutoff)
            return torque.maximum(lower).minimum(upper).maximum(
                -self.force_limit).minimum(self.force_limit)

    @dataclass(kw_only=True)
    class PrimebotActuatorCfg(IdealPdActuatorCfg):
        def build(self, entity, target_ids, target_names):
            return PrimebotActuator(self, entity, target_ids, target_names)

    def spec_fn():
        spec = mujoco.MjSpec.from_file(str(XML))
        spec.meshdir = str(XML.parent.parent / "meshes")
        for actuator in list(spec.actuators):
            spec.delete(actuator)
        for sensor in list(spec.sensors):
            spec.delete(sensor)
        for light in list(spec.lights):
            spec.delete(light)
        for geom in list(spec.worldbody.geoms):
            spec.delete(geom)
        for side in ("left", "right"):
            body = spec.body(f"{side}_ankle_roll_link")
            count = 0
            for geom in body.geoms:
                if geom.contype or geom.conaffinity:
                    count += 1
                    geom.name = f"{side}_foot{count}_collision"
                    geom.priority = 1
                    geom.condim = 3
                    geom.friction[0] = .6
            if count != 5:
                raise ValueError(f"Expected Q1 foot mesh and four points, got {side}:{count}")
        return spec

    return EntityCfg(
        spec_fn=spec_fn,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0., 0., .419), joint_pos={".*": 0.}, joint_vel={".*": 0.}),
        articulation=EntityArticulationInfoCfg(
            actuators=(PrimebotActuatorCfg(target_names_expr=(".*",),
                stiffness=1., damping=1., effort_limit=36.),),
            soft_joint_pos_limit_factor=.9),
    )


@cache
def q1_spec():
    from morphmimic.constants import JOINT_NAMES, TRACKED_BODY_NAMES
    from morphmimic.robots.spec import RobotSpec, height_scale
    return RobotSpec(
        name="q1", entity_cfg_fn=q1_entity, action_scale={".*": .5},
        joint_names=Q1_JOINTS, root_body="pelvis", anchor_body="torso_link",
        imu_body="pelvis", tracked_bodies=TRACKED_BODY_NAMES,
        foot_geom_regex=r"^(left|right)_foot[1-5]_collision$", com_body="torso_link",
        translator=SparseTranslator(SparseJointMap.q1(JOINT_NAMES, Q1_JOINTS)),
        csv_joint_columns=Q1_JOINTS,
        height_scale=height_scale(q1_entity, "left_knee_link"),
        contact_budget=(96, 1024),
    )


def presence(spec, ref):
    indices = spec.translator.perm
    return torch.tensor([index is not None for index in indices],
                        device=ref.device, dtype=torch.bool)
