"""Local development scene, deliberately NOT certified as the training-export plant.

The original public X2 mesh/MJCF + release PD/timing provide an integration smoke
test. Exact check_same_model export and independent quality evaluation remain
separate. This module never connects to hardware and never resets a fall silently.
"""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from .artifacts import verify


def development_scene(root, output):
    verify(root, meshes=True)
    source = root/"assets/x2_source/x2_ultra.xml"
    tree = ET.parse(source)
    doc = json.loads((root/"assets/morphmimic_policy/x2.json").read_text())["robot"]
    m = tree.getroot()
    m.set("model", "x2_morph_human_development_only")
    m.find("compiler").set("meshdir", str((source.parent/"assets").resolve()))
    m.find("option").set("timestep", str(doc["timing"]["physics_dt"]))
    m.find("option").set("integrator", "implicitfast")
    for el in list(m):
        if el.tag in ("keyframe", "actuator"):
            m.remove(el)
    actuator = ET.SubElement(m, "actuator")
    for i, name in enumerate(doc["native_joint_names"]):
        joint = next(j for j in m.findall(".//joint") if j.get("name") == name)
        joint.set("armature", "0.03"); joint.set("frictionloss", "0.3"); joint.set("damping", "0")
        limit = doc["pd"]["effort_limit"][i]
        ET.SubElement(actuator, "position", name="morph_"+name, joint=name,
                      kp=str(doc["pd"]["kp"][i]), kv=str(doc["pd"]["kd"][i]),
                      forcelimited="true", forcerange=f"{-limit} {limit}")
    ET.SubElement(m.find("worldbody"), "geom", name="development_floor", type="plane",
                  size="10 10 .1", friction="1 .005 .0001", rgba=".2 .22 .25 1")
    ET.SubElement(m.find("worldbody"), "light", pos="1 -2 4", dir="-1 2 -4")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="unicode")
    return output


class DevelopmentSim:
    def __init__(self, runtime, xml):
        import mujoco as mj
        self.mj, self.runtime = mj, runtime
        self.model = mj.MjModel.from_xml_path(str(xml))
        self.data = mj.MjData(self.model)
        joints = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, n) for n in runtime.robot.native_joint_names]
        if min(joints) < 0:
            raise ValueError("X2 joint missing")
        self.qadr = self.model.jnt_qposadr[joints]
        self.dadr = self.model.jnt_dofadr[joints]
        self.pelvis = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "pelvis")
        self.left_ankle = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self.right_ankle = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        self.left_wrist = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
        self.right_wrist = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        self.floor_geom = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "development_floor")
        self.left_foot_geoms = [
            i for i in range(self.model.ngeom)
            if (mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, i) or "").startswith("left_foot")
        ]
        self.right_foot_geoms = [
            i for i in range(self.model.ngeom)
            if (mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, i) or "").startswith("right_foot")
        ]
        self.effort_limit = np.maximum(np.abs(self.model.actuator_forcerange[:, 1]), 1e-6)
        self.data.qpos[self.qadr] = runtime.robot.default_joint_pos
        self.data.ctrl[:] = runtime.robot.default_joint_pos
        mj.mj_forward(self.model, self.data)
        # Source model's root height, unchanged. No auto level/trajectory snapping.
        self.initial_z = float(self.data.xpos[self.pelvis, 2])
        self.last_targets = runtime.robot.default_joint_pos.copy()

    def measurement(self, seq, now_ns):
        from .runtime import synthetic_state
        p = synthetic_state(self.runtime, seq, now_ns)
        p["q_rad"] = self.data.qpos[self.qadr].tolist()
        p["dq_rad_s"] = self.data.qvel[self.dadr].tolist()
        p["imu_quat_wxyz"] = self.data.xquat[self.pelvis].tolist()
        vel = np.zeros(6)
        self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_XBODY, self.pelvis, vel, 1)
        p["imu_gyro_rad_s"] = vel[:3].tolist()
        return p

    def _foot_clearance(self, geom_ids):
        if not geom_ids:
            return 0.0
        bottoms = [
            float(self.data.geom_xpos[g, 2] - self.model.geom_size[g, 0])
            for g in geom_ids
        ]
        return float(max(0.0, min(bottoms)))

    def _foot_forces_and_slip(self):
        left_fz = 0.0
        right_fz = 0.0
        left_set = set(self.left_foot_geoms)
        right_set = set(self.right_foot_geoms)
        cf = np.zeros(6)
        for c_i in range(self.data.ncon):
            c = self.data.contact[c_i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if self.floor_geom not in (g1, g2):
                continue
            other = g2 if g1 == self.floor_geom else g1
            sign = 1.0 if g1 == self.floor_geom else -1.0
            self.mj.mj_contactForce(self.model, self.data, c_i, cf)
            f_world = sign * (c.frame.reshape(3, 3).T @ cf[:3])
            fz = max(0.0, float(f_world[2]))
            if other in left_set:
                left_fz += fz
            elif other in right_set:
                right_fz += fz
        vel_l = np.zeros(6)
        vel_r = np.zeros(6)
        self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_XBODY, self.left_ankle, vel_l, 0)
        self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_XBODY, self.right_ankle, vel_r, 0)
        speed_l = float(np.hypot(vel_l[3], vel_l[4]))
        speed_r = float(np.hypot(vel_r[3], vel_r[4]))
        stance_slips = []
        if left_fz > 20.0:
            stance_slips.append(speed_l)
        if right_fz > 20.0:
            stance_slips.append(speed_r)
        slip = float(max(stance_slips)) if stance_slips else 0.0
        return left_fz, right_fz, slip

    def step(self, targets):
        self.last_targets = np.asarray(targets, dtype=np.float64).copy()
        self.data.ctrl[:] = self.last_targets
        peak_torque_ratio = 0.0
        for _ in range(self.runtime.robot.decimation):
            self.mj.mj_step(self.model, self.data)
            ratio = float(np.max(np.abs(self.data.actuator_force) / self.effort_limit))
            if ratio > peak_torque_ratio:
                peak_torque_ratio = ratio
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("nonfinite simulation")
        q = self.data.xquat[self.pelvis]
        w, x, y, z_q = q
        roll = float(np.rad2deg(np.arctan2(2*(w*x + y*z_q), 1 - 2*(x*x + y*y))))
        pitch = float(np.rad2deg(np.arcsin(np.clip(2*(w*y - z_q*x), -1.0, 1.0))))
        tilt = float(np.rad2deg(np.arccos(np.clip(1-2*(x*x+y*y), -1, 1))))
        pos = self.data.xpos[self.pelvis].copy()
        z = float(pos[2])
        left_fz, right_fz, slip = self._foot_forces_and_slip()
        return dict(
            pelvis_pos_m=pos.tolist(),
            pelvis_z_m=z,
            pelvis_roll_deg=roll,
            pelvis_pitch_deg=pitch,
            pelvis_tilt_deg=tilt,
            left_foot_clearance_m=self._foot_clearance(self.left_foot_geoms),
            right_foot_clearance_m=self._foot_clearance(self.right_foot_geoms),
            left_vertical_force_n=left_fz,
            right_vertical_force_n=right_fz,
            stance_slip_speed_m_s=slip,
            max_torque_ratio=peak_torque_ratio,
            left_wrist_pos_m=self.data.xpos[self.left_wrist].tolist(),
            right_wrist_pos_m=self.data.xpos[self.right_wrist].tolist(),
            q_actual_rad=self.data.qpos[self.qadr].tolist(),
            fallen=bool(z < .3 or tilt > 60.0),
        )

    def video_frame(self, renderer):
        import mujoco as mj
        camera = mj.MjvCamera()
        px, py = float(self.data.xpos[self.pelvis, 0]), float(self.data.xpos[self.pelvis, 1])
        camera.lookat[:] = [px, py, .7]
        camera.distance, camera.azimuth, camera.elevation = 2.6, 132, -14
        renderer.update_scene(self.data, camera)
        return renderer.render()
