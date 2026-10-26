"""Materialize a pinned HP source snapshot and a separate Q1 embodiment."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

UPSTREAM = "310244b73e45775744a75b789eb25a6c9c5a81c9"
TASK = Path("src/unisupermimic/tasks/motion_tracking")
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(repo, output):
    output.mkdir(parents=True, exist_ok=False)
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", UPSTREAM], cwd=repo, text=True
    ).splitlines()
    manifest = {}
    for name in names:
        path = Path(name)
        code = path.parts[0] in ("src", "tools") and path.suffix in (".py", ".yaml")
        license_file = name in ("LICENSE", "NOTICE", "pyproject.toml") or name.startswith("LICENSES/")
        if not code and not license_file:
            continue
        payload = subprocess.check_output(["git", "show", f"{UPSTREAM}:{name}"], cwd=repo)
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        manifest[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    (output / "upstream_source.json").write_text(json.dumps(
        {"commit": UPSTREAM, "files": manifest}, indent=2) + "\n")
    print(json.dumps({"source_commit": UPSTREAM, "files": len(manifest)}))


class Q1Dimensions(ast.NodeTransformer):
    """Change embodiment dimensions only; never transform the shared learner."""

    def __init__(self, filename):
        self.numbers = {
            "mdp.py": {31: 22, 32: 23, 35: 26, 75: 57, 77: 59},
            "manager_terms.py": {32: 23, 35: 26},
            "motion.py": {38: 29},
        }.get(filename, {})

    def visit_Constant(self, node):
        if type(node.value) is int and node.value in self.numbers:
            return ast.copy_location(ast.Constant(self.numbers[node.value]), node)
        if isinstance(node.value, str):
            node.value = node.value.replace("motion_tracking.x2.", "motion_tracking.q1.")
            node.value = node.value.replace("X2SuperMimic", "Q1HPSuperMimic")
        return node

    def visit_Name(self, node):
        node.id = {"X2Kinematics": "Q1Kinematics", "X2_JOINT_NAMES": "Q1_JOINT_NAMES"}.get(node.id, node.id)
        return node

    def visit_ImportFrom(self, node):
        if node.module:
            node.module = node.module.replace("motion_tracking.x2.", "motion_tracking.q1.")
        for alias in node.names:
            alias.name = {"X2Kinematics": "Q1Kinematics", "X2_JOINT_NAMES": "Q1_JOINT_NAMES"}.get(alias.name, alias.name)
        return node


def configure(root, runtime):
    import mujoco
    import numpy as np
    from omegaconf import OmegaConf

    saved = json.loads((runtime / "upstream_source.json").read_text())
    if saved["commit"] != UPSTREAM:
        raise ValueError("Unexpected HP snapshot")
    for name, identity in saved["files"].items():
        if digest(runtime / name) != identity["sha256"]:
            raise ValueError(f"Upstream source changed: {name}")
    source = root / "data/references/q1/model_20261023_P01/q1_v3/xml/q1_v3_mc.xml"
    model = mujoco.MjModel.from_xml_path(str(source))
    if (model.nq, model.nv, model.nu) != (29, 28, 22):
        raise ValueError("Expected Q1 free root plus 22 joints")
    names = [model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    urdf = ET.parse(source.parent.parent / "urdf/q1_v3.urdf").getroot()
    limits = {j.get("name"): j.find("limit") for j in urdf.findall("joint") if j.find("limit") is not None}
    control = {}
    defaults = []
    for index, name in enumerate(names):
        j = model.joint(name).id
        armature = float(model.dof_armature[model.jnt_dofadr[j]])
        if armature <= 0:
            raise ValueError(f"No positive Q1 model armature: {name}")
        effort = float(limits[name].get("effort"))
        expected = float(model.actuator_ctrlrange[index, 1])
        if not math.isclose(effort, expected):
            raise ValueError(f"Model and supplied URDF effort differ: {name}")
        kp = armature * (10 * 2 * math.pi) ** 2
        kd = 2 * 2 * armature * (10 * 2 * math.pi)
        q0 = 0.0
        for suffix, value in (("hip_pitch_joint", -.28), ("knee_joint", .6), ("ankle_pitch_joint", -.32), ("elbow_joint", .3), ("shoulder_pitch_joint", .15)):
            if name.endswith(suffix):
                q0 = value
        if name == "left_shoulder_roll_joint": q0 = .15
        if name == "right_shoulder_roll_joint": q0 = -.15
        q0 = float(np.clip(q0, *model.jnt_range[j]))
        defaults.append(q0)
        control[name] = {"effort": effort, "velocity": float(limits[name].get("velocity")), "armature": armature, "kp": kp, "kd": kd, "default": q0}
    task_dir = runtime / TASK / "q1/supermimic"
    task_dir.mkdir(parents=True, exist_ok=False)
    for path in (runtime / TASK / "x2/supermimic").glob("*.py"):
        if path.name in ("robot.py", "kinematics.py", "evaluate.py", "render.py"):
            continue
        tree = Q1Dimensions(path.name).visit(ast.parse(path.read_text()))
        ast.fix_missing_locations(tree)
        (task_dir / path.name).write_text(ast.unparse(tree) + "\n")
    (task_dir.parent / "__init__.py").touch()
    shutil.copy2(HERE / "kinematics.py", task_dir / "kinematics.py")
    shutil.copy2(HERE / "robot.py", task_dir / "robot.py")
    tree = ET.parse(source)
    xml = tree.getroot()
    xml.find("compiler").set("meshdir", str((source.parent / "../meshes").resolve()))
    option = xml.find("option")
    option.set("timestep", ".005")
    option.set("integrator", "implicitfast")
    option.set("cone", "pyramidal")
    for index, geom in enumerate(xml.find("worldbody").iter("geom")):
        if not geom.get("name"):
            geom.set("name", f"q1_geom_{index:04d}")
    for old in list(xml.findall("actuator")) + list(xml.findall("keyframe")):
        xml.remove(old)
    actuators = ET.SubElement(xml, "actuator")
    for name, spec in control.items():
        ET.SubElement(actuators, "position", name=name, joint=name, kp=str(spec["kp"]), kv=str(spec["kd"]), forcerange=f"{-spec['effort']} {spec['effort']}")
    assets = runtime / "q1_assets"
    assets.mkdir(exist_ok=False)
    scene = assets / "scene.xml"
    tree.write(scene)
    configured = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(configured)
    data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
    data.qpos[7:] = defaults
    mujoco.mj_forward(configured, data)
    import sys
    sys.path.insert(0, str(runtime))
    from tools.x2_geometry import CollisionGeometry
    geometry = CollisionGeometry(configured, ("left_ankle_roll_link", "right_ankle_roll_link"))
    height = .002 - geometry.minimum(data)
    stand = [0, 0, height, 1, 0, 0, 0, *defaults]
    ET.SubElement(ET.SubElement(xml, "keyframe"), "key", name="stand", qpos=" ".join(map(str, stand)), ctrl=" ".join(map(str, defaults)))
    tree.write(scene)
    collision = [configured.geom(g).name for g in range(configured.ngeom) if configured.geom_bodyid[g] != 0 and (configured.geom_contype[g] or configured.geom_conaffinity[g])]
    tracked = ["pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link", "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link", "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link"]
    payload = {"joints": names, "control": control, "tracked": tracked, "collision": collision, "source_model": str(source), "source_sha256": digest(source), "scene": str(scene), "scene_sha256": digest(scene), "armature_source": "Supplied Q1 MJCF (currently 0.004 on every hinge); no X2 motor inertia copied", "pd_formula": "kp=armature*(10*2*pi)^2; kd=4*armature*(10*2*pi); implicit PD/full supplied URDF effort", "stand_height": height}
    (task_dir / "control.json").write_text(json.dumps(payload, indent=2) + "\n")
    base = OmegaConf.load(runtime / "src/unisupermimic/conf/supermimic/config.yaml")
    base.pop("defaults", None)
    base.pop("hydra", None)
    embodiment = OmegaConf.to_container(OmegaConf.load(runtime / "src/unisupermimic/conf/supermimic/task/x2_super_mimic/base.yaml"))
    def relocate(value):
        if isinstance(value, str): return value.replace("motion_tracking.x2.", "motion_tracking.q1.")
        if isinstance(value, list): return [relocate(v) for v in value]
        if isinstance(value, dict): return {k: relocate(v) for k, v in value.items()}
        return value
    cfg = OmegaConf.merge(base, relocate(embodiment))
    cfg.algo.update({"num_envs": 4096, "wm_batch": 1024, "pi_batch": 1024, "ckpt_every": 500, "log_every": 10})
    cfg.training.update({"task_name": "Q1HPSuperMimic", "sim_backend": "mjwarp"})
    cfg.env.scene.model_file = str(scene)
    robot = cfg.env.scene.entities.robot
    robot.joint_names = names
    robot.actuator_names = names
    robot.body_names = tracked
    cfg.env.commands.motion.update({"mjcf_path": str(scene), "context_dim": 26})
    cfg.env.mjwarp_nconmax = 128
    cfg.env.mjwarp_njmax = 500
    OmegaConf.save(cfg, runtime / "q1_config.yaml", resolve=True)
    (runtime / "q1_adapter.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": "Q1_RUNTIME_CONFIGURED_NOT_YET_VALIDATED", "joints": len(names), "scene": str(scene), "pd_unique": sorted({(v['kp'], v['kd']) for v in control.values()})}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("export", "configure"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    args = p.parse_args()
    if args.action == "export": export(args.root, args.runtime)
    else: configure(args.root, args.runtime)


if __name__ == "__main__":
    main()
