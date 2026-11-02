"""Calibrate an external SOMA target from the real Q1 V3 geometry on CPU."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET


MAPPING = {
    "Hips": "pelvis", "Chest": "torso_link", "Head": "head_link",
    "LeftArm": "left_shoulder_roll_link", "LeftForeArm": "left_elbow_link",
    "LeftHand": "left_wrist_yaw_link", "RightArm": "right_shoulder_roll_link",
    "RightForeArm": "right_elbow_link", "RightHand": "right_wrist_yaw_link",
    "LeftLeg": "left_hip_yaw_link", "LeftShin": "left_knee_link",
    "LeftFoot": "left_ankle_pitch_link", "RightLeg": "right_hip_yaw_link",
    "RightShin": "right_knee_link", "RightFoot": "right_ankle_pitch_link",
}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_helpers(path, wp):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"_compute_local_offset", "_apply_local_offset"}]
    viewer = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "Viewer")
    selected += [node for node in viewer.body if isinstance(node, ast.FunctionDef)
                 and node.name == "_build_post_processing_config"]
    if len(selected) != 3:
        raise ValueError("Pinned configurator helper layout changed")
    namespace = {"wp": wp}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.source = args.source.resolve(strict=True)
    args.xml = args.xml.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    xml_sha = hashlib.sha256(args.xml.read_bytes()).hexdigest()
    sys.path.insert(0, str(args.source))
    import numpy as np
    import warp as wp
    wp.config.enable_cuda = False
    wp.init()
    wp.set_device("cpu")
    import newton
    from soma_retargeter.animation.skeleton import SkeletonInstance
    from soma_retargeter.io.bvh import load_bvh
    from soma_retargeter.pipelines import utils as pipeline
    from soma_retargeter.robotics.robot_registry import register_robots_path
    from soma_retargeter.utils.space_conversion_utils import SpaceConverter, FacingDirectionType

    tree = ET.parse(args.xml)
    root = tree.getroot()
    root.find("compiler").set("meshdir", str(args.xml.parent.parent / "meshes"))
    root.find("option").set("timestep", "0.002")
    for section in ("actuator", "sensor", "keyframe"):
        for node in list(root.findall(section)):
            root.remove(node)
    world = root.find("worldbody")
    for node in list(world):
        if node.tag != "body":
            world.remove(node)
    derived_xml = args.output / "q1_v3_kinematic.xml"
    tree.write(derived_xml, encoding="unicode")
    save(args.output / "manifest.json", dict(name="q1_v3",
        desc=dict(xml_path=str(derived_xml.resolve())),
        retarget_configs=dict(soma="configs/retarget.json")))
    register_robots_path(str(args.output))
    builder = pipeline.create_robot_builder("q1_v3")
    model = builder.finalize(device="cpu")
    if model.joint_coord_count != 29 or model.joint_dof_count != 28:
        raise ValueError("Expected free root and22 real hinges")
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    labels = [name.split("/")[-1] for name in model.body_label]
    if len(set(labels)) != len(labels) or any(name not in labels for name in MAPPING.values()):
        raise ValueError("Real Q1 mapped links are not uniquely represented")
    robot = state.body_q.numpy()
    zero_bvh = Path(pipeline.get_source_zero_pose_asset_path(pipeline.SourceType.SOMA))
    skeleton, animation = load_bvh(str(zero_bvh))
    converter = SpaceConverter(FacingDirectionType.MUJOCO)
    z_rot = wp.transform(wp.vec3(0., 0., 0.),
        wp.quat_from_axis_angle(wp.vec3(0., 0., 1.), wp.radians(90.)))
    instance = SkeletonInstance(skeleton, (1., 1., 1.),
        wp.mul(z_rot, converter.transform(wp.transform_identity())))
    instance.set_local_transforms(animation.get_local_transforms(0))
    human_unscaled = instance.compute_global_transforms(scale=1.)
    scale = float(robot[labels.index("pelvis"), 2] /
                  human_unscaled[skeleton.joint_index("Hips"), 2])
    if not .1 < scale < 1.5:
        raise ValueError("Nonphysical pelvis scale")
    instance.scale = scale
    human = instance.compute_global_transforms()
    helper_source = args.source / "app/tools/robot_config_generator.py"
    helpers = source_helpers(helper_source, wp)
    offsets, errors = {}, []
    for joint, body in MAPPING.items():
        src = wp.transform(*human[skeleton.joint_index(joint)])
        target = wp.transform(*robot[labels.index(body)])
        offset = helpers["_compute_local_offset"](src, target)
        restored = helpers["_apply_local_offset"](src, offset)
        pos_error = float(np.linalg.norm(np.array(restored.p) - np.array(target.p)))
        quat_error = min(float(np.linalg.norm(np.array(restored.q) - np.array(target.q))),
                         float(np.linalg.norm(np.array(restored.q) + np.array(target.q))))
        errors.append(dict(joint=joint, position_m=pos_error, quaternion_l2=quat_error))
        if pos_error > 2e-6 or quat_error > 2e-6:
            raise ValueError("Calibration roundtrip failed: " + joint)
        offsets[joint] = [list(offset.p), list(offset.q)]
    defaults_file = args.source / "app/tools/assets/robot_configurator/robot_config_generator_config.json"
    defaults = json.loads(defaults_file.read_text())
    ik_table = {joint: dict(t_body=body, r_body=body,
        t_weight=defaults["default_soma_weights"][joint][0],
        r_weight=defaults["default_soma_weights"][joint][1]) for joint, body in MAPPING.items()}
    retarget = dict(defaults["default_ik_params"], smooth_joint_filter_weight=0.,
        human_robot_scaler_config="configs/scaler.json",
        post_processing=dict(robot_config="configs/post.json"), ik_match_table=ik_table)
    fake = SimpleNamespace(body_link_names=["(none)", *MAPPING.values()],
        mapping_body_idx={joint: i + 1 for i, joint in enumerate(MAPPING)})
    post = helpers["_build_post_processing_config"](fake)
    save(args.output / "configs/retarget.json", retarget)
    save(args.output / "configs/scaler.json", dict(human_type="soma", human_root="Hips",
        robot_type="q1_v3", robot_root="pelvis", human_scale_ratio=scale,
        reference_joint_q=model.joint_q.numpy().tolist(), human_joint_offsets=offsets))
    save(args.output / "configs/post.json", post)
    result = dict(status="CALIBRATED", source_xml_sha256=xml_sha,
        derived_xml_sha256=hashlib.sha256(derived_xml.read_bytes()).hexdigest(),
        configurator_source_sha256=hashlib.sha256(helper_source.read_bytes()).hexdigest(),
        defaults_source_sha256=hashlib.sha256(defaults_file.read_bytes()).hexdigest(),
        zero_bvh_sha256=hashlib.sha256(zero_bvh.read_bytes()).hexdigest(),
        human_scale_ratio=scale, reference_joint_q=model.joint_q.numpy().tolist(),
        body_labels=labels, mapping=MAPPING, calibration_roundtrip=errors,
        new_dynamics_steps=0, new_optimizer_updates=0, device="cpu",
        geometry="actual Q1 V3; no borrowed X2 scale, offsets or elbow reference")
    if hashlib.sha256(args.xml.read_bytes()).hexdigest() != xml_sha:
        raise RuntimeError("Original Q1 XML changed")
    save(args.output / "calibration.json", result)
    print(json.dumps(dict(status=result["status"], scale=scale, mapped_links=len(MAPPING))), flush=True)


if __name__ == "__main__":
    main()
