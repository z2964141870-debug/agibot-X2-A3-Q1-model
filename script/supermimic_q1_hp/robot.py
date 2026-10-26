"""Q1 constants read from the supplied Q1 asset, separate from X2 estimates."""

import json
from pathlib import Path

_data = json.loads(Path(__file__).with_name("control.json").read_text())
Q1_JOINT_NAMES = _data["joints"]
TRACKED_BODIES = _data["tracked"]
COLLISION_GEOM_NAMES = tuple(_data["collision"])


def motor_of(name):
    value = _data["control"][name]
    return value["effort"], value["velocity"], value["armature"]


def action_scale(name):
    value = _data["control"][name]
    return .25 * value["effort"] / value["kp"]


def default_joint_pos_list():
    return [_data["control"][name]["default"] for name in Q1_JOINT_NAMES]
