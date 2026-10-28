"""Generate independent leg-only PD variants without changing target units."""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET


CANDIDATES = {
    "baseline": (1., 1.),
    "damping075": (1., .75),
    "damping125": (1., 1.25),
    "damping150": (1., 1.5),
    "stiffness125": (1.25, math.sqrt(1.25)),
    "stiffness150": (1.5, math.sqrt(1.5)),
}
LEG_NAMES = tuple(side + "_" + joint + "_joint" for side in ("left", "right")
                  for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"))


def scene_variant(source: Path, output: Path, candidate: str):
    if source.resolve() == output.resolve() or output.exists():
        raise FileExistsError("A PD probe must write a new independent asset")
    kp_scale, kd_scale = CANDIDATES[candidate]
    tree = ET.parse(source)
    actuator = tree.getroot().find("actuator")
    if actuator is None or len(actuator) != 22:
        raise ValueError("Expected22 explicit Q1 position actuators")
    contracts, found = [], set()
    for element in actuator:
        if element.tag != "position" or not all(k in element.attrib for k in ("name", "joint", "kp", "kv", "forcerange")):
            raise ValueError("Explicit position kp/kv/forcerange required")
        joint = element.attrib["joint"]
        leg = joint in LEG_NAMES
        kp, kd = float(element.attrib["kp"]), float(element.attrib["kv"])
        if not all(math.isfinite(v) and v > 0 for v in (kp, kd)):
            raise ValueError("Finite positive PD gains required")
        if leg:
            if joint in found:
                raise ValueError("Duplicate leg actuator")
            found.add(joint)
            element.set("kp", repr(kp * kp_scale))
            element.set("kv", repr(kd * kd_scale))
        contracts.append(dict(name=element.attrib["name"], joint=joint, leg=leg,
                              kp=float(element.attrib["kp"]), kd=float(element.attrib["kv"]),
                              forcerange=[float(v) for v in element.attrib["forcerange"].split()]))
    if found != set(LEG_NAMES):
        raise ValueError("Expected exactly12 named Q1 leg actuators")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return dict(candidate=candidate, kp_multiplier=kp_scale, kd_multiplier=kd_scale,
                physical_action_scale="original robot.action_scale unchanged", actuators=contracts)


def verify_native_gains(env, contract):
    import numpy as np
    import warp as wp

    model = env.backend._cpu_model
    ordered = []
    for row in contract["actuators"]:
        indices = [i for i in range(model.nu) if model.actuator(i).name.split("/")[-1] == row["name"]]
        if len(indices) != 1:
            raise ValueError("Ambiguous compiled actuator " + row["name"])
        ordered.append(indices[0])
    expected_kp = np.array([r["kp"] for r in contract["actuators"]])
    expected_kd = np.array([r["kd"] for r in contract["actuators"]])
    expected_force = np.array([r["forcerange"] for r in contract["actuators"]])
    # The existing scene composition rounds15.791367 to15.7914 before loading.
    tolerance = dict(rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(model.actuator_gainprm[ordered, 0], expected_kp, **tolerance)
    np.testing.assert_allclose(model.actuator_biasprm[ordered, 1], -expected_kp, **tolerance)
    np.testing.assert_allclose(model.actuator_biasprm[ordered, 2], -expected_kd, **tolerance)
    np.testing.assert_allclose(model.actuator_forcerange[ordered], expected_force, rtol=0, atol=0)
    device = env.backend._device_model
    gain = wp.to_torch(device.actuator_gainprm).detach().cpu().numpy()
    bias = wp.to_torch(device.actuator_biasprm).detach().cpu().numpy()
    np.testing.assert_allclose(gain[..., ordered, 0], np.broadcast_to(expected_kp, gain[..., ordered, 0].shape), **tolerance)
    np.testing.assert_allclose(bias[..., ordered, 1], np.broadcast_to(-expected_kp, bias[..., ordered, 1].shape), **tolerance)
    np.testing.assert_allclose(bias[..., ordered, 2], np.broadcast_to(-expected_kd, bias[..., ordered, 2].shape), **tolerance)
    return dict(cpu_and_device_gains_verified=True, gain_shape=list(gain.shape), bias_shape=list(bias.shape),
                requested_kp=expected_kp.tolist(), requested_kd=expected_kd.tolist(),
                cpu_kp=model.actuator_gainprm[ordered, 0].tolist(), cpu_kd=(-model.actuator_biasprm[ordered, 2]).tolist(),
                device_kp=gain[..., ordered, 0].tolist(), device_kd=(-bias[..., ordered, 2]).tolist(),
                tolerance=tolerance, forcerange=expected_force.tolist())
