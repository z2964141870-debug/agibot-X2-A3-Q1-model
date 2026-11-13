"""Source-backed X2 torso IMU profile; never claims a new physical calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from script.x2_morph_human.artifacts import home, manifest, check
from .contract import Contract

SONIC_IO_SHA = "b1f31a59dc7d22e165452b532943a7de7b88287c6e4371bc69f294358bc5ce35"
SONIC_SAFETY_SHA = "c5f290cdb2c9448fb42a97e4e74dbcc5dd23f6e268875a1df3a29dcd7a844daa"
# BuildPureDampingCommand's vendor damping, independent of policy PD gains.
FAULT_KD = [9.,8.,8.,4.,7.5,3.]*2 + [5.,6.25,4.] + [4.,5.,3.,3.,3.,1.,1.]*2 + [0.,0.]


def prepare(sonic_source, root=None):
    root = Path(root or home())
    io = Path(sonic_source)/"src/aimdk_io.cpp"
    safety = Path(sonic_source)/"src/safety.cpp"
    if hashlib.sha256(io.read_bytes()).hexdigest() != SONIC_IO_SHA:
        raise ValueError("Sonic feedback/writer source differs from inspected source")
    if hashlib.sha256(safety.read_bytes()).hexdigest() != SONIC_SAFETY_SHA:
        raise ValueError("Sonic vendor fault damping source differs from inspected source")
    xml = root/"assets/x2_source/x2_ultra.xml"
    row = next(r for r in manifest()["files"] if r["path"] == "assets/x2_source/x2_ultra.xml")
    check(xml, row)
    tree = ET.parse(xml)
    torso = tree.find(".//body[@name='torso_link']")
    site = torso.find("site[@name='imu_1']") if torso is not None else None
    if site is None or site.get("quat", "1 0 0 0") != "1 0 0 0" or site.get("euler"):
        raise ValueError("expected exported torso IMU with identity installation rotation")
    c = Contract.load(root/"assets/morphmimic_policy/x2.json")
    joints = {j.get("name"): j for j in tree.findall(".//joint[@name]")}
    ranges = [list(map(float, joints[n].get("range").split())) for n in c.names]
    return dict(sensor_convention_id="x2_ultra_torso_imu1_source_20261113",
                measured_sensor_calibration=False, imu_body="torso_link",
                body_from_sensor_wxyz=[1., 0., 0., 0.],
                imu_topic="/aima/hal/imu/torso/state",
                joint_state_topics={g:f"/aima/hal/joint/{g}/state" for g in ("leg","waist","arm","head")},
                evidence=dict(sonic_io_path=str(io.resolve()), sonic_io_sha256=SONIC_IO_SHA,
                              sonic_safety_path=str(safety.resolve()), sonic_safety_sha256=SONIC_SAFETY_SHA,
                              model_xml_path=str(xml.resolve()), model_xml_sha256=row["sha256"],
                              basis="Sonic SDK wxyz/body-gyro convention plus exported torso imu_1 mount",
                              vendor_header_frame_id="base_link", physical_recalibration_performed=False),
                joint_names=list(c.names), kp=list(c.kp), kd=list(c.kd),
                fault_kd=FAULT_KD,
                lower=[r[0] for r in ranges], upper=[r[1] for r in ranges],
                feedback_age_s=.12, target_age_s=.08, engage_s=2., return_s=2.,
                policy_hz=50, writer_hz=500, max_velocity=35., fall_cos=-.3,
                hardware_validation="PENDING_FIELD_TEST")


def validate(cfg):
    measured = cfg.get("measured_sensor_calibration_id")
    if isinstance(measured, str) and measured and not measured.startswith("REPLACE"):
        return "MEASURED_PROFILE"
    evidence = cfg.get("evidence", {})
    if (cfg.get("sensor_convention_id") != "x2_ultra_torso_imu1_source_20261113"
            or cfg.get("imu_body") != "torso_link" or cfg.get("body_from_sensor_wxyz") != [1., 0., 0., 0.]
            or evidence.get("sonic_io_sha256") != SONIC_IO_SHA):
        raise ValueError("measured profile or verified source-backed torso convention required")
    if cfg.get("fault_kd") != FAULT_KD or evidence.get("sonic_safety_sha256") != SONIC_SAFETY_SHA:
        raise ValueError("vendor fault damping must match inspected Sonic source")
    for key, expected in (("sonic_io_path", evidence["sonic_io_sha256"]),
                          ("model_xml_path", evidence.get("model_xml_sha256")),
                          ("sonic_safety_path", evidence.get("sonic_safety_sha256"))):
        if not expected or hashlib.sha256(Path(evidence[key]).read_bytes()).hexdigest() != expected:
            raise ValueError("sensor convention source changed")
    return "SOURCE_BACKED_TORSO_CONVENTION_NOT_NEW_PHYSICAL_CALIBRATION"


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--sonic-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a=p.parse_args()
    profile=prepare(a.sonic_source)
    validate(profile)
    with a.output.open("x") as f:
        json.dump(profile,f,indent=2,allow_nan=False); f.write("\n")
    print(json.dumps(dict(status="SOURCE_BACKED_SENSOR_PROFILE_PREPARED", output=str(a.output),
                          measured_sensor_calibration=False, hardware_commands=0)))


if __name__ == "__main__": main()
