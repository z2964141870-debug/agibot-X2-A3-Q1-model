"""RF06 simulation-only leg torque intervention, with baseline action scaling."""

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "supermimic_for_q1"))
from contact_protocol import ContactCorrectedAssets
from supermimic.q1_assets import identity

VARIANT = "Q1_LEGS_CAP50_OTHER25_SCALE25_V1"


def leg_mask(names):
    selected = np.asarray([any(part in name for part in ("hip", "knee", "ankle"))
                           for name in names])
    if len(names) != 22 or int(selected.sum()) != 12:
        raise ValueError("The registered Q1 must have exactly 12 leg and 10 other joints")
    return selected


class Leg50Assets(ContactCorrectedAssets):
    def __init__(self, root):
        super().__init__(root)
        self.leg_joints = leg_mask(self.names)
        self.effort[self.leg_joints] *= 2.

    def manifest(self):
        result = super().manifest()
        fractions = np.where(self.leg_joints, .50, .25)
        result.update(torque_fraction_of_model=fractions.tolist(),
            control_variant={"version": VARIANT, "source": identity(__file__),
                "changed_joint_names": [name for name, selected in zip(self.names, self.leg_joints)
                                        if selected],
                "leg_torque_fraction": .50, "other_torque_fraction": .25,
                "action_scale_torque_fraction": .25,
                "main_variable": "Only the 12 leg actuator force caps change; original action scale, PD, contact, reward and reference remain",
                "hardware_validated": False})
        return result


def augment_training_manifest(manifest, entry):
    if manifest["assets"].get("control_variant", {}).get("version") != VARIANT:
        raise ValueError("RF06 training must use the registered leg-only intervention")
    original = "CPU MuJoCo backend; Q1 simulation PD50/2 and 25% model torque bounds"
    if manifest["paper_deviations"].count(original) != 1:
        raise ValueError("The underlying control declaration changed; review the adapter")
    manifest["paper_deviations"] = [
        "CPU MuJoCo backend; Q1 simulation PD50/2, 50% model leg torque bounds, 25% waist/arm bounds, original 25%-based action scale"
        if item == original else item for item in manifest["paper_deviations"]]
    manifest["base_training_entry"] = manifest["training_entry"]
    manifest["training_entry"] = identity(entry)
    manifest["main_variable"] = "LEG_ACTUATOR_FORCE_CAPS_25_TO_50_PERCENT_ONLY"
    return manifest
