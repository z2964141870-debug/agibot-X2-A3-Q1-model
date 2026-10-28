"""Counterexamples for independent Q1 leg PD asset generation."""

import tempfile
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

from pd_control import CANDIDATES, LEG_NAMES, scene_variant


class TestPDAssets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.xml"
        root = ET.Element("mujoco")
        ET.SubElement(root, "worldbody", {"unchanged": "geometry"})
        actuator = ET.SubElement(root, "actuator")
        for name in (*LEG_NAMES, *("upper_"+str(i) for i in range(10))):
            ET.SubElement(actuator, "position", dict(name=name, joint=name, kp="16", kv="1", forcerange="-22 22"))
        ET.ElementTree(root).write(self.source)
        self.original = self.source.read_bytes()

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_leg_gains_change(self):
        for candidate, (kp, kd) in CANDIDATES.items():
            out = self.root / (candidate+".xml")
            contract = scene_variant(self.source, out, candidate)
            for row in contract["actuators"]:
                self.assertAlmostEqual(row["kp"], 16*(kp if row["leg"] else 1))
                self.assertAlmostEqual(row["kd"], kd if row["leg"] else 1)
                self.assertEqual(row["forcerange"], [-22., 22.])
            parsed = ET.parse(out)
            self.assertEqual(parsed.find("worldbody").attrib, {"unchanged": "geometry"})
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_no_source_or_existing_asset_overwrite(self):
        with self.assertRaises(FileExistsError):
            scene_variant(self.source, self.source, "baseline")
        out = self.root / "copy.xml"
        scene_variant(self.source, out, "baseline")
        with self.assertRaises(FileExistsError):
            scene_variant(self.source, out, "baseline")

    def test_wrong_dimensions_and_duplicates_rejected(self):
        tree = ET.parse(self.source)
        actuator = tree.find("actuator")
        actuator[0].set("joint", LEG_NAMES[1])
        tree.write(self.source)
        with self.assertRaises(ValueError):
            scene_variant(self.source, self.root / "bad.xml", "baseline")
        actuator.remove(actuator[-1])
        tree.write(self.source)
        with self.assertRaises(ValueError):
            scene_variant(self.source, self.root / "bad2.xml", "baseline")

    def test_nonfinite_gain_rejected(self):
        tree = ET.parse(self.source)
        tree.find("actuator")[0].set("kp", "nan")
        tree.write(self.source)
        with self.assertRaises(ValueError):
            scene_variant(self.source, self.root / "bad.xml", "baseline")


if __name__ == "__main__":
    unittest.main()
