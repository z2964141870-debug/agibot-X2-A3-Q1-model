"""Controls rejecting old inputs, changed files and crossed PHUMA source groups."""

import json
from pathlib import Path
import tempfile
import unittest

from run_training import PROTOCOL, admitted_clips, identity


class TrainingContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.verifier = self.root / "verifier.txt"
        self.verifier.write_text("test verifier identity")
        self.selection = self.root / "selection.txt"
        self.selection.write_text("test grouped selection identity")
        self.motions = [
            {"clip": name, "split": split, "bucket": bucket,
             "qualified_for_training_pilot": True, "phuma_source": {"source_group": name},
             "output": self.reference(name)}
            for name, split, bucket in (("moving", "train", "locomotion"),
                ("arms", "train", "upper"), ("dev", "development", "other"), ("test", "test", "other"))]
        self.manifest = {"reference_protocol": PROTOCOL, "pilot": False,
                         "sources": [], "motions": self.motions, "selection": identity(self.selection)}
        self.receipt = self.root / "independent_check.json"
        self.scan_check = self.root / "scan_check.json"
        self.write_contract()

    def reference(self, name):
        path = self.root / (name + ".npz")
        path.write_bytes(("test identity " + name).encode())
        return identity(path)

    def write_contract(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))
        checked = {"status": "INDEPENDENT_REFERENCE_CHECK_COMPLETED", "reference_protocol": PROTOCOL,
                   "input_manifest": identity(self.root / "manifest.json"), "source": identity(self.verifier),
                   "clips": [{key: row[key] for key in ("clip", "split", "qualified_for_training_pilot")}
                             for row in self.motions]}
        self.receipt.write_text(json.dumps(checked))
        self.scan_check.write_text(json.dumps({"status": "PHUMA_SCAN_INDEPENDENT_RECONCILIATION_PASSED",
            "selection": identity(self.selection), "candidate_pairwise_group_intersections": 0}))

    def admit(self):
        return admitted_clips(self.root, self.receipt, self.scan_check)

    def test_training_selection_excludes_development_and_test(self):
        self.assertEqual(self.admit(), {"train": ["arms", "moving"], "development": ["dev"], "test": ["test"]})

    def test_old_reference_protocol_and_pilot_cannot_train(self):
        for field, value in (("reference_protocol", "P03_NATIVE_TIME_GROUNDED_CALIBRATION_V2"), ("pilot", True)):
            previous = self.manifest[field]
            self.manifest[field] = value
            self.write_contract()
            with self.assertRaises(ValueError):
                self.admit()
            self.manifest[field] = previous

    def test_same_source_group_cannot_cross_splits(self):
        self.motions[2]["phuma_source"]["source_group"] = "moving"
        self.write_contract()
        with self.assertRaisesRegex(ValueError, "source group repeated"):
            self.admit()

    def test_changed_reference_cannot_train(self):
        (self.root / "moving.npz").write_bytes(b"changed reference")
        with self.assertRaisesRegex(ValueError, "reference changed"):
            self.admit()


if __name__ == "__main__":
    unittest.main()
