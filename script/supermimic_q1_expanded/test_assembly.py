"""Exercise composition boundaries before admitting an expanded robot library."""

import copy
import unittest

from assemble_library import combine


class AssemblyTests(unittest.TestCase):
    def fixture(self):
        rows = []
        bundles = []
        for index, split in enumerate(("train", "development", "test")):
            source = {"source_group": str(index), "sha256": str(index), "split": split}
            rows.append(source)
            name = "clip" + str(index)
            reference = {"path": "/references/" + name + ".npz"}
            motion = {"clip": name, "phuma_source": source, "split": split,
                      "qualified_for_training_pilot": True, "output": reference}
            check = {"clip": name, "source_group": str(index), "split": split,
                     "qualified_for_training_pilot": True, "reference": reference,
                     "gates": {str(gate): True for gate in range(19)}}
            batch = {"batch": {"offset": index}, "selected": [source]}
            manifest = {"reference_protocol": "native_time", "source_receipt": {}, "archive": {},
                        "source_model": {}, "q1_assets": {"control": 1}, "scale": .7, "pilot": False,
                        "sources": [], "selection": {"index": index}, "motions": [motion]}
            checked = {"status": "INDEPENDENT_REFERENCE_CHECK_COMPLETED", "reference_protocol": "native_time",
                       "source": {}, "input_manifest": {"index": index}, "clips": [check]}
            bundles.append((batch, manifest, checked))
        return {"selected": rows}, bundles

    def test_reorders_complete_batches_and_preserves_check_records(self):
        selection, bundles = self.fixture()
        original = copy.deepcopy(bundles)
        manifest, receipt, inherited = combine(selection, list(reversed(bundles)))
        self.assertEqual(manifest["qualified_split_counts"], {"train": 1, "development": 1, "test": 1})
        self.assertEqual(receipt["clips"], [x[2]["clips"][0] for x in original])
        self.assertEqual(bundles, original)
        self.assertEqual(len(inherited), 3)

    def test_partial_and_repeated_batches_are_rejected(self):
        selection, bundles = self.fixture()
        for supplied in (bundles[:2], bundles + [bundles[0]]):
            with self.assertRaisesRegex(ValueError, "Complete fixed selection"):
                combine(selection, supplied)

    def test_changed_model_or_incomplete_gate_cannot_enter_library(self):
        selection, bundles = self.fixture()
        bundles[1][1]["q1_assets"]["control"] = 2
        with self.assertRaisesRegex(ValueError, "Incompatible batch"):
            combine(selection, bundles)
        selection, bundles = self.fixture()
        bundles[0][2]["clips"][0]["gates"].pop("18")
        with self.assertRaisesRegex(ValueError, "complete 19 checks"):
            combine(selection, bundles)

    def test_source_bytes_cannot_reappear_in_another_split(self):
        selection, bundles = self.fixture()
        selection["selected"][2]["sha256"] = "0"
        with self.assertRaisesRegex(ValueError, "Repeated source group"):
            combine(selection, bundles)


if __name__ == "__main__":
    unittest.main()
