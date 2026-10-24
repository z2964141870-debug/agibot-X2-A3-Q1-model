import copy
import unittest

from retarget_amass_body import body_rows


class AmassQ1InputTests(unittest.TestCase):
    def manifest(self):
        source = {"source": {"sha256": "a" * 64}, "split": "UNASSIGNED_DIAGNOSTIC_ONLY",
                  "qualified_for_q1_training": False, "source_hz": 120., "num_betas": 16,
                  "source_start_frame": 0, "source_stop_frame_exclusive": 481}
        return {"status": "AMASS_BODY_REFERENCES_PREPARED_Q1_PENDING",
                "qualified_for_q1_training": False, "new_q1_references": 0,
                "motions": [{"clip": "synthetic_contract_only", "source": source,
                             "split": source["split"], "qualified_for_q1_training": False,
                             "frames": 201, "duration_s": 4.}]}

    def test_changed_clock_shape_and_admission_are_refused(self):
        self.assertEqual(len(body_rows(self.manifest())), 1)
        for key, value in (("split", "train"), ("frames", 200), ("duration_s", 3.98),
                           ("qualified_for_q1_training", True)):
            record = self.manifest()
            record["motions"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                body_rows(record)
        for key, value in (("source_hz", 30.), ("num_betas", 10), ("source_start_frame", 1),
                           ("source_stop_frame_exclusive", 480), ("qualified_for_q1_training", True)):
            record = self.manifest()
            record["motions"][0]["source"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                body_rows(record)

    def test_duplicate_source_cannot_appear_under_different_clip_names(self):
        record = self.manifest()
        duplicate = copy.deepcopy(record["motions"][0])
        duplicate["clip"] = "another_name_same_source"
        record["motions"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "Repeated source"):
            body_rows(record)


if __name__ == "__main__":
    unittest.main()
