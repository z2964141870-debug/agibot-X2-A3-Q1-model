import tempfile
from pathlib import Path
import unittest

import numpy as np

from inventory_amass import inspect_motion


class AmassHeaderTests(unittest.TestCase):
    def fixture(self, folder, **changes):
        path = Path(folder) / "collection" / "subject" / "walk_stageii.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = {"poses": np.zeros((121, 165)), "trans": np.zeros((121, 3)),
                  "betas": np.zeros(16), "mocap_frame_rate": 120.,
                  "gender": "male", "surface_model_type": "smplx"}
        fields.update(changes)
        fields = {key: value for key, value in fields.items() if value is not None}
        np.savez_compressed(path, **fields)
        return path

    def test_explicit_frame_rate_spellings_preserve_duration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.fixture(folder)
            row = inspect_motion(path, Path(folder))
            self.assertEqual(row["nominal_duration_s"], 1.)
            self.assertEqual(row["source_hz"], 120.)
            self.assertFalse(row["qualified_for_q1_training"])
            path = self.fixture(folder, mocap_frame_rate=None, mocap_framerate=60.)
            self.assertEqual(inspect_motion(path, Path(folder))["nominal_duration_s"], 2.)

    def test_missing_or_conflicting_rate_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for changes in ({"mocap_frame_rate": None}, {"mocap_framerate": 30.}):
                path = self.fixture(folder, **changes)
                with self.assertRaises(ValueError):
                    inspect_motion(path, Path(folder))

    def test_static_shape_is_not_a_motion(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "male_stagei.npz"
            np.savez(path, betas=np.zeros(16), gender="male")
            row = inspect_motion(path, Path(folder))
            self.assertEqual(row["kind"], "shape_or_static_asset")
            self.assertFalse(row["schema_valid"])

    def test_object_motion_is_rejected_without_unpickling(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.fixture(folder, poses=np.full((121, 165), None, dtype=object))
            with self.assertRaises(ValueError):
                inspect_motion(path, Path(folder))


if __name__ == "__main__":
    unittest.main()
