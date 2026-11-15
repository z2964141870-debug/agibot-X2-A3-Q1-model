import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock
import numpy as np
from .server import Preview


class MinimalPreviewTests(TestCase):
    def test_pose_publication_does_not_require_mesh_or_disk_recording(self):
        with tempfile.TemporaryDirectory() as folder:
            engine, body = Mock(), Mock()
            body.from_pose.side_effect = AssertionError("mesh on critical path")
            engine.gp = None
            engine.model_profile = {"selection": "test"}
            preview = Preview("ble", Path(folder), {"joints": [], "vertices": []},
                              engine=engine, live_body=body, live_mesh=False, log_smpl=False)
            preview.calibration = dict(state="static_passed", captured_unix_ns=1, calibration_data=[0]*120)
            preview.source = Mock()
            preview.source.pair.return_value = dict(epoch=[0, 0], received_ns=1)
            preview.source.active_values.return_value = np.zeros((10, 7))
            def step(_):
                preview.stop_event.set()
                return dict(pose_mat=np.tile(np.eye(3), (24, 1, 1)),
                            root_translation_m=np.zeros(3), inference_ms=1.)
            engine.step.side_effect = step
            preview.run_inference()
            body.from_pose.assert_not_called()
            self.assertEqual(preview.frame_seq, 1)
            self.assertEqual(preview.live_status, "receiving")
            self.assertIsNone(preview.smpl["joints_yup_m"])
            self.assertFalse((Path(folder)/"smpl.jsonl").exists())
