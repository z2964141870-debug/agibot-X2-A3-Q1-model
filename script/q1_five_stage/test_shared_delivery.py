"""Missing shared cases, media, and changed local artifacts must block acceptance."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from shared_delivery import diagnostic_table, required_paths, verify, write_conclusions
from common import FIXED, TARGET_TRANSITIONS
from comparison import exposure_comparison
from diagnostic_summary import summarize_complete
from test_diagnostic_summary import case


class SharedDeliveryTests(unittest.TestCase):
    def test_complete_report_accepts_minimal_and_full_exposure_records(self):
        from shared_review import comparisons
        final = dict(iteration=10000, global_env_steps=102563840)
        complete = dict(left=dict(complete_cases=0, total_cases=9, diagnostics=summarize_complete([case(complete=False)])),
                        right=dict(complete_cases=9, total_cases=9, diagnostics=summarize_complete([case()])))
        comparison = dict(status="MISSING_COMPARABLE_PREFIX", complete_motion=complete,
                          common_prefix_metrics={})
        exposures = [dict(status="UNAVAILABLE_EARLY_STOP_BEFORE_TARGET"), exposure_comparison([final])]
        plan = dict(code_commit="a" * 40, cfg=dict(iterations=160000, num_envs=1024,
                    wm_only_iters=2000, warmup_env_steps=400, imag_batch=1024, utd=.4))
        for exposure in exposures:
            original = copy.deepcopy(exposure)
            with self.subTest(exposure=exposure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                primebot, output = root / "primebot", root / "shared"
                records = {primebot / "checkpoint_audit.json": dict(initial=dict(iteration=0), final=final),
                           primebot / "final_fixed/result.json": dict(complete_cases=9),
                           primebot / "final_library/result.json": dict(cases=[{}] * 393)}
                records.update({output / row["name"] / "comparison.json": comparison for row in comparisons()})
                for path, value in records.items():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(value))
                write_conclusions(output, plan, dict(primebot_review=str(primebot)), exposure)
                result = json.loads((output / "conclusions.json").read_text())
                report = (output / "README_RESULTS.md").read_text()
                self.assertEqual(len(result["comparisons"]), 14)
                self.assertEqual(result["primebot"]["final"], final)
                self.assertEqual(result["shared"]["per_robot_transitions"], TARGET_TRANSITIONS)
                self.assertEqual(result["equal_exposure"]["status"], exposure["status"])
                self.assertEqual(result["equal_exposure"]["target_q1_transitions"], TARGET_TRANSITIONS)
                self.assertFalse(result["equal_exposure"]["additional_training_authorized"])
                self.assertFalse(result["goal_completed"])
                self.assertFalse(result["conclusions"]["natural_gait_qualified"])
                self.assertIn("Unavailable", report)
                self.assertIn("UNAVAILABLE_EARLY_STOP_BEFORE_TARGET", report)
                self.assertIn("102563840 Q1 transitions", report)
                self.assertEqual(exposure, original)

    def test_diagnostic_report_preserves_missing_initial_and_uses_complete_clip_denominator(self):
        comparison = dict(complete_motion={side: dict(diagnostics=summarize_complete([case(complete=complete)]))
                                          for side, complete in (("left", False), ("right", True))})
        lines = diagnostic_table({"initial_final": comparison})
        self.assertIn("| left | 0 / 1 | Unavailable |", lines[2])
        self.assertIn("| right | 1 / 1 | 1 | 1 / -1 | 100 / 200 | 0.2 |", lines[3])

    def names(self):
        return [*FIXED, *(f"clip_{i}" for i in range(390))]

    def manifest(self):
        paths, videos = required_paths(self.names())
        media = []
        for video in sorted(videos):
            contacts = [str(Path(video).parent / f"frame_{i:05d}.png") for i in (0, 5, 9)]
            paths.update(contacts)
            media.append(dict(relative=video, contact_frames=contacts, probe=dict(nb_frames=10)))
        return dict(status="READY_FOR_MAC_VERIFICATION", clip_names=self.names(), identity={}, media=media,
                    artifacts=[dict(relative=name, bytes=1, sha256="a" * 64) for name in sorted(paths)])

    def test_inventory_requires_every_case_timeseries_and_exact_video_set(self):
        paths, videos = required_paths(self.names())
        self.assertEqual(len(videos), 36)
        self.assertEqual(sum(name.endswith("foot_posture.csv") for name in paths), 4824)
        self.assertEqual(sum(name.endswith("upper_timing.csv") for name in paths), 4824)
        self.assertFalse(any(Path(name).suffix in (".npz", ".pt", ".mjb") for name in paths))
        with self.assertRaises(RuntimeError):
            required_paths(self.names()[:-1])

    def test_missing_duplicated_or_large_members_fail_before_decode(self):
        for mutate in (lambda v: v["artifacts"].pop(),
                       lambda v: v["artifacts"].append(v["artifacts"][0]),
                       lambda v: v["media"].pop(),
                       lambda v: v["artifacts"].append(dict(relative="weights.pt")),
                       lambda v: v["media"][0].update(contact_frames=[])):
            manifest = self.manifest()
            mutate(manifest)
            with tempfile.TemporaryDirectory() as directory, \
                 patch("shared_delivery.load_manifest", return_value=manifest), \
                 patch("shared_delivery.read", return_value={}), \
                 patch("shared_delivery.subprocess.check_output") as decode:
                with self.assertRaises(RuntimeError):
                    verify(Path("manifest"), Path(directory), Path("visual"), Path(directory) / "receipt", Path("ffmpeg"))
                decode.assert_not_called()

    def test_changed_local_artifact_fails_and_records_failure_without_decode(self):
        manifest = self.manifest()
        with tempfile.TemporaryDirectory() as directory, patch("shared_delivery.load_manifest", return_value=manifest), \
             patch("shared_delivery.read", return_value={}), patch("shared_delivery.identity", return_value={}), \
             patch("shared_delivery.artifact", return_value={}), \
             patch("shared_delivery.subprocess.check_output") as decode, patch("shared_delivery.save") as save:
            with self.assertRaisesRegex(RuntimeError, "Mac artifact identity"):
                verify(Path("manifest"), Path(directory), Path("visual"), Path(directory) / "receipt", Path("ffmpeg"))
            decode.assert_not_called()
            self.assertEqual(save.call_args.args[1]["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
