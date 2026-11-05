"""Reference identity, time support, training and held-out selection contracts."""

import csv
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from forecast_data import ADMISSION_SHA, identity
from upper_forecast import (CHANNELS, SavedUpperCorpus, evaluate, execute, fit_ridges,
                            history_features, predict, specification)


def delivered(root, *, bad_time=False, wrong_reference=False):
    name, frames = "source/clip.npz", 35
    row = dict(relative=name, split="train", frames=frames)
    plan = dict(clips=[row], history_frames=25, future_frames=10)
    admission = dict(rows=[dict(relative=name, q1_primebot=dict(sha256="a"*64))])
    folder = root / "final_library/clip_s0"
    folder.mkdir(parents=True)
    path = folder / "upper_timing.csv"
    with path.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "time_s"] + [prefix+channel
            for prefix in ("actual_", "reference_") for channel in CHANNELS])
        for index in range(frames):
            writer.writerow([index, index*.02 + (.1 if bad_time and index == 20 else 0.)]
                            + ["not a model input"]*8 + [index*.01]*8)
    result = folder / "result.json"
    result.write_text(json.dumps(dict(clip="clip", seed=0, complete=True,
        failure_frame=None, valid_steps=34, reference_steps=34,
        reference_sha256=("b" if wrong_reference else "a")*64)))
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(dict(frozen_data=dict(sha256=ADMISSION_SHA),
        artifacts=[dict(relative=str(p.relative_to(root)), **identity(p)) for p in (path, result)])))
    return plan, admission, manifest


class UpperForecastTests(unittest.TestCase):
    def test_preregistration_matches_fixed_options(self):
        path = Path(__file__).with_name("upper_forecast_plan_20261106.json")
        self.assertEqual(json.loads(path.read_text()), specification())

    def test_exact_reference_identity_and_actual_angles_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, admission, manifest = delivered(root)
            corpus = SavedUpperCorpus(plan, admission, root, manifest,
                                      expected_manifest_sha=identity(manifest)["sha256"])
            values = corpus.load("source/clip.npz", "train")
            np.testing.assert_allclose(values[:, 0], np.arange(35)*.01)
            self.assertFalse(values.flags.writeable)
            self.assertEqual(len(corpus.verify_unchanged()), 2)
            with self.assertRaisesRegex(ValueError, "split"):
                corpus.load("source/clip.npz", "test")

    def test_changed_content_and_incomplete_wrong_reference_are_rejected(self):
        for wrong in (False, True):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                plan, admission, manifest = delivered(root, wrong_reference=wrong)
                corpus = SavedUpperCorpus(plan, admission, root, manifest,
                                          expected_manifest_sha=identity(manifest)["sha256"])
                if not wrong:
                    (root / "final_library/clip_s0/upper_timing.csv").write_text("changed")
                with self.assertRaises(ValueError):
                    corpus.load("source/clip.npz", "train")

    def test_time_axis_must_be_contiguous_and_unshifted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, admission, manifest = delivered(root, bad_time=True)
            corpus = SavedUpperCorpus(plan, admission, root, manifest,
                                      expected_manifest_sha=identity(manifest)["sha256"])
            with self.assertRaisesRegex(ValueError, "50 Hz"):
                corpus.load("source/clip.npz", "train")

    def test_identical_full_reference_cannot_cross_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, admission, manifest = delivered(root)
            plan["clips"].append(dict(relative="other/duplicate.npz", frames=35, split="test"))
            admission["rows"].append(dict(relative="other/duplicate.npz",
                                          q1_primebot=dict(sha256="a"*64)))
            with self.assertRaisesRegex(ValueError, "cross"):
                SavedUpperCorpus(plan, admission, root, manifest,
                                 expected_manifest_sha=identity(manifest)["sha256"])

    def test_history_support_and_analytic_baselines(self):
        timeline = np.repeat((np.arange(35)*.02)[None, :, None], 8, axis=2)
        history = timeline[:, :25].copy()
        features = history_features(history)
        timeline[:, 25:] = 999.
        np.testing.assert_array_equal(features, history_features(timeline[:, :25]))
        expected = np.repeat((np.arange(25, 35)*.02)[None, :, None], 8, axis=2)
        np.testing.assert_allclose(predict(history, "cv", {}), expected, atol=1e-15)
        np.testing.assert_allclose(predict(history, "hold", {}), np.full((1, 10, 8), .48))
        damped = predict(history, "damped_cv", {})
        self.assertGreater(float(damped[0, -1, 0]), .48)
        self.assertLess(float(damped[0, -1, 0]), float(expected[0, -1, 0]))
        with self.assertRaises(ValueError):
            history_features(timeline)

    def test_normalization_and_fit_only_train_and_zero_motion_stays_hold(self):
        history, future = np.full((4, 25, 8), .3), np.full((4, 10, 8), .3)
        for split in ("validation", "test"):
            with self.assertRaisesRegex(ValueError, "training"):
                fit_ridges(history, future, split=split)
        models = fit_ridges(history, future, split="train")
        for method, model in models.items():
            self.assertEqual(model["fitted_split"], "train")
            self.assertTrue(np.all(model["scale"] >= 1e-6))
            np.testing.assert_allclose(predict(history, method, models), future, atol=1e-14)

    def test_selection_persisted_before_test_and_test_cannot_change_choice(self):
        plan = dict(history_frames=25, future_frames=10, clips=[
            dict(relative=split+"/clip.npz", split=split, frames=35)
            for split in ("train", "validation", "test")])

        class Corpus:
            def __init__(self, test_value):
                self.reads, self.test_value = [], test_value

            def load(self, name, split):
                self.reads.append(split)
                values = np.repeat((np.arange(35)*.01)[:, None], 8, axis=1)
                if split == "test":
                    values[25:] = self.test_value
                return values

        choices = []
        for value in (-100., 100.):
            corpus = Corpus(value)
            persisted = []

            def selected(selection, models):
                self.assertNotIn("test", corpus.reads)
                self.assertFalse(selection["test_labels_read"])
                persisted.append(deepcopy(selection))

            result = evaluate(plan, corpus, train_limit=1, eval_limit=1,
                              on_selection=selected)
            self.assertEqual(len(persisted), 1)
            self.assertIn("test", corpus.reads)
            self.assertEqual(result["selection"], persisted[0])
            choices.append(result["selection"])
        self.assertEqual(choices[0], choices[1])

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "data/supermimic_for_q1/existing"
            output.mkdir(parents=True)
            marker = output / "result.json"
            marker.write_text("original")
            args = SimpleNamespace(local_cpu_study=True, project_root=root, output=output)
            with self.assertRaisesRegex(ValueError, "new independent"):
                execute(args)
            self.assertEqual(marker.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
