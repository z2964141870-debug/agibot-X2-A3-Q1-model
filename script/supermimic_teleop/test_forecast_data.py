"""Leakage, content, interpolation-dependency and normalization boundaries."""

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from forecast_data import (ForecastCorpus, HUMAN_NAMES, TrainingMoments, identity,
                           make_plan, policy_slot_mapping, relative_clip, source_support)


def fixtures():
    admission = dict(status="FROZEN", original_sources_edited=False, admitted=40,
        fixed_motion_names=["fixed_"+str(i) for i in range(3)], rows=[])
    sources = dict(status="COMPLETED_CPU_SOURCE_EXTRACTION", extraction_result=dict(outputs=[]))
    for i in range(40):
        name = ("fixed_"+str(i) if i < 3 else "clip_"+str(i))
        relative = f"{i:06d}/{name}.npz"
        human = dict(bytes=100, sha256=hashlib.sha256(("human"+str(i)).encode()).hexdigest())
        admission["rows"].append(dict(relative=relative, frames=101, human=human))
        sources["extraction_result"]["outputs"].append(dict(clip=relative,
            archive_member="soma_uniform/bvh/"+relative[:-4]+".bvh", bytes=200,
            sha256=hashlib.sha256(("bvh"+str(i)).encode()).hexdigest()))
    return admission, sources


def sidecar(root, row, **changes):
    frames = row["frames"]
    positions = np.zeros((frames, 24, 3))
    positions[:, :, 0] = np.arange(frames)[:, None]
    quaternions = np.zeros((frames, 24, 4))
    quaternions[:, :, 0] = 1.
    values = dict(human_pos_w=positions, human_quat_w=quaternions,
                  human_joint_names=np.array(HUMAN_NAMES), fps=np.array(50),
                  source_frames=np.array((frames-1)*12//5+1))
    values.update(changes)
    path = root / row["relative"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **values)
    row["human"] = identity(path)
    return path


class ForecastDataTests(unittest.TestCase):
    def test_order_independent_whole_directories_and_transitive_content_aliases(self):
        admission, sources = fixtures()
        admission["rows"][8]["human"] = deepcopy(admission["rows"][0]["human"])
        sources["extraction_result"]["outputs"][8]["sha256"] = \
            sources["extraction_result"]["outputs"][9]["sha256"]
        plan = make_plan(admission, sources)
        admission["rows"].reverse()
        sources["extraction_result"]["outputs"].reverse()
        self.assertEqual(plan, make_plan(admission, sources))
        grouped = {r["relative"].split("/")[0]: r for r in plan["clips"]}
        self.assertEqual({grouped[d]["split"] for d in ("000000", "000008", "000009")}, {"test"})
        self.assertEqual(len({grouped[d]["group_id"] for d in ("000000", "000008", "000009")}), 1)
        self.assertFalse(plan["subject_disjoint_verified"])
        self.assertFalse(plan["unseen_end_to_end_policy_generalization_claim_allowed"])
        for key in ("human", "original_bvh"):
            owners = {}
            for row in plan["clips"]:
                previous = owners.setdefault(row[key]["sha256"], row["split"])
                self.assertEqual(previous, row["split"])

    def test_entire_directory_of_fixed_clip_is_held_out(self):
        admission, sources = fixtures()
        admission["rows"][3]["relative"] = "000000/other.npz"
        sources["extraction_result"]["outputs"][3].update(clip="000000/other.npz",
            archive_member="soma_uniform/bvh/000000/other.bvh")
        rows = make_plan(admission, sources)["clips"]
        self.assertEqual([r["split"] for r in rows if r["source_directory"] == "000000"], ["test", "test"])

    def test_source_identity_and_source_path_failures_are_rejected(self):
        for invalid in ("/a/b.npz", "a/../b.npz", "a//b.npz", "a\\b.npz", "a/b/c.npz"):
            with self.assertRaises(ValueError):
                relative_clip(invalid)
        admission, sources = fixtures()
        admission["rows"][0]["human"]["sha256"] = "bad"
        with self.assertRaises(ValueError):
            make_plan(admission, sources)
        admission, sources = fixtures()
        sources["extraction_result"]["outputs"][0]["archive_member"] = "other.bvh"
        with self.assertRaises(ValueError):
            make_plan(admission, sources)

    def test_exact_rational_source_dependencies_and_unchanged_policy_time_slots(self):
        self.assertEqual([source_support(i)["right_support_ns"] for i in range(5)],
                         [0, 5_000_000, 1_666_667, 6_666_667, 3_333_334])
        for i in range(1000):
            row = source_support(i)
            self.assertGreaterEqual(row["dependency_end_ns"], row["nominal_ns"])
            self.assertLess(row["right_support_ns"], 1_000_000_000/120)
            self.assertGreaterEqual(row["source_index"]*5, i*12)
            self.assertLess((row["source_index"]-1)*5, i*12)
        for buffer in (0, 4, 5, 10):
            rows = policy_slot_mapping(buffer)
            self.assertEqual([r["nominal_offset_ns"] for r in rows], [i*20_000_000 for i in range(1, 11)])
            self.assertEqual(sum(r["measured"] for r in rows), buffer)
            self.assertEqual([r["forecast_horizon"] for r in rows[buffer:]], list(range(1, 11-buffer)))

    def test_window_never_crosses_clip_and_unseen_tail_cannot_change_inputs(self):
        admission, sources = fixtures()
        plan = make_plan(admission, sources)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = plan["clips"][0]
            path = sidecar(root, row)
            corpus = ForecastCorpus(plan, root)
            cutoff = 35
            first = corpus.window(row["relative"], cutoff, split=row["split"])
            self.assertEqual(first.history_indices, tuple(range(11, 36)))
            self.assertEqual(first.target_indices, tuple(range(36, 46)))
            self.assertFalse(first.input_positions.flags.writeable)
            with np.load(path) as data:
                values = {k: data[k] for k in data.files}
            values["human_pos_w"][cutoff+1:] = 1e6
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, "content"):
                ForecastCorpus(plan, root).load(row["relative"])
            row["human"] = identity(path)
            second = ForecastCorpus(plan, root).window(row["relative"], cutoff, split=row["split"])
            np.testing.assert_array_equal(first.input_positions, second.input_positions)
            np.testing.assert_array_equal(first.input_quaternions, second.input_quaternions)
            self.assertFalse(np.array_equal(first.target_positions, second.target_positions))
            for invalid in (23, 91, -1, 101):
                with self.assertRaises(ValueError):
                    corpus.window(row["relative"], invalid, split=row["split"])
            with self.assertRaisesRegex(ValueError, "split"):
                corpus.window(row["relative"], cutoff, split="train")

    def test_actual_sidecar_schema_and_test_only_cannot_be_substituted(self):
        admission, sources = fixtures()
        plan = make_plan(admission, sources)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for changes in (dict(fps=np.array(25)), dict(training_allowed=np.array(False)),
                            dict(source_frames=np.array(2)),
                            dict(human_joint_names=np.array(list(reversed(HUMAN_NAMES))))):
                sidecar(root, plan["clips"][0], **changes)
                with self.assertRaises(ValueError):
                    ForecastCorpus(plan, root).load(plan["clips"][0]["relative"])

    def test_online_normalization_cannot_fit_validation_or_test(self):
        moments = TrainingMoments()
        with self.assertRaises(ValueError):
            moments.parameters()
        values = np.arange(42, dtype=float).reshape(14, 3)
        for split in ("validation", "test"):
            with self.assertRaisesRegex(ValueError, "training"):
                moments.update(values, split=split)
        moments.update(values[:3], split="train")
        moments.update(values[3:], split="train")
        parameters = moments.parameters()
        np.testing.assert_allclose(parameters["mean"], values.mean(axis=0))
        np.testing.assert_allclose(parameters["scale"], values.std(axis=0))
        self.assertEqual(parameters["count"], 14)
        self.assertEqual(parameters["fitted_split"], "train")


if __name__ == "__main__":
    unittest.main()
