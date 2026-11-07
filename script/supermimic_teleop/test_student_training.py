"""CPU fixtures for real-data provenance, selection and guarded dispatch contracts."""

import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from student_fit import batch_goal, fit_r1, module_sha, sample_batch, sampling_groups
from student_rollouts import collection_jobs, paired_source, validate_records
from student_training import NEW_CODE, WALL_SECONDS, profile_gate, cache_boundary
from common import identity


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(934, 29)

    def forward(self, goal, history):
        return self.net(torch.cat((goal, history), -1))


def dataset(rows=10):
    rng = np.random.default_rng(8)
    return dict(full_goal=rng.normal(size=(rows, 2254)).astype(np.float32),
        history=rng.normal(size=(rows, 930)).astype(np.float32),
        f0=np.zeros((rows, 10, 226), dtype=np.float32),
        teacher_action=rng.normal(size=(rows, 29)).astype(np.float32),
        actual_state=np.zeros((rows, 79), dtype=np.float64),
        prefix=np.tile([10, 5, 4, 2, 1], (rows, 1)), frame=np.arange(10, 10+rows*2, 2),
        clip_id=np.arange(rows) % 2)


class StudentTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def test_collection_selects_distinct_owned_groups_without_loading_test_files(self):
        path = Path(__file__).resolve().parents[2]/"data/manifests/q1_teleop_forecast_split_20261106.json"
        plan = json.loads(path.read_text())
        jobs = collection_jobs(plan)
        self.assertEqual(len(jobs), 18)
        test_groups = {row["group_id"] for row in plan["clips"] if row["split"] == "test"}
        self.assertFalse(test_groups & {row["group_id"] for row in jobs})
        for split, count in (("train", 12), ("validation", 6)):
            rows = [row for row in jobs if row["split"] == split]
            self.assertEqual(len({row["group_id"] for row in rows}), count)
        self.assertLessEqual(sum(row["maximum_steps"] for row in jobs), 18*256)
        bad = copy.deepcopy(plan)
        target = next(row for row in bad["clips"] if row["split"] == "train")
        target["group_id"] = next(iter(test_groups))
        with self.assertRaisesRegex(ValueError, "crosses dataset splits"):
            collection_jobs(bad)

    def test_paired_human_identity_compares_content_and_relative_path(self):
        job = dict(relative="source/clip.npz", human=dict(bytes=12, sha256="a"*64))
        item = dict(human=dict(job["human"], path="/private/data/source/clip.npz"))
        self.assertTrue(paired_source(item, job))
        self.assertFalse(paired_source(dict(human=dict(item["human"], sha256="b"*64)), job))
        self.assertFalse(paired_source(dict(human=dict(item["human"], path="/other/clip.npz")), job))

    def provenance(self):
        data = dataset(4)
        jobs = [dict(split="train", maximum_steps=40) for _ in range(2)]
        traces = {i: dict(actual=np.zeros((41, 79)), canonical_action=np.zeros((40, 29))) for i in range(2)}
        for i, (clip, frame) in enumerate(zip(data["clip_id"], data["frame"])):
            traces[int(clip)]["canonical_action"][frame] = data["teacher_action"][i]
        return data, jobs, traces

    def test_stored_training_states_and_labels_must_match_the_actual_physical_trace(self):
        data, jobs, traces = self.provenance()
        self.assertEqual(validate_records(data, "train", jobs, traces)["records"], 4)
        for name, field in (("actual_state", "actual"), ("teacher_action", "canonical_action")):
            bad = copy.deepcopy(data)
            bad[name][0, 0] += 1
            with self.assertRaisesRegex(ValueError, "actual rollout"):
                validate_records(bad, "train", jobs, traces)
        jobs[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "boundary crossed"):
            validate_records(data, "train", jobs, traces)

    def test_duplicate_and_fractional_row_identities_are_rejected(self):
        data, jobs, traces = self.provenance()
        data["frame"] = data["frame"].astype(float)+.5
        with self.assertRaisesRegex(ValueError, "integers"):
            validate_records(data, "train", jobs, traces)
        data, jobs, traces = self.provenance()
        data["frame"][2] = data["frame"][0]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_records(data, "train", jobs, traces)

    def test_actual_visible_counts_drive_masking_and_empty_goals_are_not_fit(self):
        data = dataset()
        data["prefix"][0] = [3, 3, 3, 1, 0]
        goal, _, _, counts = batch_goal(data, np.array([0]), np.array([2]), "cpu")
        self.assertEqual(counts.tolist(), [3])
        self.assertEqual(torch.count_nonzero(goal[0, 3*216:2160]).item(), 0)
        self.assertEqual(torch.count_nonzero(goal[0, -42:]).item(), 0)
        with self.assertRaisesRegex(ValueError, "no available first target"):
            batch_goal(data, np.array([0]), np.array([4]), "cpu")

    def test_horizon_balancing_excludes_missing_first_targets_without_hiding_coverage(self):
        data = dataset()
        data["prefix"][:5, 4] = 0
        groups = sampling_groups(data)
        indices, columns = sample_batch(groups, np.random.default_rng(9), 10, 1)
        self.assertEqual(np.bincount(columns).tolist(), [2]*5)
        self.assertTrue(bool((data["prefix"][indices, columns] > 0).all()))
        data["prefix"][:, 4] = 0
        with self.assertRaisesRegex(ValueError, "H1"):
            sampling_groups(data)

    def test_fixture_adam_counts_validation_selection_and_cpu_save_do_not_change_teacher(self):
        torch.manual_seed(1)
        policy = nn.Module()
        policy.human_encoder = nn.Sequential(nn.Linear(2254, 4), nn.Tanh())
        policy.decoder = Decoder()
        native = SimpleNamespace(policy=policy, present=torch.arange(29) < 22)
        initial = module_sha(policy)
        train, val = dataset(), dataset()
        val["prefix"][0, 4] = 0
        settings = dict(learning_rate=.001, maximum_updates=3, batch_size=10,
            gradient_norm_limit=1., initial_validation_then_every_updates=1)
        result = fit_r1(native, dict(train=train, validation=val), settings, self.root,
            time.monotonic_ns()+60_000_000_000, dict(scope="synthetic_cpu_fixture_not32k"))
        self.assertEqual(result["actual_updates"], 3)
        self.assertEqual(set(result["actual_adam_steps"]), {3})
        self.assertEqual(module_sha(policy), initial)
        self.assertFalse(result["test_sources_read"])
        trace = json.loads((self.root/"training_trace.json").read_text())
        best = min(trace, key=lambda row: row["validation_risk"])
        self.assertEqual(result["selected_update"], best["update"])
        self.assertEqual(trace[0]["validation"][-1]["unavailable_first_target"], 1)
        for name in ("selected.pt", "last.pt"):
            saved = torch.load(self.root/name, map_location="cpu", weights_only=True)
            self.assertTrue(all(t.device.type == "cpu" and torch.isfinite(t).all()
                                for t in saved["encoder"].values()))
        with self.assertRaisesRegex(ValueError, "never test"):
            fit_r1(native, dict(train=train, validation=val, test=val), settings, self.root,
                time.monotonic_ns()+60_000_000_000, {})

    def write_profile(self):
        cases = []
        for event in ("stop", "late_burst"):
            for method, horizon in (("masked", 10), ("damped_cv", 4)):
                cases.append(dict(event=event, method=method, horizon_frames=horizon, calls=128,
                    status="PROFILE_ACTION_PARITY_PASSED", maximum_absolute_action_difference=0.))
                (self.root/f"{event}_{method}_h{horizon}.guard.json").write_text(json.dumps(dict(
                    status="WORKER_COMPLETED", cleanup=dict(worker_reaped=True, worker_return_code=0))))
        source = self.root/"saved_input.json"
        source.write_text("{}\n")
        (self.root/"authorization.json").write_text(json.dumps(dict(
            saved_trace_inputs=[identity(source)], controller_inputs=[])))
        value = dict(stage="SAVED32K_EVENT_PROFILE", status="COMPLETED", source_integrity="PASSED",
            new_policy_calls=512, new_physical_transitions=0, new_training_updates=0, cases=cases)
        (self.root/"result.json").write_text(json.dumps(value))
        return value

    def test_dispatch_cannot_pass_on_unknown_r0_or_changed_saved_source(self):
        value = self.write_profile()
        self.assertEqual(len(profile_gate(self.root)), 7)
        (self.root/"saved_input.json").write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            profile_gate(self.root)
        value["status"] = "RUNNING"
        (self.root/"result.json").write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError, "actually complete"):
            profile_gate(self.root)

    def test_one_failed_profile_worker_preserves_closed_dispatch_gate(self):
        self.write_profile()
        (self.root/"stop_masked_h10.guard.json").write_text(json.dumps(dict(
            status="WORKER_FAILED", cleanup=dict(worker_reaped=True, worker_return_code=1))))
        with self.assertRaisesRegex(RuntimeError, "cleanly reaped"):
            profile_gate(self.root)
        self.assertLess(WALL_SECONDS+5, 1800)
        self.assertIn("student_fit.py", NEW_CODE)

    def test_below_threshold_boundary_does_not_import_or_advise_cache_helper(self):
        sample = dict(total_cache_bytes=1024, oom_kill=0, under_oom=0)
        with patch("student_training.paused_state"), patch("student_training.assert_owner"), \
                patch("student_training.compute_pids", return_value=set()), \
                patch("student_training.memory", return_value=sample), \
                patch("student_training.importlib.util.spec_from_file_location", side_effect=AssertionError("No advice")):
            value = cache_boundary({}, {}, dict(pid=123, start_ticks="456"), self.root)
        self.assertFalse(value["attempted"])
        self.assertEqual(value["status"], "BELOW_4GIB_BOUNDARY")


if __name__ == "__main__":
    unittest.main()
