"""Check owned joins, startup causality, balanced sampling and fixed-anchor fitting."""

import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from student_coverage_data import balanced_indices, cells, combine, validate_records
from student_coverage_fit import fit
from student_coverage_training import validate_visibility
from student_fit import cpu_state, module_sha
from student_input_audit import expected_flags
from student_inputs import fit_current_statistics
from student_models import current_student
from test_student_causal_fit import Decoder, dataset


def fixture():
    rows = [dict(relative=name, group_id=group, split=split, reused_original_prefix=reuse)
        for name, group, split, reuse in (("old_a", "a", "train", True), ("new_a", "a", "train", False),
            ("new_b", "b", "train", False), ("old_v", "v", "validation", True),
            ("new_v", "v", "validation", False))]
    plan = dict(selected_sources=rows, original_jobs=[rows[0], rows[3]],
        new_collection_jobs=[rows[1], rows[2], rows[4]])
    def values(ids):
        value = dataset()
        value["clip_id"] = np.asarray([ids[i % len(ids)] for i in range(8)], dtype=np.int64)
        return value
    origins = [dict(train=values([0]), validation=values([1])) for _ in range(2)]
    origins.append(dict(train=values([0, 1]), validation=values([2])))
    return plan, origins


class CoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_remapping_retains_both_old_origins_and_never_reads_future_payload(self):
        plan, origins = fixture()
        data, sources = combine(origins, plan)
        self.assertEqual(set(data["train"]["clip_id"]), {0, 1, 2})
        self.assertEqual(set(data["validation"]["clip_id"]), {3, 4})
        self.assertEqual(set(data["train"]), {"f0", "history", "teacher_action", "clip_id", "origin_id"})
        self.assertTrue(np.array_equal(data["train"]["f0"][:8], origins[0]["train"]["f0"]))
        self.assertEqual(len(cells(data["train"], sources)), 2)

    def test_unknown_source_test_payload_and_group_split_crossing_are_rejected(self):
        plan, origins = fixture()
        bad = copy.deepcopy(origins)
        bad[2]["train"]["clip_id"][0] = 2
        with self.assertRaisesRegex(ValueError, "owned collection source"):
            combine(bad, plan)
        bad = copy.deepcopy(origins)
        bad[2]["test"] = dataset(9)
        with self.assertRaisesRegex(ValueError, "two preserved origins"):
            combine(bad, plan)
        plan["selected_sources"][3]["group_id"] = "a"
        with self.assertRaisesRegex(ValueError, "crosses source splits"):
            combine(origins, plan)

    def test_sampling_balances_groups_then_clips_then_available_origins(self):
        plan, origins = fixture()
        data, sources = combine(origins, plan)
        groups = cells(data["train"], sources)
        ids = balanced_indices(groups, np.random.default_rng(7), 120, 0)
        counts = {key:int(np.count_nonzero(data["train"]["clip_id"][ids] == key)) for key in range(3)}
        self.assertEqual(counts, {0:30, 1:30, 2:60})
        self.assertEqual(np.bincount(data["train"]["origin_id"][ids][data["train"]["clip_id"][ids] == 0]).tolist(), [15, 15])

    def test_physical_rows_match_teacher_and_include_first_available_startup(self):
        spec = dict(kind="nominal", first_frame=1, duration_frames=2, transport_frames=0)
        jobs = [dict(split="train", maximum_steps=4, packet_event=spec)]
        actual = np.arange(5*79, dtype=np.float64).reshape(5, 79)
        action = np.ones((4, 29), dtype=np.float32)
        frames = np.asarray([0, 2], dtype=np.int64)
        values = np.zeros((2, 10, 226), dtype=np.float32)
        for i, frame in enumerate(frames):
            values[i, :, 223:] = expected_flags(int(frame), 14, spec)
        data = dict(f0=values, history=np.zeros((2, 930), dtype=np.float32),
            teacher_action=action[frames], actual_state=actual[frames], frame=frames,
            clip_id=np.zeros(2, dtype=np.int64))
        traces = {0:dict(actual=actual, canonical_action=action, input_ready=np.ones(4, dtype=bool))}
        value = validate_records(data, "train", jobs, traces)
        self.assertEqual(value["first_available_even_frames"], {"0":0})
        for key in ("actual_state", "teacher_action"):
            bad = {k:v.copy() for k, v in data.items()}
            bad[key][0, 0] += 1
            with self.assertRaisesRegex(ValueError, "differs from the physical trace"):
                validate_records(bad, "train", jobs, traces)
        with self.assertRaisesRegex(ValueError, "startup row"):
            validate_records({k:v[1:] for k, v in data.items()}, "train", jobs, traces)

    def test_visibility_rejects_future_dependency_and_future_capture(self):
        control = 200_000_000
        timing = dict(frame=10, anchor_ns=control, human_real_future_frames=0, future_preview_ns=0,
            requested_history_ns=[i*20_000_000 for i in range(1, 11)], collection_behavior_is_causal=False,
            measured_end_to_end_latency_ns=None, used_capture_ns=[None]*9+[control],
            used_arrival_ns=[None]*9+[control], max_used_dependency_ns=control)
        validate_visibility(timing, 10)
        with self.assertRaisesRegex(RuntimeError, "resampling dependency"):
            validate_visibility(dict(timing, max_used_dependency_ns=control+1), 10)
        with self.assertRaisesRegex(RuntimeError, "Future packet"):
            validate_visibility(dict(timing, used_capture_ns=[None]*9+[control+1]), 10)

    def test_three_updates_freeze_statistics_and_select_only_anchor_eligible_points(self):
        plan, origins = fixture()
        policy = nn.Module()
        policy.cfg = SimpleNamespace(latent=4)
        policy.human_encoder = nn.Linear(2254, 4)
        policy.decoder = Decoder()
        native = SimpleNamespace(policy=policy, present=torch.arange(29) < 22)
        statistics = fit_current_statistics(origins[0]["train"]["f0"], split="train")
        initial = current_student(native, statistics)
        payload = dict(encoder=cpu_state(initial.encoder), current_statistics=statistics,
            frozen_teacher_policy_sha256=module_sha(policy), decoder_sha256=module_sha(policy.decoder))
        settings = dict(seed=2026110704, maximum_updates=3, learning_rate=.0001,
            batch_size=8, validation_every_updates=1, early_stop_stale_validations=6, gradient_norm_limit=1.)
        before = module_sha(policy)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = fit(native, origins, payload, settings, plan, root,
                time.monotonic_ns()+60_000_000_000, dict(scope="synthetic_cpu_fixture_not32k"))
            trace = json.loads((root/"training_trace.json").read_text())
            candidates = [r for r in trace if r["fixed_anchor_non_regression"]]
            self.assertEqual(result["selected_update"], min(candidates, key=lambda r:r["validation_risk"])["update"])
            self.assertEqual(result["actual_adam_steps"], [3]*6)
            self.assertEqual(result["current_statistics"], statistics)
            self.assertEqual(module_sha(policy), before)
            saved = torch.load(root/"selected.pt", map_location="cpu", weights_only=True)
            self.assertTrue(all(v.device.type == "cpu" and torch.isfinite(v).all() for v in saved["encoder"].values()))
            self.assertFalse(result["normalization_refitted"])
            with self.assertRaises(FileExistsError):
                fit(native, origins, payload, settings, plan, root, time.monotonic_ns()+60_000_000_000, {})


if __name__ == "__main__":
    unittest.main()
