"""CPU contracts for the equal-source pure-student data intervention."""

import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
import torch

from student_fit import cpu_state, module_sha
from student_onpolicy_pair_data import prepare, schedule
from student_onpolicy_pair_fit import ARMS, eligible, fit_pair, warm_student
from student_proprio_models import residual_student
from test_student_proprio import fixtures as base_fixtures


def fixtures():
    native, original, origins, candidate = base_fixtures()
    student = residual_student(native, original, "causal_history")
    with torch.no_grad():
        student.adapter[-1].weight.fill_(.001)
        student.adapter[-1].bias.fill_(.002)
    payload = dict(base_encoder=cpu_state(student.encoder), base_encoder_sha256=module_sha(student.encoder),
        adapter=cpu_state(student.adapter), current_statistics=original["current_statistics"],
        decoder_sha256=original["decoder_sha256"], frozen_teacher_policy_sha256=module_sha(native.policy))
    pure = copy.deepcopy(origins[0])
    pure["train"]["history"] += .3
    pure["validation"]["history"] += .4
    jobs = copy.deepcopy(candidate["original_jobs"])
    return native, payload, origins, candidate, pure, jobs


class FuturePoison(dict):
    def __getitem__(self, name):
        if name in ("next_state", "actual_state", "terminal_after_action", "teacher_future_dependency"):
            raise AssertionError("Future or state-evidence metadata reached the model-input loader")
        return super().__getitem__(name)


class OnpolicyPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_matching_retains_registered_source_groups_and_rejects_reassigned_splits(self):
        _, _, origins, candidate, pure, jobs = fixtures()
        old, matched, current, sources = prepare(origins, pure, candidate, jobs)
        for split in ("train", "validation"):
            self.assertEqual(set(matched[split]["clip_id"]), set(current[split]["clip_id"]))
            self.assertEqual(set(current[split]["origin_id"]), {3})
            self.assertEqual(set(matched[split]["origin_id"]), {0, 1})
            self.assertEqual(len(old[split]["clip_id"]), 24)
        bad = copy.deepcopy(jobs); bad[0]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "groups and splits"):
            prepare(origins, pure, candidate, bad)
        bad = copy.deepcopy(origins); bad[0]["test"] = pure["train"]
        with self.assertRaisesRegex(ValueError, "two preserved origins"):
            prepare(bad, pure, candidate, jobs)

    def test_future_metadata_is_never_read_as_a_model_input(self):
        _, _, origins, candidate, pure, jobs = fixtures()
        for split in pure:
            pure[split] = FuturePoison(dict(pure[split], next_state=object(), actual_state=object(),
                terminal_after_action=object(), teacher_future_dependency=object()))
        _, _, current, _ = prepare(origins, pure, candidate, jobs)
        self.assertEqual(set(current["train"]), {"f0", "history", "teacher_action", "clip_id", "origin_id"})

    def test_schedules_have_identical_anchor_indices_and_source_slots(self):
        _, _, origins, candidate, pure, jobs = fixtures()
        old, matched, current, sources = prepare(origins, pure, candidate, jobs)
        first, second = np.random.default_rng(77), np.random.default_rng(77)
        for offset in (0, 4, 8, 12):
            a = schedule(old["train"], matched["train"], current["train"], sources, first, 8, offset)
            b = schedule(old["train"], matched["train"], current["train"], sources, second, 8, offset)
            self.assertTrue(all(np.array_equal(i, j) for i, j in zip(a, b)))
            self.assertTrue(np.array_equal(matched["train"]["clip_id"][a[1]], current["train"]["clip_id"][a[2]]))
        with self.assertRaisesRegex(ValueError, "even batch"):
            schedule(old["train"], matched["train"], current["train"], sources, first, 7, 0)

    def test_nonzero_warm_start_has_exact_adapter_and_frozen_base_identity(self):
        native, payload, origins, _, _, _ = fixtures()
        students = [warm_student(native, payload).train() for _ in ARMS]
        self.assertEqual(module_sha(students[0].adapter), module_sha(students[1].adapter))
        for student in students:
            self.assertTrue(all(torch.equal(v, payload["adapter"][k]) for k, v in student.adapter.state_dict().items()))
            self.assertGreater(int(torch.count_nonzero(student.adapter[-1].weight)), 0)
            self.assertFalse(student.encoder.training or student.decoder.training)
            self.assertTrue(all(p.requires_grad == n.startswith("adapter.") for n, p in student.named_parameters()))
        self.assertTrue(eligible([1., 2., 3.], [1., 1.9, 2.9]))
        for anchors in ([1.1, 1., 2.], [0., 2.1, 2.], [0., 1., 3.1]):
            self.assertFalse(eligible([1., 2., 3.], anchors))

    def test_real_tiny_adam_changes_only_adapter_and_saves_auditable_paired_selection(self):
        native, payload, origins, candidate, pure, jobs = fixtures()
        frozen = module_sha(native.policy)
        settings = dict(seed=44, maximum_updates=3, batch_size=8, learning_rate=.001,
            gradient_norm_limit=1., validation_every_updates=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = fit_pair(native, origins, pure, candidate, jobs, payload, settings, root,
                time.monotonic_ns()+120_000_000_000, dict(scope="synthetic_cpu_only"))
            self.assertEqual(result["actual_optimizer_updates"], 6)
            self.assertEqual(result["arms"][0]["schedule_sha256"], result["arms"][1]["schedule_sha256"])
            self.assertEqual(result["arms"][0]["source_sample_counts"], result["arms"][1]["source_sample_counts"])
            for arm in ARMS:
                selected = torch.load(root/arm/"selected.pt", map_location="cpu", weights_only=True)
                last = torch.load(root/arm/"last.pt", map_location="cpu", weights_only=True)
                trace = json.loads((root/arm/"training_trace.json").read_text())
                allowed = [r for r in trace if r["fixed_anchor_non_regression"]]
                self.assertEqual(selected["selected_update"], min(allowed, key=lambda r:r["validation_risk"])["update"])
                self.assertEqual(selected["checkpoint_update"], selected["selected_update"])
                self.assertEqual(last["checkpoint_update"], 3)
                self.assertEqual(last["actual_adam_steps"], [3]*4)
                self.assertEqual(selected["schema"], "q1_f0_onpolicy_residual_v1")
                self.assertEqual(selected["variant"], arm)
                self.assertEqual(selected["human_real_future_frames"], 0)
                self.assertFalse(all(torch.equal(v, payload["adapter"][k]) for k, v in last["adapter"].items()))
                self.assertTrue(all(torch.equal(v, payload["base_encoder"][k]) for k, v in last["base_encoder"].items()))
                self.assertTrue(all(v.device.type == "cpu" and torch.isfinite(v).all() for v in selected["adapter"].values()))
            self.assertEqual(frozen, module_sha(native.policy))
            with self.assertRaises(FileExistsError):
                fit_pair(native, origins, pure, candidate, jobs, payload, settings, root, 0, {})

    def test_update_zero_is_selected_when_initial_validation_is_exact(self):
        native, payload, origins, candidate, pure, jobs = fixtures()
        student = warm_student(native, payload)
        for values in [s for o in origins for s in o.values()]+list(pure.values()):
            with torch.no_grad():
                values["teacher_action"] = student(torch.from_numpy(values["history"]),
                    torch.from_numpy(values["f0"])).numpy().copy()
        settings = dict(seed=45, maximum_updates=1, batch_size=8, learning_rate=.001,
            gradient_norm_limit=1., validation_every_updates=1)
        with tempfile.TemporaryDirectory() as directory:
            result = fit_pair(native, origins, pure, candidate, jobs, payload, settings, Path(directory),
                time.monotonic_ns()+120_000_000_000, dict(scope="synthetic_cpu_only"))
            self.assertTrue(all(r["selected_update"] == 0 and r["selected_adapter_unchanged"] for r in result["arms"]))


if __name__ == "__main__":
    unittest.main()
