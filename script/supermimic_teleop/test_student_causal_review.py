"""Adversarial visibility, reused-target and checkpoint contracts for f0 replay."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from student_causal_review import (baseline_cases, child_command, review_cases,
                                  trained_causal_evidence, validate_causal_timing)
from student_causal_review_worker import checkpoint_contract
from common import FIXED, identity
from stream_contract import DT_NS
from stream_events import EventSpec
from student_data import OfflineStudentStream


def stream(kind="nominal", duration=10):
    positions = np.zeros((90, 24, 3))
    positions[:, :, 0] = np.arange(90)[:, None]*.01
    quaternions = np.zeros((90, 24, 4))
    quaternions[..., 0] = 1.
    return OfflineStudentStream(positions, quaternions, spec=EventSpec(kind, 50, duration, 0),
        sha256="a"*64, session="synthetic_cpu_contract")


def timing(source, frame):
    window = source.student(frame, mode="f0_history")
    return dict(window.timing(), frame=frame, variant="f0_student", horizon_frames=0,
        human_real_future_frames=0, event=source.events.spec.kind,
        input_ready=window.can_step, input_status=window.status,
        fallback=None if window.can_step else ("initial_zero_delta" if frame == 0 else "previous_action_hold"),
        policy_device="cuda", gpu_policy_wall_compute_ns=1, worker_action_wall_ns=2)


class CausalReviewTests(unittest.TestCase):
    def test_real_receiver_timings_match_independent_visibility_for_all_events(self):
        for kind in ("nominal", "stop", "reverse", "dropout", "late_burst"):
            source = stream(kind)
            for frame in (0, 1, 9, 49, 50, 55, 59, 60, 61, 70, 89):
                value = timing(source, frame)
                self.assertIs(validate_causal_timing(value, frame, source.events), value)
                self.assertTrue(all(t is None or t <= frame*DT_NS for t in value["used_capture_ns"]))
                self.assertEqual(value["future_preview_ns"], 0)

    def test_future_sequence_or_omitted_interpolation_dependency_cannot_claim_f0(self):
        source = stream()
        value = timing(source, 1)
        self.assertEqual(value["used_sequences"][-1], 0)
        self.assertEqual(value["current_capture_age_ns"], DT_NS)
        for key, changed in (("used_sequences", value["used_sequences"][:-1]+[1]),
                ("current_capture_age_ns", 0), ("max_used_dependency_ns", DT_NS),
                ("future_preview_ns", DT_NS), ("horizon_frames", 1),
                ("measured_end_to_end_latency_ns", 0)):
            invalid = dict(value, **{key: changed})
            with self.assertRaisesRegex(RuntimeError, "boundaries"):
                validate_causal_timing(invalid, 1, source.events)

    def test_expired_packet_support_requires_hold_and_later_recovery(self):
        source = stream("dropout", duration=20)
        stale, recovery = timing(source, 65), timing(source, 71)
        self.assertFalse(stale["input_ready"])
        self.assertEqual(stale["fallback"], "previous_action_hold")
        self.assertTrue(recovery["input_ready"])
        validate_causal_timing(stale, 65, source.events)
        validate_causal_timing(recovery, 71, source.events)
        with self.assertRaisesRegex(RuntimeError, "boundaries"):
            validate_causal_timing(dict(stale, input_ready=True, fallback=None), 65, source.events)

    def test_baselines_require_all_fourteen_owned_cases_and_equal_targets(self):
        rows = [dict(event=e, clip=c, variant=v, horizon_frames=4, derived_reference_sha256=e+c)
                for e, c in review_cases() for v in ("teacher", "student")]
        self.assertEqual(len(baseline_cases(dict(rows=rows))), 14)
        for invalid in (rows[:-1], rows+[rows[0]]):
            with self.assertRaisesRegex(RuntimeError, "fourteen"):
                baseline_cases(dict(rows=invalid))
        wrong = copy.deepcopy(rows)
        wrong[0]["derived_reference_sha256"] = "different"
        with self.assertRaisesRegex(RuntimeError, "targets differ"):
            baseline_cases(dict(rows=wrong))

    def test_future_or_mismatched_weights_fail_before_student_construction(self):
        teacher = dict(path="/fixed/ckpt_32000.pt", bytes=1, sha256="a"*64)
        value = dict(schema="q1_low_latency_f0_history_encoder_v1", interface="f0_plus_history",
            human_real_future_frames=0, history_frames=10, history_frame_dim=226,
            selection_split="validation", test_used_for_selection=False, actual_updates=100,
            provenance=dict(teacher=teacher), frozen_teacher_policy_sha256="b"*64, decoder_sha256="c"*64)
        checkpoint_contract(value, teacher, "b"*64, "c"*64)
        for key, change in (("human_real_future_frames", 1), ("history_frames", 11),
                ("decoder_sha256", "wrong"), ("selection_split", "test"), ("actual_updates", 0)):
            with self.assertRaisesRegex(RuntimeError, "provenance differs"):
                checkpoint_contract(dict(value, **{key: change}), teacher, "b"*64, "c"*64)

    def test_media_uses_the_pinned_plant_interpreter_and_seven_new_cases(self):
        self.assertEqual(len(review_cases()), 7)
        self.assertEqual(review_cases()[:3], [("nominal", c) for c in FIXED])
        args = SimpleNamespace(config=Path("/config"), runtime_config=Path("/runtime"),
            output=Path("/out"), causal_training_root=Path("/r2"), evaluation_commit="a"*40)
        owner = dict(pid=1, start_ticks="2")
        for phase in ("replay", "media"):
            command = child_command(args, dict(primebot_python="/pinned/plant"), owner, review_cases()[0], phase)
            self.assertEqual(command[0], "/pinned/plant")
            self.assertEqual(command[command.index("--causal-training-root")+1], "/r2")
            self.assertEqual(command[command.index("--child")+1], phase)

    def test_causal_review_requires_real_source_gated_fit_settled_cost_and_reaping(self):
        with tempfile.TemporaryDirectory() as temp:
            root, parent = Path(temp)/"r2", Path(temp)/"r1"
            root.mkdir()
            parent.mkdir()

            def save(path, value):
                path.write_text(json.dumps(value))

            save(parent/"fit_result.json", dict(scope="synthetic_provenance_not_trained32k"))
            statistics = dict(fitted_split="train")
            save(root/"current_statistics.json", statistics)
            fit = dict(status="TRAINED_PENDING_CLOSED_LOOP", actual_updates=3, human_real_future_frames=0,
                provenance=dict(original_r1_fit=identity(parent/"fit_result.json")), current_statistics=statistics)
            save(root/"fit_result.json", fit)
            result = dict(stage="R2_F0_HISTORY_ENCODER_DISTILLATION", status="TRAINED_PENDING_CLOSED_LOOP",
                source_integrity="PASSED", cost_settlement=dict(state="SETTLED"),
                fit_result=identity(root/"fit_result.json"), controller=dict(pid=2147483647))
            guard = dict(status="WORKER_COMPLETED", cleanup=dict(worker_reaped=True, worker_return_code=0))
            save(root/"result.json", result)
            save(root/"fit.guard.json", guard)
            save(root/"authorization.json", dict(stage=result["stage"], evaluation_sources=[],
                training_source_inputs=[], controller_inputs=[]))
            for name in ("selected.pt", "training_trace.json", "reused_dataset.json"):
                save(root/name, {})
            self.assertEqual(len(trained_causal_evidence(root, parent)), 8)
            for key, changed in (("status", "RUNNING"), ("source_integrity", "PENDING"),
                    ("cost_settlement", dict(state="ACTIVE"))):
                save(root/"result.json", dict(result, **{key: changed}))
                with self.assertRaisesRegex(RuntimeError, "Wait for actual"):
                    trained_causal_evidence(root, parent)
            save(root/"result.json", result)
            save(root/"fit.guard.json", dict(guard, cleanup=dict(worker_reaped=False, worker_return_code=0)))
            with self.assertRaisesRegex(RuntimeError, "Wait for actual"):
                trained_causal_evidence(root, parent)


if __name__ == "__main__":
    unittest.main()
