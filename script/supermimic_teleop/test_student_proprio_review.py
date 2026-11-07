"""CPU checks for selected residual loading and unchanged causal action semantics."""

import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from student_causal_review_worker import CausalReviewSession
from student_fit import cpu_state, module_sha
from student_models import current_student
from student_proprio_models import residual_student
from student_proprio_review import child_command, review_cases, zero_identity
from student_proprio_review_worker import ProprioReviewSession, checkpoint_contract, load_student
from test_student_proprio import fixtures


def saved_fixture(nonzero=True):
    native, previous, origins, _ = fixtures()
    model = residual_student(native, previous, "causal_history")
    if nonzero:
        with torch.no_grad():
            model.adapter[-1].weight.fill_(.01)
    teacher = dict(path="/teacher", bytes=1, sha256="a"*64)
    previous.update(schema="q1_low_latency_f0_history_encoder_v1", selected_update=500,
        interface="f0_plus_history", human_real_future_frames=0, history_frames=10,
        history_frame_dim=226, provenance=dict(teacher=teacher))
    payload = dict(schema="q1_f0_proprio_residual_v1", interface="f0_plus_history", arm="causal_history",
        human_real_future_frames=0, history_frames=10, history_frame_dim=226,
        robot_history_features=930, adapter_hidden_width=128,
        residual_rule="z+(1-abs(z))*tanh(adapter(history))", selection_split="validation",
        test_used_for_selection=False, normalization_refitted=False, actual_updates=3000,
        actual_adam_steps=[3000]*4, selected_update=3000, provenance=dict(teacher=teacher),
        frozen_teacher_policy_sha256=module_sha(native.policy), decoder_sha256=module_sha(native.policy.decoder),
        current_statistics=previous["current_statistics"], base_encoder=cpu_state(model.encoder),
        base_encoder_sha256=module_sha(model.encoder), adapter=cpu_state(model.adapter))
    return native, previous, payload, model, origins


class ProprioReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_wrong_arm_future_teacher_selection_and_exposure_are_rejected(self):
        _, previous, payload, _, _ = saved_fixture()
        def check(value):
            checkpoint_contract(value, previous["provenance"]["teacher"],
                previous["frozen_teacher_policy_sha256"], previous["decoder_sha256"], "causal_history")
        check(payload)
        for key, value in (("arm", "zero_history"), ("human_real_future_frames", 4),
                ("selected_update", 0), ("selection_split", "test"), ("test_used_for_selection", True),
                ("actual_adam_steps", [2999]*4), ("frozen_teacher_policy_sha256", "x"),
                ("decoder_sha256", "x")):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                check(dict(payload, **{key:value}))
        bad = copy.deepcopy(payload)
        bad["provenance"]["teacher"]["sha256"] = "b"*64
        with self.assertRaises(RuntimeError):
            check(bad)

    def test_reconstructed_nonzero_checkpoint_has_exact_saved_model_actions(self):
        native, _, payload, model, origins = saved_fixture()
        restored = load_student(native, payload)
        history = torch.from_numpy(origins[0]["train"]["history"])
        goal = torch.from_numpy(origins[0]["train"]["f0"])
        self.assertTrue(torch.equal(restored(history, goal), model(history, goal)))
        self.assertFalse(restored.training)
        self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
        bad = copy.deepcopy(payload)
        bad["base_encoder"][next(iter(bad["base_encoder"]))].add_(1.)
        with self.assertRaisesRegex(RuntimeError, "encoder digest"):
            load_student(native, bad)

    def test_zero_adapter_is_rejected_from_identical_replay(self):
        native, _, payload, _, _ = saved_fixture(nonzero=False)
        with self.assertRaisesRegex(RuntimeError, "reuse the old f0"):
            load_student(native, payload)

    def test_zero_selection_requires_equal_base_normalization_and_decoder(self):
        native, previous, payload, _, origins = saved_fixture(nonzero=False)
        payload.update(arm="zero_history", selected_update=0)
        proof = zero_identity(payload, previous)
        self.assertEqual(proof["base_tensors_exact"], 8)
        base = current_student(native, previous["current_statistics"])
        base.encoder.load_state_dict(previous["encoder"])
        zero = residual_student(native, previous, "zero_history")
        history = torch.from_numpy(origins[0]["train"]["history"])
        goal = torch.from_numpy(origins[0]["train"]["f0"])
        self.assertTrue(torch.equal(base(history, goal), zero(history, goal)))
        for key, value in (("selected_update", 500), ("decoder_sha256", "other"),
                ("current_statistics", {}), ("history_frames", 9)):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                zero_identity(dict(payload, **{key:value}), previous)
        changed = copy.deepcopy(payload)
        changed["base_encoder"][next(iter(changed["base_encoder"]))].add_(.1)
        with self.assertRaises(RuntimeError):
            zero_identity(changed, previous)

    def test_action_and_event_start_are_the_exact_inherited_original_methods(self):
        self.assertIs(ProprioReviewSession.action, CausalReviewSession.action)
        self.assertIs(ProprioReviewSession.start, CausalReviewSession.start)
        self.assertIs(ProprioReviewSession.preview_config, CausalReviewSession.preview_config)

    def test_children_are_bounded_to_the_registered_seven_cases(self):
        args = SimpleNamespace(config=Path("/config"), runtime_config=Path("/runtime"), output=Path("/new"))
        command = child_command(args, {"primebot_python":"/python"}, {"pid":123,"start_ticks":"456"},
            review_cases()[0], "replay")
        self.assertNotIn("--horizon", command)
        self.assertNotIn("--variant", command)
        self.assertEqual(command[command.index("--clip")+1], review_cases()[0][1])
        for case, phase in ((("unknown", "clip"), "replay"), (review_cases()[0], "train")):
            with self.assertRaises(RuntimeError):
                child_command(args, {"primebot_python":"/python"}, {"pid":123,"start_ticks":"456"}, case, phase)


if __name__ == "__main__":
    unittest.main()
