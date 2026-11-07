"""Adversarial selection and frozen-normalization checks for coverage replay."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from student_coverage_review import changed_encoder, coverage_evidence, selection_gate


def fixture():
    trace = [dict(update=u, validation_risk=r, fixed_anchor_r1_risk=a, fixed_anchor_mixed_risk=b)
        for u,r,a,b in [(0,.1,.2,.3),(500,.06,.1,.2),(1000,.05,.21,.2),(3000,.07,.1,.2)]]
    fit = dict(method="owned_source_coverage", actual_updates=3000, actual_adam_steps=[3000]*6,
        selection_split="validation", test_used_for_selection=False, human_real_future_frames=0,
        normalization_refitted=False, selected_update=500, selected_validation_risk=.06,
        initial_fixed_anchor_risks=[.2,.3], selected_fixed_anchor_risks=[.1,.2])
    return fit, trace


class CoverageReviewTests(unittest.TestCase):
    def test_expanded_minimum_cannot_override_regressed_fixed_anchor(self):
        fit, trace = fixture()
        self.assertEqual(selection_gate(fit, trace)["update"], 500)
        with self.assertRaisesRegex(RuntimeError, "eligible"):
            selection_gate(dict(fit, selected_update=1000, selected_validation_risk=.05), trace)

    def test_last_untrained_future_and_test_selected_checkpoints_are_rejected(self):
        fit, trace = fixture()
        for key, value in (("selected_update",3000), ("selected_update",0), ("actual_updates",0),
                ("human_real_future_frames",1), ("selection_split","test"), ("normalization_refitted",True),
                ("actual_adam_steps",[2999]*6)):
            with self.subTest(key=key,value=value), self.assertRaises(RuntimeError):
                selection_gate(dict(fit, **{key:value}), trace)

    def test_changed_selected_cpu_encoder_preserves_statistics_and_frozen_metadata(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            roots = [Path(temp)/n for n in ("current","previous")]
            for p in roots:p.mkdir()
            value = dict(schema="fixture", interface="f0_plus_history", current_statistics={"count":1},
                human_real_future_frames=0, history_frames=10, history_frame_dim=226,
                decoder_sha256="a"*64, frozen_teacher_policy_sha256="b"*64,
                provenance={"teacher":{"sha256":"c"*64}}, selected_update=500,
                encoder={"mean":torch.zeros(2),"scale":torch.ones(2),"weight":torch.zeros(2)})
            torch.save(value, roots[1]/"selected.pt")
            with self.assertRaisesRegex(RuntimeError, "changed finite"):
                torch.save(value, roots[0]/"selected.pt")
                changed_encoder(*roots)
            modified = copy.deepcopy(value)
            modified["encoder"]["weight"][0] = .1
            torch.save(modified, roots[0]/"selected.pt")
            self.assertEqual(changed_encoder(*roots)["status"], "CHANGED_SELECTED_ENCODER")
            modified["encoder"]["scale"][0] = 2.
            torch.save(modified, roots[0]/"selected.pt")
            with self.assertRaisesRegex(RuntimeError, "normalization"):
                changed_encoder(*roots)

    def test_running_or_unsettled_training_cannot_enter_new_review(self):
        fit, trace = fixture()
        fit["status"] = "TRAINED_PENDING_CLOSED_LOOP"
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            objects={"fit_result.json":fit,"training_trace.json":trace,"fit.guard.json":{},
                "authorization.json":{"stage":"R3_OWNED_SOURCE_COVERAGE"},
                "result.json":{"stage":"R3_OWNED_SOURCE_COVERAGE","status":"RUNNING"}}
            for name,value in objects.items():(root/name).write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError,"real coverage completion"):
                coverage_evidence(root,{})


if __name__ == "__main__":
    unittest.main()
