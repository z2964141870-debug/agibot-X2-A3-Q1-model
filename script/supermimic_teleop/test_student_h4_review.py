"""Reject wrong H4 selections and mismatched saved targets before new physics."""

import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from student_h4_review import child_command, matched_baselines, review_cases, selection_gate


def fixture():
    fit = dict(method="h4_only_continuation",actual_updates=3000,actual_adam_steps=[3000]*10,
        interface="human_prefix_adapter",allowed_request_horizons=[4],human_real_future_frames=4,
        maximum_preview_ns=80_000_000,selection_split="validation",test_used_for_selection=False,
        test_sources_read=False,normalization_refitted=False,selected_update=1500,selected_validation_risk=.09)
    trace = [dict(update=u,validation_risk=r) for u,r in ((0,.1),(1500,.09),(3000,.095))]
    return fit,trace


class H4ReviewTests(unittest.TestCase):
    def test_only_true_validation_minimum_can_be_replayed(self):
        fit,trace = fixture()
        self.assertEqual(selection_gate(fit,trace)["update"],1500)
        for key,value in (("selected_update",3000),("selection_split","test"),
                ("allowed_request_horizons",[4,5]),("human_real_future_frames",0),
                ("normalization_refitted",True),("actual_adam_steps",[2999]*10)):
            with self.subTest(key=key),self.assertRaises(RuntimeError):
                selection_gate(dict(fit,**{key:value}),trace)

    def test_missing_initial_or_actual_last_validation_is_rejected(self):
        fit,trace = fixture()
        for rows in (trace[1:],trace[:-1],list(reversed(trace))):
            with self.assertRaises(RuntimeError):
                selection_gate(fit,rows)

    def test_fourteen_h4_and_three_h10_saved_controls_are_reused_once(self):
        rows = [dict(event=e,clip=c,variant=v,horizon_frames=4,derived_reference_sha256=e+":"+c)
            for e,c in review_cases() for v in ("teacher","student")]
        rows += [dict(r,horizon_frames=10) for r in rows[:6:2]]
        self.assertEqual(len(matched_baselines(dict(rows=rows))),17)
        for transform in (lambda a:a[:-1],lambda a:a+[a[0]]):
            with self.assertRaises(RuntimeError):
                matched_baselines(dict(rows=transform(copy.deepcopy(rows))))
        rows[-1]["derived_reference_sha256"] = "wrong target"
        with self.assertRaises(RuntimeError):
            matched_baselines(dict(rows=rows))

    def test_children_use_existing_worker_path_and_only_the_changed_training_root(self):
        args = SimpleNamespace(config=Path("/config"),runtime_config=Path("/runtime"),output=Path("/new"),
            training_root=Path("/h4_selected1500"))
        command = child_command(args,{"primebot_python":"/python"},{"pid":123,"start_ticks":"456"},review_cases()[0],"replay")
        self.assertEqual(command[command.index("--training-root")+1],"/h4_selected1500")
        self.assertNotIn("--horizon",command)
        self.assertNotIn("--variant",command)
        with self.assertRaises(RuntimeError):
            child_command(args,{"primebot_python":"/python"},{"pid":123,"start_ticks":"456"},("unknown","clip"),"replay")


if __name__ == "__main__":
    unittest.main()
