"""Focused contracts for reuse, completed-review gating and the R2 launch."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from student_causal_training import PLAN_SHA, WALL_SECONDS, reused_dataset, saved_analysis_gate, worker_command
from common import identity


class StudentCausalTrainingTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def analysis(self):
        parent = self.root/"trained_student.json"
        self.save(parent, {"role": "synthetic preserved teacher-state fixture"})
        source = self.root/"review/result.json"
        self.save(source, dict(status="COMPLETED", source_integrity="PASSED", physical_cases=23,
            stage="R1_MATCHED_CLOSED_LOOP_CONTINUATION", controller=dict(pid=2147483647)))
        summary = dict(stage="R1_MATCHED_SAVED_REVIEW_ANALYSIS", total_cases=23, rows=[{}]*23,
            new_inference_calls=0, new_physical_transitions=0, new_training_updates=0,
            inputs=[identity(source), identity(parent)])
        output = self.root/"analysis"
        self.save(output/"summary.json", summary)
        self.save(output/"audit.json", dict(status="PASSED", source_inputs_unchanged=True, source_files=2))
        return output, parent, source, summary

    def test_source_gated_analysis_is_required_and_remains_bound_to_the_same_r1(self):
        output, parent, source, summary = self.analysis()
        self.assertEqual(len(saved_analysis_gate(output, [identity(parent)])), 4)
        self.save(parent, {"role": "changed"})
        with self.assertRaisesRegex(RuntimeError, "sources changed"):
            saved_analysis_gate(output, [identity(parent)])
        output, parent, source, summary = self.analysis()
        summary["total_cases"] = 22
        self.save(output/"summary.json", summary)
        with self.assertRaisesRegex(RuntimeError, "real completed R1"):
            saved_analysis_gate(output, [identity(parent)])

    def test_running_review_cannot_unlock_training_with_a_passed_looking_summary(self):
        output, parent, source, summary = self.analysis()
        value = json.loads(source.read_text())
        value["status"] = "RUNNING"
        self.save(source, value)
        summary["inputs"][0] = identity(source)
        self.save(output/"summary.json", summary)
        with self.assertRaisesRegex(RuntimeError, "must complete"):
            saved_analysis_gate(output, [identity(parent)])

    def data(self):
        jobs = [dict(clip="clip"+str(i), split="train" if i < 12 else "validation", maximum_steps=40)
                for i in range(18)]
        self.save(self.root/"collection_plan.json", dict(jobs=jobs))
        clips = []
        for i, job in enumerate(jobs):
            path = self.root/f"collection_{i:02d}.npz"
            np.savez(path, actual=np.zeros((41, 79)), canonical_action=np.zeros((40, 29)))
            clips.append(dict(clip_id=i, clip=job["clip"], split=job["split"], trace=identity(path)))
        files = {}
        for split in ("train", "validation"):
            selected = [i for i, job in enumerate(jobs) if job["split"] == split]
            count = len(selected)*6
            current = np.zeros((count, 10, 226), dtype=np.float32)
            current[..., 223:] = [1., 0., .02]
            data = dict(full_goal=np.zeros((count, 2254), dtype=np.float32),
                history=np.zeros((count, 930), dtype=np.float32), f0=current,
                teacher_action=np.zeros((count, 29), dtype=np.float32), actual_state=np.zeros((count, 79)),
                prefix=np.tile([10, 5, 4, 2, 1], (count, 1)),
                frame=np.tile(np.arange(10, 22, 2), len(selected)), clip_id=np.repeat(selected, 6))
            path = self.root/(split+"_data.npz")
            np.savez(path, **data)
            files[split] = identity(path)
        self.save(self.root/"dataset_receipt.json", dict(status="PACKING_ACTION_PARITY_PASSED",
            test_sources_read=False, files=files))
        self.save(self.root/"collection_result.json", dict(status="PASSED", test_sources_read=False, clips=clips))
        return jobs

    def test_reuse_keeps_all_owned_source_groups_and_passes_only_causal_model_inputs(self):
        jobs = self.data()
        receipt, values = reused_dataset(self.root, jobs, load=True)
        self.assertEqual(receipt["new_physical_transitions"], 0)
        self.assertEqual(receipt["validation"]["train"]["source_clips"], 12)
        self.assertEqual(receipt["validation"]["validation"]["source_clips"], 6)
        for split in values.values():
            self.assertEqual(set(split), {"f0", "history", "teacher_action", "clip_id"})

    def test_updated_dataset_hash_alone_cannot_replace_actual_robot_state_provenance(self):
        jobs = self.data()
        path = self.root/"train_data.npz"
        with np.load(path, allow_pickle=False) as arrays:
            data = dict(arrays)
        data["actual_state"][0, 0] += 1.
        np.savez(path, **data)
        receipt = json.loads((self.root/"dataset_receipt.json").read_text())
        receipt["files"]["train"] = identity(path)
        self.save(self.root/"dataset_receipt.json", receipt)
        with self.assertRaisesRegex(ValueError, "actual rollout"):
            reused_dataset(self.root, jobs)

    def test_plan_phase_and_worker_interpreter_preserve_the_published_contract(self):
        plan = Path(__file__).with_name("student_causal_plan_20261107.json")
        self.assertEqual(hashlib.sha256(plan.read_bytes()).hexdigest(), PLAN_SHA)
        self.assertLess(WALL_SECONDS+5, 1800)
        training = self.root/"original"
        self.save(training/"e1024_train_plan.json", dict(source="/pinned/morph"))
        args = SimpleNamespace(training_root=self.root/"r1", output=self.root/"r2", causal_plan=plan)
        audit = dict(records=[dict(robot="q1", checkpoint=dict(path="/pinned/ckpt_32000.pt"))])
        owner = dict(gpu="GPU-fixture", pid=1234, start_ticks="5678")
        with patch("student_causal_training.TRAINING", training):
            command = worker_command(args, dict(morph_python="/pinned/morph-python"), audit, owner, 123456)
        self.assertEqual(command[0], "/pinned/morph-python")
        self.assertIn("--gpu-owner-ticks", command)
        self.assertEqual(command[command.index("--deadline-ns")+1], "123456")


if __name__ == "__main__":
    unittest.main()
