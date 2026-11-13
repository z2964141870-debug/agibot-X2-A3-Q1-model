import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .allocation import PLAN, bind, sources, validate_available, validate_prefix
from .deadlines import profile
from .run import select
from .worker import physics_command


class Contract(unittest.TestCase):
    def test_restricted_cpuset_is_rejected(self):
        validate_available(set(range(8)))
        with self.assertRaises(ValueError):
            validate_available(set(range(7)))

    def test_thread_local_bind_verifies_actual_mask(self):
        with patch("os.sched_setaffinity", create=True) as setter, patch("os.sched_getaffinity", return_value={4,5}, create=True):
            self.assertEqual(bind("cpu"), [4,5])
            setter.assert_called_once_with(0, {4,5})
        with patch("os.sched_setaffinity", create=True), patch("os.sched_getaffinity", return_value={2,3}, create=True):
            with self.assertRaises(RuntimeError):
                bind("cpu")

    def test_preserve_native_observe_arguments_and_reject_arm(self):
        command = ["/project/x2_morph_native", "--mode", "observe", "--seconds", "55"]
        self.assertEqual(select(command, "/bin/taskset"), ["/bin/taskset", "--cpu-list", "0,1", *command])
        with self.assertRaises(ValueError):
            select(["x2_morph_native", "--mode", "armed"], "/bin/taskset")
        unrelated=["other", "--mode", "observe"]
        self.assertEqual(select(unrelated, "/bin/taskset"), unrelated)

    def test_only_exact_worker_modes_are_redirected(self):
        command=["python", "-m", "script.x2_joint_pipeline.worker", "fgp"]
        self.assertEqual(select(command, "taskset")[2], "script.x2_joint_alloc.worker")
        with self.assertRaises(ValueError):
            select(command[:-1]+["policy"], "taskset")
        with patch("shutil.which", return_value="/bin/taskset"):
            native=["python", "-m", "stage_tran_gp.worker", "--smpl", "model"]
            self.assertEqual(physics_command(native)[3:], native)

    def test_original_r1_prefix_cannot_unlock_new_allocation(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            receipt=dict(variant="bounded_affinity_r2", mode="prefix", plan=PLAN, source_sha256=sources())
            worker=dict(status="WORKER_CLEAN_EXIT", source_sha256=sources(), plan=PLAN, hardware_commands=0,
                        roles={k:dict(tid=i,cpus=PLAN[k]) for i,k in enumerate(("gpu","cpu","sender"))})
            (p/"allocation.json").write_text(json.dumps(receipt))
            (p/"affinity.json").write_text(json.dumps(worker))
            validate_prefix(p)
            bad=copy.deepcopy(worker);bad["roles"]["cpu"]["cpus"]=[2,3]
            (p/"affinity.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):validate_prefix(p)
            (p/"affinity.json").write_text(json.dumps(worker))
            receipt["source_sha256"]["worker.py"]="changed"
            (p/"allocation.json").write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):validate_prefix(p)

    def test_relaxed_budget_does_not_rewrite_actual_acceptances(self):
        rows=[dict(kind="policy_sample",rpc_ms=v,accepted=v<=20,reply=dict(ready=True)) for v in (19,22,27,32,36)]
        rows.append(dict(kind="policy_sample",rpc_ms=1,accepted=False,reply=dict(ready=False)))
        result=profile(rows)
        self.assertEqual(result["actual_native_accepted"],1)
        self.assertEqual(result["ready"],5)
        self.assertEqual([r["over_budget"] for r in result["hypothetical_deadline_only"]],[4,3,2,1])
        with self.assertRaises(ValueError):profile([])
        with self.assertRaises(ValueError):profile([dict(kind="policy_sample",rpc_ms=float('nan'),accepted=False,reply=dict(ready=True))])


if __name__ == "__main__":
    unittest.main()
