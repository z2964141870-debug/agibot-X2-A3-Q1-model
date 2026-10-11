"""Live kernel readiness and critical evidence without torch/GPU execution."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from script.diagnostics import gpu_runtime_compare as runtime
from script.diagnostics import gpu_operator_compare as engine


class RuntimeTests(unittest.TestCase):
    def test_case_paths_bound_to_exact_worker_and_failed_predecessor(self):
        runtime.configure("E31")
        self.assertEqual(runtime.OUTPUT,engine.OUTPUT)
        self.assertEqual(runtime.LOGS,engine.LOGS)
        self.assertFalse(runtime.predecessor_allowed(dict(state=dict(status="STOPPED_REBOOT",boot_id="x")),"x"))
        runtime.configure("E30")

    def test_live_critical_filter_and_durable_ready(self):
        with tempfile.TemporaryDirectory() as folder:
            follow=runtime.KernelFollow(Path(folder),100)
            rows=[dict(MESSAGE="NVRM: Xid historical",__REALTIME_TIMESTAMP="99000000"),
                  dict(MESSAGE="NVRM: Xid new",__REALTIME_TIMESTAMP="101000000")]
            follow.child=type("Child",(),dict(stdout=io.StringIO("".join(json.dumps(x)+"\n" for x in rows))))()
            follow.consume()
            self.assertTrue(follow.ready.is_set())
            self.assertEqual(follow.critical,["NVRM: Xid new"])
            saved=[json.loads(x) for x in (Path(folder)/"kernel_live.jsonl").read_text().splitlines()]
            self.assertEqual([x["critical"] for x in saved],[False,True])


if __name__ == "__main__":unittest.main()
