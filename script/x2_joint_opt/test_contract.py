import json
from pathlib import Path
import tempfile
import unittest

from .config import PLANS, settings, sources, validate_prefix
from .profile import Recorder
from .run import select


class Contract(unittest.TestCase):
    def test_only_registered_config(self):
        settings(dict(variant="dedicated_hosts", release="a"*40))
        for c in (dict(variant="other",release="a"*40),dict(variant="r2_profile",release="short")):
            with self.assertRaises(ValueError):settings(c)

    def test_native_observe_only_and_own_worker_modes(self):
        c=["/project/x2_morph_native","--mode","observe"]
        result=select(c,Path("out"),"dedicated_hosts")
        self.assertEqual(result[1:3],["--cpu-list","1"])
        self.assertEqual(result[-len(c):],c)
        with self.assertRaises(ValueError):select(c[:-1]+["armed"],Path("out"),"r2_profile")
        with self.assertRaises(ValueError):select(["python","-m","script.x2_joint_pipeline.worker","policy"],Path("out"),"r2_profile")

    def test_profile_preserves_values_exceptions_and_bound(self):
        rec=Recorder(limit=2)
        value=object()
        self.assertIs(rec.wrap("ok",lambda:value)(),value)
        with self.assertRaisesRegex(ValueError,"original"):
            rec.wrap("bad",lambda:(_ for _ in ()).throw(ValueError("original")))()
        self.assertEqual([r["stage"] for r in rec.rows],["ok","bad"])
        self.assertTrue(all(r["wall_ms"]>=0 and r["thread_cpu_ms"]>=0 for r in rec.rows))
        with self.assertRaisesRegex(RuntimeError,"capacity"):
            rec.wrap("extra",lambda:None)()

    def test_candidate_prefix_is_not_interchangeable(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);config=dict(variant="dedicated_hosts",release="a"*40)
            (root/"opt_config.json").write_text(json.dumps(dict(config=config,mode="prefix",source_sha256=sources(),source_unchanged=True)))
            rows=[dict(role=role,tid=i,cpus=PLANS[config["variant"]][role]) for i,role in enumerate(("gpu","cpu","sender"))]
            (root/"kernel_binding_fgp.jsonl").write_text("\n".join(map(json.dumps,rows)))
            validate_prefix(root,config)
            with self.assertRaises(ValueError):validate_prefix(root,dict(variant="r2_profile",release="a"*40))
            rows[-1]["cpus"]=[2,3]
            (root/"kernel_binding_fgp.jsonl").write_text("\n".join(map(json.dumps,rows)))
            with self.assertRaises(ValueError):validate_prefix(root,config)


if __name__ == "__main__":
    unittest.main()
