"""Dependency-free boundary tests, not a Torch/Gloo execution or numerical proof."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from collective_transport import HostCollectives, installed_transport
from verify_collectives_cpu import UPSTREAM_HASHES, source_contract


class CollectiveGuards(unittest.TestCase):
    def setUp(self):
        self.api = SimpleNamespace(
            is_initialized=lambda: True,
            get_backend=lambda: "gloo",
            ReduceOp=SimpleNamespace(SUM="sum", MIN="min", MAX="max", PRODUCT="product"),
        )
        self.torch = SimpleNamespace(distributed=self.api, strided="strided")
        self.modules = patch.dict("sys.modules", {"torch": self.torch,
                                                  "torch.distributed": self.api})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_requires_initialized_gloo(self):
        self.api.is_initialized = lambda: False
        with self.assertRaises(RuntimeError):
            HostCollectives(self.api)
        self.api.is_initialized = lambda: True
        self.api.get_backend = lambda: "nccl"
        with self.assertRaises(RuntimeError):
            HostCollectives(self.api)

    def test_rejects_unaudited_calls_before_transport(self):
        bridge = HostCollectives(self.api)
        with self.assertRaises(TypeError):
            bridge.all_reduce(None, async_op=True)
        with self.assertRaises(AttributeError):
            bridge.barrier()
        with self.assertRaises(ValueError):
            bridge.all_reduce(None, op=self.api.ReduceOp.PRODUCT)
        self.assertTrue(all(count == 0 for count in bridge.calls.values()))

    def test_rejects_unsupported_storage(self):
        bridge = HostCollectives(self.api)
        for layout, device in (("sparse", "cpu"), ("strided", "mps")):
            tensor = SimpleNamespace(layout=layout, device=SimpleNamespace(type=device))
            with self.assertRaises(ValueError):
                bridge._host(tensor)
        self.assertEqual(bridge.staged_input_bytes, 0)

    def test_rejects_empty_duplicate_or_foreign_module_sets(self):
        valid = SimpleNamespace(__name__="morphmimic.train.distributed", dist=self.api)
        foreign = SimpleNamespace(__name__="unrelated.module", dist=self.api)
        altered = SimpleNamespace(__name__="morphmimic.train.trainer", dist=object())
        for modules in ([], [valid, valid], [foreign], [valid, altered]):
            with self.assertRaises(ValueError):
                with installed_transport(modules):
                    self.fail("Invalid module aliases were installed")
            self.assertIs(valid.dist, self.api)

    def test_alias_lifecycle_and_global_api(self):
        module = SimpleNamespace(__name__="morphmimic.train.distributed", dist=self.api)
        with installed_transport([module]) as bridge:
            self.assertIs(module.dist, bridge)
            self.assertIs(self.torch.distributed, self.api)
        self.assertIs(module.dist, self.api)
        with self.assertRaisesRegex(LookupError, "fixture"):
            with installed_transport([module]):
                raise LookupError("fixture")
        self.assertIs(module.dist, self.api)

    def test_pinned_source_hash_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src/morphmimic/train"
            source.mkdir(parents=True)
            (source / "distributed.py").write_text("dist.barrier()\n")
            with self.assertRaisesRegex(ValueError, "Pinned upstream changed"):
                source_contract(root)

    def test_dist_call_surface_checked_in_addition_to_hash(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src/morphmimic/train"
            source.mkdir(parents=True)
            content = b"dist.barrier()\n"
            (source / "distributed.py").write_bytes(content)
            hashes = {"distributed.py": hashlib.sha256(content).hexdigest()}
            with patch("verify_collectives_cpu.UPSTREAM_HASHES", hashes):
                with self.assertRaisesRegex(ValueError, "Unaudited upstream collective"):
                    source_contract(root)

    def test_source_contract_accepts_only_audited_call_surface(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src/morphmimic/train"
            source.mkdir(parents=True)
            content = b"dist.all_reduce(tensor)\ndist.get_world_size()\n"
            for name in UPSTREAM_HASHES:
                (source / name).write_bytes(content)
            hashes = {name: hashlib.sha256(content).hexdigest() for name in UPSTREAM_HASHES}
            with patch("verify_collectives_cpu.UPSTREAM_HASHES", hashes):
                rows = source_contract(root)
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["direct_dist_calls"] == ["all_reduce", "get_world_size"]
                                for row in rows))


if __name__ == "__main__":
    unittest.main()
