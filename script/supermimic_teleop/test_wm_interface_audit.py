"""Source identity failures and resolution of the effective native step override."""

from pathlib import Path
import tempfile
import unittest

from wm_interface_audit import function_index, identity, source_records, verified_source


class WorldModelSourceAuditTests(unittest.TestCase):
    def test_same_size_source_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.py"
            path.write_text("def step(x):\n    return x + 1\n")
            expected = identity(path)
            verified_source(path, expected)
            path.write_text("def step(x):\n    return x + 2\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                verified_source(path, expected)

    def test_conflicting_nested_authorization_records_are_rejected(self):
        row = dict(path="/frozen/model.py", bytes=12, sha256="a" * 64)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            source_records(dict(old=[row], nested=dict(source=dict(row, sha256="b" * 64))))
        self.assertEqual(source_records([row, dict(row)]), {row["path"]:row})

    def test_native_override_keeps_class_scope_and_grad_decorators(self):
        source = """class WorldModel:
    @torch.no_grad()
    def step(self, state, action, ctx):
        return state
def make_world_model(spec):
    class NativeWorldModel(WorldModel):
        def step(self, state, action, ctx):
            return action
    return NativeWorldModel
"""
        index = {row["name"]:row for row in function_index(source)}
        self.assertEqual(index["WorldModel.step"]["decorators"], ["torch.no_grad()"])
        native = index["make_world_model.NativeWorldModel.step"]
        self.assertEqual(native["decorators"], [])
        self.assertEqual(native["arguments"], ["self", "state", "action", "ctx"])


if __name__ == "__main__":
    unittest.main()
