"""Static terrain and exact rollout-body parity for the isolated CPU repair."""

import ast
from pathlib import Path
import unittest

from native_cpu import static_floor


class NativeCpuTests(unittest.TestCase):
    def test_static_child_terrain_and_world_floor_are_accepted(self):
        for body in (0, 1):
            self.assertEqual(static_floor([0], [body], [0, 0], [[0, 0, 0]],
                [[1, 0, 0, 0, 1, 0, 0, 0, 1]], 0), {0})

    def test_plane_under_dynamic_ancestor_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "static world"):
            static_floor([0], [1], [0, 1], [[0, 0, 0]], [[1, 0, 0, 0, 1, 0, 0, 0, 1]], 0)

    def test_elevated_or_tilted_floor_is_rejected(self):
        for position, rotation in (([0, 0, .1], [1, 0, 0, 0, 1, 0, 0, 0, 1]),
                ([0, 0, 0], [1, 0, 0, 0, 0, 1, 0, -1, 0])):
            with self.assertRaisesRegex(RuntimeError, "world z0"):
                static_floor([0], [1], [0, 0], [position], [rotation], 0)

    def test_replay_body_only_changes_contact_reader(self):
        root = Path(__file__).resolve().parent
        def function(path):
            tree = ast.parse(path.read_text())
            return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "rollout")
        original = function(root.parent / "q1_five_stage/native_review.py")
        repair = function(root / "native_cpu.py")
        for node in ast.walk(repair):
            if isinstance(node, ast.Name) and node.id == "StaticTerrainContactLoads":
                node.id = "ContactLoads"
        self.assertEqual(ast.dump(original, include_attributes=False), ast.dump(repair, include_attributes=False))


if __name__ == "__main__":
    unittest.main()
