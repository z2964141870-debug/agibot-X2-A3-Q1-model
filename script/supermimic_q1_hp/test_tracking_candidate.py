"""Offline construction checks; these do not validate Torch or physics."""

import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest

from build_tracking_candidate import (
    HERE, PINNED, TASK, build_candidate, identity, relocate, reward_error_fields,
)


MDP_FIXTURE = '''
def reward(error_ori, error_linvel, error_angvel, terms=False):
    tracking = error_ori + error_linvel + error_angvel
    if terms:
        return tracking, {"track": tracking, "e_body_ori": error_ori}
    return tracking
'''


def tree_identity(root):
    return {str(p.relative_to(root)): identity(p) for p in root.rglob("*") if p.is_file()}


class RewardTransformTests(unittest.TestCase):
    def test_only_adds_two_existing_error_fields(self):
        result = ast.parse(reward_error_fields(MDP_FIXTURE))
        term = next(n for n in ast.walk(result) if isinstance(n, ast.Dict))
        self.assertEqual([k.value for k in term.keys][-2:], ["e_body_linvel", "e_body_angvel"])
        self.assertEqual([v.id for v in term.values][-2:], ["error_linvel", "error_angvel"])
        del term.keys[-2:]
        del term.values[-2:]
        self.assertEqual(ast.dump(result), ast.dump(ast.parse(MDP_FIXTURE)))

    def test_reward_and_original_terms_unchanged(self):
        original, candidate = {}, {}
        exec(MDP_FIXTURE, original)
        exec(reward_error_fields(MDP_FIXTURE), candidate)
        for values in ((0., 0., 0.), (.1, 1., 2.), (3., 4., 5.)):
            self.assertEqual(original["reward"](*values), candidate["reward"](*values))
            before, after = original["reward"](*values, True), candidate["reward"](*values, True)
            self.assertEqual(before[0], after[0])
            self.assertEqual(before[1], {k: after[1][k] for k in before[1]})
            self.assertEqual((after[1]["e_body_linvel"], after[1]["e_body_angvel"]), values[1:])

    def test_ambiguous_or_repeated_transformation_rejected(self):
        for source in ("value = {}", MDP_FIXTURE+MDP_FIXTURE, reward_error_fields(MDP_FIXTURE)):
            with self.subTest(source=source), self.assertRaises(ValueError):
                reward_error_fields(source)

    def test_policy_schema_extends_without_mutating_base(self):
        tree = ast.parse((HERE/"tracking_mdp.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "policy_input_spec")
        base = {"schema_version": 1, "obs_dim": 80, "observation_fields": ["root_xy"], "goal_frame_dim": 59}
        namespace = {"base_policy_input_spec": lambda _: copy.deepcopy(base), "NUM_J": 22,
                     "_JOINT_ERROR": True, "_ERROR_FIELD": "reference_minus_robot_joint_position"}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "input_spec", "exec"), namespace)
        candidate = namespace["policy_input_spec"]("root_xy")
        self.assertEqual((candidate["schema_version"], candidate["obs_dim"]), (2, 102))
        self.assertEqual(candidate["goal_frame_dim"], 59)
        self.assertEqual(len(base["observation_fields"]), 1)
        namespace["_JOINT_ERROR"] = False
        self.assertEqual(namespace["policy_input_spec"]("root_xy"), base)


class CandidateBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source, self.output = self.root/"frozen", self.root/"candidate"
        self.source.mkdir()
        files = {"LICENSE": "Apache fixture\n", "NOTICE": "Notice fixture\n",
                 "LICENSES/MIT-supermimic.txt": "MIT fixture\n",
                 "src/unisupermimic/rl/algos/supermimic/learner.py": "value = 'untouched'\n"}
        for name, text in files.items():
            path = self.source/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        manifest = {"commit": PINNED, "files": {name: identity(self.source/name) for name in files}}
        (self.source/"upstream_source.json").write_text(json.dumps(manifest))
        task = self.source/TASK
        task.mkdir(parents=True)
        (task/"mdp.py").write_text(MDP_FIXTURE)
        (task/"motion.py").write_text("value = 'original motion'\n")
        (task/"control.json").write_text(json.dumps({"scene": str(self.source/"q1_assets/scene.xml")}))
        (self.source/"q1_assets").mkdir()
        (self.source/"q1_assets/scene.xml").write_text("<mujoco/>\n")
        (self.source/"q1_adapter.json").write_text(json.dumps({"joints": list(range(22)), "scene": str(self.source/"q1_assets/scene.xml")}))
        self.config = {"algo": {"num_envs": 4096, "pi_batch": 1024}, "env": {"scene": str(self.source/"q1_assets/scene.xml")}}
        (self.source/"q1_config.yaml").write_text(json.dumps(self.config))

    def test_independent_build_preserves_source_learner_assets_and_licenses(self):
        before = tree_identity(self.source)
        config_before = copy.deepcopy(self.config)
        result = build_candidate(self.source, self.output, self.config)
        self.assertEqual(before, tree_identity(self.source))
        self.assertEqual(self.config, config_before)
        for name in ("LICENSE", "NOTICE", "LICENSES/MIT-supermimic.txt",
                     "src/unisupermimic/rl/algos/supermimic/learner.py", "q1_assets/scene.xml"):
            self.assertEqual(identity(self.output/name), before[name])
        self.assertEqual((self.output/TASK/"motion_base.py").read_bytes(), (self.source/TASK/"motion.py").read_bytes())
        self.assertEqual((self.output/TASK/"mdp.py").read_bytes(), (HERE/"tracking_mdp.py").read_bytes())
        self.assertEqual(json.loads((self.output/"q1_config.yaml").read_text())["env"]["scene"], str(self.output/"q1_assets/scene.xml"))
        self.assertEqual(json.loads((self.output/TASK/"control.json").read_text())["scene"], str(self.output/"q1_assets/scene.xml"))
        self.assertTrue(result["options"]["joint_error_observation"])
        self.assertFalse(result["training_started"])
        self.assertIn("NOT_PHYSICALLY_VALIDATED", result["status"])

    def test_sampler_only_keeps_observation_and_reward_options_disabled(self):
        options = build_candidate(self.source, self.output, self.config, "sampler_only")["options"]
        self.assertTrue(options["repair_transition_bins"])
        self.assertFalse(options["joint_error_observation"])
        self.assertFalse(options["joint_tracking_reward"])
        self.assertFalse(options["future_goal_padding_changed"])
        self.assertFalse(options["controller_or_reset_changed"])

    def test_exact_path_components_only_relocated(self):
        old, new = str(self.source), str(self.output)
        values = [old, old+"/asset", old+"_other/asset", {"value": "prefix "+old}]
        self.assertEqual(relocate(values, self.source, self.output), [new, new+"/asset", old+"_other/asset", {"value": "prefix "+old}])

    def test_source_directory_alias_relocated_to_independent_runtime(self):
        alias = self.root/"source_alias"
        alias.symlink_to(self.source, target_is_directory=True)
        self.config["env"]["scene"] = str(alias/"q1_assets/scene.xml")
        build_candidate(alias, self.output, self.config)
        result = json.loads((self.output/"q1_config.yaml").read_text())
        self.assertEqual(result["env"]["scene"], str(self.output/"q1_assets/scene.xml"))

    def test_changed_upstream_fails_without_creating_destination(self):
        (self.source/"LICENSE").write_text("unexpected change\n")
        with self.assertRaisesRegex(ValueError, "identity differs"):
            build_candidate(self.source, self.output, self.config)
        self.assertFalse(self.output.exists())

    def test_wrong_source_commit_fails_without_creating_destination(self):
        path = self.source/"upstream_source.json"
        manifest = json.loads(path.read_text())
        manifest["commit"] = "0"*40
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "source commit"):
            build_candidate(self.source, self.output, self.config)
        self.assertFalse(self.output.exists())

    def test_existing_or_nested_destination_refused(self):
        for path in (self.source, self.source/"new", self.root):
            with self.subTest(path=path), self.assertRaises((ValueError, FileExistsError)):
                build_candidate(self.source, path, self.config)
        self.assertFalse((self.source/"new").exists())

    def test_configuration_and_joint_dimensions_checked_before_copy(self):
        self.config["algo"]["num_envs"] = 1024
        with self.assertRaisesRegex(ValueError, "4096"):
            build_candidate(self.source, self.output, self.config)
        self.config["algo"]["num_envs"] = 4096
        (self.source/"q1_adapter.json").write_text(json.dumps({"joints": list(range(31))}))
        with self.assertRaisesRegex(ValueError, "22-joint"):
            build_candidate(self.source, self.output, self.config)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
