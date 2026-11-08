"""CPU constants stay data-only; physical recovery probe changes no other logic."""

import ast
import copy
import hashlib
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/"q1_five_stage"), str(HERE.parent/"primebot_q1")]
from delivery import read
from student_recovery_resume import canonical_joint_names


def fixture(stems=None, expression='tuple(f"{j}_joint" for j in _JOINT_STEMS)'):
    stems = tuple("joint"+str(i) for i in range(29)) if stems is None else stems
    return 'raise RuntimeError("must never execute module")\n_JOINT_STEMS: tuple = '+repr(stems)+'\nJOINT_NAMES: tuple = '+expression+'\n'


class RecoveryResumeTests(unittest.TestCase):
    def test_literal_names_need_no_morphmimic_package_initialization(self):
        before = set(sys.modules)
        self.assertEqual(canonical_joint_names(fixture()), tuple("joint"+str(i)+"_joint" for i in range(29)))
        self.assertEqual(set(sys.modules), before)

    def test_dynamic_duplicate_and_changed_name_declarations_fail_closed(self):
        for text in (fixture(("a",)*29), fixture(expression='tuple(_JOINT_STEMS)'),
                fixture().replace(repr(tuple("joint"+str(i) for i in range(29))), 'arbitrary_call()')):
            with self.subTest(text=text[:45]), self.assertRaises((RuntimeError, ValueError)):
                canonical_joint_names(text)

    def test_probe_changes_only_the_cpu_joint_name_import(self):
        old = ast.parse((HERE/"student_recovery.py").read_text())
        new = ast.parse((HERE/"student_recovery_resume.py").read_text())
        old_probe = copy.deepcopy(next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "probe"))
        new_probe = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == "probe")
        replacement = ast.parse('JOINT_NAMES = canonical_joint_names((Path(runtime["morph_source"])/"src/morphmimic/constants.py").read_text())').body[0]
        matches = [i for i,n in enumerate(old_probe.body) if isinstance(n, ast.ImportFrom) and n.module == "morphmimic.constants"]
        self.assertEqual(len(matches), 1)
        old_probe.body[matches[0]] = replacement
        self.assertEqual(ast.dump(old_probe), ast.dump(new_probe))

    def test_the_five_previously_executed_sources_are_preserved(self):
        manifest = read(HERE.parents[1]/"data/manifests/q1_teacher_recovery_code_20261108.json")
        self.assertEqual(len(manifest["files"]), 5)
        for row in manifest["files"]:
            path = HERE/Path(row["path"]).name
            raw = path.read_bytes()
            self.assertEqual((len(raw), hashlib.sha256(raw).hexdigest()), (row["bytes"], row["sha256"]))


if __name__ == "__main__":
    unittest.main()
