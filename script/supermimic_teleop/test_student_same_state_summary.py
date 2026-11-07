"""Physical22 indices must be mapped by names into sparse canonical29 slots."""

import unittest
from student_same_state_summary import canonical_names, canonical_upper, failed_gate


class MappingTests(unittest.TestCase):
    def test_annotated_frozen_stems_are_parsed_without_import_execution(self):
        stems=tuple("joint"+str(i) for i in range(29))
        text='_JOINT_STEMS: tuple[str, ...] = '+repr(stems)+'\nJOINT_NAMES: tuple[str, ...] = tuple(f"{j}_joint" for j in _JOINT_STEMS)\n'
        self.assertEqual(canonical_names(text),[s+"_joint" for s in stems])
        with self.assertRaises(RuntimeError):canonical_names(text.replace('{j}_joint','{j}_different'))

    def fixture(self):
        names=["j"+str(i) for i in range(29)]
        ids=[15,16,17,18,22,23,24,25]
        present=[i in list(range(14))+ids for i in range(29)]
        case=dict(upper_joint_indices=list(range(14,22)),upper_joint_names=[names[i] for i in ids])
        return names,present,case,ids

    def test_native_upper_indices_are_not_canonical_slots(self):
        names,present,case,ids=self.fixture()
        self.assertFalse(present[case["upper_joint_indices"][0]])
        self.assertEqual(canonical_upper(names,present,case),ids)

    def test_unknown_duplicate_and_absent_names_rejected(self):
        for mode in ("unknown","duplicate","absent"):
            names,present,case,ids=self.fixture()
            if mode=="unknown":case["upper_joint_names"][0]="unknown"
            elif mode=="duplicate":case["upper_joint_names"][0]=case["upper_joint_names"][1]
            else:present[ids[0]]=False
            with self.assertRaises(RuntimeError):canonical_upper(names,present,case)

    def test_arbitrary_failure_cannot_bypass_completion_gate(self):
        with self.assertRaises(RuntimeError):failed_gate(dict(status="FAILED",error="different failure"))


if __name__=="__main__":unittest.main()
