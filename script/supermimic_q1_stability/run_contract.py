"""Bind the independent action-domain trial to the completed S02 evidence."""

import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
PARTIAL = HERE.parent / "supermimic_q1_partial"
sys.path.insert(0, str(PARTIAL))
import parent


EXPERIMENT = "PHUMA_R02D_bounded_partial5000_extend7000_20261025"
CONFIG = HERE / "bounded_R02D_20261025.json"


def verified_prerequisites(root):
    base = Path(root) / "data/supermimic_for_q1"
    paths = [base / name for name in (
        "PHUMA_S02_controls_result_20261025.json",
        "PHUMA_S02_R02C_saved1500_audit_20261025.json",
        "PHUMA_S02_bounded_H32_probe_20261025.json")]
    controls, audit, probe = [json.loads(path.read_text()) for path in paths]
    if (controls["status"] != "PASSED" or controls["tests_run"] != 5
            or any(controls[key] for key in ("failures", "errors", "skipped"))):
        raise ValueError("Require all five native S02 controls, with no skipped tests")
    if (audit["status"] != "FAILED_R02C_SAVED1500_AUDIT_PASSED"
            or len(audit["gates"]) != 15 or not all(audit["gates"].values())
            or audit["training_terminal"] != "FAILED_UNCHANGED"):
        raise ValueError("Require the completed saved1500 audit without changing the failed trial")
    if (probe["status"] != "BOUNDED_SAVED_POLICY_H32_PROBE_PASSED"
            or len(probe["gates"]) != 7 or not all(probe["gates"].values())
            or probe["new_optimizer_updates"] != 0 or probe["new_physics_control_steps"] != 2048
            or probe["horizon"] != 32 or probe["batch_size"] != 128):
        raise ValueError("Require the completed real-replay H32 forward/backward probe")
    sources = {Path(row["path"]).resolve(): row for row in controls["sources"]}
    for name in ("bounded_policy.py", "test_bounded_policy.py", "audit_failed_checkpoint.py",
                 "probe_bounded_imagination.py"):
        if sources.get(HERE / name) != parent.identity(HERE / name):
            raise ValueError("Native controls do not bind the current S02 producer: " + name)
    if (audit["source"] != parent.identity(HERE / "audit_failed_checkpoint.py")
            or parent.identity(HERE / "probe_bounded_imagination.py") not in probe["inputs"]
            or parent.identity(paths[1]) not in probe["inputs"]):
        raise ValueError("S02 audit/probe producer or audit binding differs")
    records = [*controls["sources"], controls["log"], audit["source"],
               *audit["inputs"], *probe["inputs"], *[parent.identity(path) for path in paths]]
    unique = {row["path"]: row for row in records}
    for row in unique.values():
        parent.check_identity(row)
    return {"controls": parent.identity(paths[0]), "saved1500_audit": parent.identity(paths[1]),
            "h32_probe": parent.identity(paths[2]),
            "frozen_evidence": [unique[key] for key in sorted(unique)]}


def correction_declaration():
    return {"version": "Q1_REAL_AND_IMAGINED_CLIP20_V1", "action_min": -20., "action_max": 20.,
            "raw_nonfinite_action": "REJECT_BEFORE_CLIPPING",
            "nonfinite_gradient": "REJECT_BEFORE_EACH_OF_THREE_ADAM_STEPS",
            "finite_adam_updates": "UNCHANGED",
            "policy_state_dict": "PRESERVED_KEYS_AND_SHAPES",
            "source": parent.identity(HERE / "bounded_policy.py")}
