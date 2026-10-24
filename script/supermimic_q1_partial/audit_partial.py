#!/usr/bin/env python3
"""Independently check state retention and bounded partial-parent continuation."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import parent


HERE = Path(__file__).resolve().parent
EXTENSION = HERE.parent / "supermimic_q1_extension"
sys.path.insert(0, str(EXTENSION))
spec = importlib.util.spec_from_file_location("q1_preserved_extension_audit", EXTENSION / "audit.py")
auditor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auditor)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    auditor.validated_parent = parent.validated_parent
    report = auditor.audit_run(args.run)
    report.update(parent_kind="AUDITED_PARTIAL_FAILED_EXPERIMENT_NOT_COMPLETED_PAIR",
                  audit_source=parent.identity(__file__), preserved_auditor=parent.identity(auditor.__file__))
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, allow_nan=False), flush=True)
    return 0 if all(report["gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
