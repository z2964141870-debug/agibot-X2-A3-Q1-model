#!/usr/bin/env python3
"""Read saved onset evidence over SSH; never load models or run rollouts."""

import argparse
import json
from pathlib import Path
import subprocess


REMOTE_AUDIT = r'''
import csv
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
result = {"root": str(root), "sources": {}}
for name in ["script/onset_analysis.py", "reports/onset_all_evaluations.csv",
             "reports/ONSET_FINAL_RESULTS.txt", "reports/SESSION_CLASSIFICATION.txt",
             "reports/RELATED_WORK_ONSET.md", "reports/ONSET_RESEARCH_SUMMARY.md"]:
    path = root / name
    if path.is_file():
        payload = path.read_bytes()
        result["sources"][name] = {"bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "text": payload.decode("utf-8")}
rows = list(csv.DictReader(io.StringIO(result["sources"]["reports/onset_all_evaluations.csv"]["text"])))
keys = [(r["session"], r["start_frame"], r["policy"]) for r in rows]
result["csv"] = {"rows": len(rows), "columns": len(rows[0]),
    "unique_keys": len(set(keys)), "policies": dict(Counter(r["policy"] for r in rows)),
    "sessions": len(set(r["session"] for r in rows))}
stages = ["onset_large", "onset_xfine", "onset_policies"]
groups = defaultdict(list)
counts = {}
for stage in stages:
    files = sorted((root / "data/experiments/p2_main" / stage).glob("*.metrics.json"))
    counts[stage] = len(files)
    for path in files:
        payload = path.read_bytes()
        value = json.loads(payload)
        groups[path.name].append({"stage": stage, "sha256": hashlib.sha256(payload).hexdigest(),
            "fall": value.get("fall"), "fall_time_s": value.get("fall_time_s"),
            "anchor_pose": value.get("anchor_pose"), "metrics_keys": sorted(value)})
result["metrics"] = {"counts": counts, "unique_filenames": len(groups),
    "duplicated_filenames": sum(len(v) > 1 for v in groups.values()),
    "byte_identical_duplicate_groups": sum(len(v) > 1 and len(set(x["sha256"] for x in v)) == 1 for v in groups.values()),
    "duplicate_example": next(((k, v) for k, v in groups.items() if len(v) > 1), None)}
cells = defaultdict(list)
for row in rows:
    cells[(row["policy"], int(row["start_frame"]))].append(row)
result["cells"] = {}
for (policy, start), cell in sorted(cells.items()):
    n = len(cell)
    k = sum(r["fall"].lower() == "true" for r in cell)
    p = k / n
    z = 1.959963984540054
    denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    result["cells"][policy + ":" + str(start)] = {"falls": k, "n": n,
        "wilson95_if_independent": [center-half, center+half]}
by_session = defaultdict(dict)
for row in rows:
    if row["policy"] == "035":
        by_session[row["session"]][int(row["start_frame"])] = row["fall"].lower() == "true"
result["paired_f0_f50"] = dict(Counter(str((v[0], v[50])) for v in by_session.values()))
result["sessions_with_no_falls_all_starts"] = [s for s, v in by_session.items() if not any(v.values())]
ref = root / "data/gen_audit/onset_large/session_20260917_174102_f200/reference.csv"
with ref.open() as stream:
    reader = csv.reader(stream)
    result["reference_schema_example"] = {"path": str(ref), "header": next(reader), "first_row": next(reader)}
result["evaluation_example"] = groups.get("session_20260917_174102_f0__035.metrics.json")
print(json.dumps(result, ensure_ascii=False))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="zhy")
    parser.add_argument("--root", default="/opt/data/private/yuanqi_ops")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.root.startswith("/") or any(c not in "/_-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in args.root):
        parser.error("--root must be an absolute ASCII path without shell metacharacters")
    process = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                              args.host, "python3 - " + args.root], input=REMOTE_AUDIT,
                             text=True, capture_output=True, check=True, timeout=60)
    result = json.loads(process.stdout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "sources"}, indent=2))


if __name__ == "__main__":
    main()
