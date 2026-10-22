#!/usr/bin/env python3
"""Audit saved follow-up metrics using only the standard library over SSH."""

import argparse
import json
from pathlib import Path
import subprocess


REMOTE = r'''
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

root = Path("/opt/data/private/yuanqi_ops")
base = root / "data/experiments/p2_main"
result = {"sources": {}, "metrics": {}, "paired": {}}
names = ["warmup_21.py", "warmup_large.py", "dio_v2_minimal.py", "dio_combo.py", "dio_leaveout.py", "dio_v2_eval.py"]
paths = [root / "script" / name for name in names]
paths += [root / "logs" / name for name in ["warmup_21.log", "warmup_large.log", "dio_v2_final.log", "dio_leaveout.log", "dio_combo.log"]]
vendor = Path("/opt/data/private/hechengzhu/ZHY/script/vendor/sonic_for_a3/gear_sonic/scripts/sim2sim_a3_mujoco.py")
paths.append(vendor)
for path in paths:
    if path.is_file():
        payload = path.read_bytes()
        result["sources"][str(path)] = {"sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload), "text": payload.decode("utf-8")}
for stage in ["warmup_large", "warmup_21", "dio_v2_eval", "dio_leaveout", "dio_warmup_combo"]:
    rows = []
    for path in sorted((base / stage).glob("*/m.json")):
        payload = path.read_bytes()
        value = json.loads(payload)
        row = {"case": path.parent.name, "sha256": hashlib.sha256(payload).hexdigest(),
            "fall": value.get("fall"), "fall_time_s": value.get("fall_time_s"),
            "num_policy_steps": value.get("num_policy_steps"), "fps": value.get("fps"),
            "orientation_max_deg": value.get("anchor_pose", {}).get("orientation_max_deg")}
        ref = path.parent / "ref.csv"
        if ref.is_file():
            with ref.open() as stream:
                row["reference_rows"] = sum(1 for line in stream) - 1
        rows.append(row)
    result["metrics"][stage] = rows
    if not stage.startswith("warmup_"):
        continue
    pairs = defaultdict(dict)
    for row in rows:
        match = re.fullmatch(r"(\d+)_f(\d+)_(.+)_(std|warm)", row["case"])
        if match:
            session, frame, group, condition = match.groups()
            pairs[(session, frame, group)][condition] = row
    summary = {}
    for group in sorted(set(key[2] for key in pairs)):
        selected = [(key, pair) for key, pair in pairs.items() if key[2] == group and "std" in pair and "warm" in pair]
        standard = [pair["std"]["fall"] for key, pair in selected]
        warm = [pair["warm"]["fall"] for key, pair in selected]
        late = [bool(pair["warm"]["fall"]) and pair["warm"]["fall_time_s"] > 50/30 for key, pair in selected]
        early = [bool(pair["warm"]["fall"]) and pair["warm"]["fall_time_s"] <= 50/30 for key, pair in selected]
        n = len(selected)
        summary[group] = {"pairs": n, "sessions": len(set(key[0] for key, pair in selected)),
            "standard_falls": sum(standard), "warm_all_falls": sum(warm),
            "warm_late_first_falls": sum(late), "warm_early_first_falls": sum(early),
            "full_endpoint_improvements": sum(a and not b for a, b in zip(standard, warm)),
            "full_endpoint_regressions": sum(not a and b for a, b in zip(standard, warm)),
            "per_session": {session: {"std": sum(pair["std"]["fall"] for key, pair in selected if key[0] == session),
                "warm": sum(pair["warm"]["fall"] for key, pair in selected if key[0] == session)}
                for session in sorted(set(key[0] for key, pair in selected))},
            "std_steps": dict(Counter(str(pair["std"]["num_policy_steps"]) for key, pair in selected)),
            "warm_steps": dict(Counter(str(pair["warm"]["num_policy_steps"]) for key, pair in selected))}
    result["paired"][stage] = summary
print(json.dumps(result))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    completed = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "zhy", "python3 -"],
                               input=REMOTE, text=True, capture_output=True, check=True, timeout=60)
    result = json.loads(completed.stdout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"paired": result["paired"], "metrics_counts": {key: len(value) for key, value in result["metrics"].items()}}, indent=2))


if __name__ == "__main__":
    main()
