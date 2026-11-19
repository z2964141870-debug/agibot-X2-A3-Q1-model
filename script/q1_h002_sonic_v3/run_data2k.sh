#!/usr/bin/env bash
set -uo pipefail
cd /home/yu/projects/SuperMimicHP
export PYTHONPATH="src:/home/yu/models/cache/GMR"
LOG="/home/yu/runs/logs/smhp_data2k.log"
RC_FILE="/tmp/smhp_data2k.rc"
rm -f "$RC_FILE"
{
  echo "=== START DATA2K $(date -Iseconds) ==="
  /home/yu/envs/smhp_retarget/bin/python -u runs/h002_sonic_v3/build_scaled_dataset_2000.py
  rc=$?
  echo "=== DATA2K_RC=$rc $(date -Iseconds) ==="
  echo "$rc" > "$RC_FILE"
  exit "$rc"
} > "$LOG" 2>&1
