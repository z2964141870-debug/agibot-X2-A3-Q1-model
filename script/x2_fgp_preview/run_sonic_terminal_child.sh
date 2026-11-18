#!/usr/bin/env bash
# Executable compatible with terminal.py's --bridge-python child contract.
set -euo pipefail
if [[ $# -lt 4 || "$1 $2 $3 $4" != "-B -m script.x2_fgp_preview.terminal --forward-fd" ]]; then
    echo "Pass this executable as terminal.py's --bridge-python for the SONIC input probe." >&2
    exit 2
fi
shift 3
base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd /home/agi/projects/smartwear_v2
export PYTHONPATH="$base" CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
exec /home/agi/.old_agi_home.bak/miniconda3/envs/teleop/bin/python -B -u \
    -m script.x2_fgp_preview.sonic_terminal_pipe "$@"
