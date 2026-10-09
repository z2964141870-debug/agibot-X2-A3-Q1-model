#!/usr/bin/env bash
set -euo pipefail

project_root=/media/yu/FAFF-E9771/YUANQI
job_dir="$project_root/data/training/a3_20261009/autoresume_control_R01"
# This small guard is installed in HOME so it can wait for the data disk.
for ((attempt = 0; attempt < 120; attempt++)); do
    if mountpoint -q /media/yu/FAFF-E9771 && [[ -f "$job_dir/job.json" && -x "$project_root/data/environments/a3-sonic/bin/python" ]]; then
        cd "$project_root"
        export PYTHONPATH="$project_root:${PYTHONPATH:-}"
        export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
        exec "$project_root/data/environments/a3-sonic/bin/python" -m script.a3.autoresume "$job_dir"
    fi
    sleep 10
done
echo 'Project disk/environment not ready after 20 minutes; stopped.' >&2
exit 2
