#!/bin/sh
# Project-local CPU/offline entry; do not inherit ROS Python import paths.
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
task_python=${X2_MORPH_ORIN_PYTHON:-"$task_root/data/environments/orin-py310/bin/python"}
if [ ! -x "$task_python" ]; then
  echo 'Prepare the project-local Orin Python environment first.' >&2
  exit 1
fi
cd "$task_root"
exec env -u PYTHONPATH -u PYTHONHOME "$task_python" -m script.x2_morph_human "$@"
