#!/bin/sh
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
task_python=${X2_MORPH_PYTHON:-"$task_root/data/environments/x2-morph-human-20261112/bin/python"}
if [ ! -x "$task_python" ]; then
  echo 'Run script/x2_morph_human/setup.sh first, or set X2_MORPH_PYTHON.' >&2
  exit 1
fi
cd "$task_root"
exec "$task_python" -m script.x2_morph_human "$@"
