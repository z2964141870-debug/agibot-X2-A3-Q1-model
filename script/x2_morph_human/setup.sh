#!/bin/sh
# Run on the DEVELOPMENT computer or Orin SoC1 only. No robot service is touched.
set -eu
task_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
task_env=${X2_MORPH_ENV:-"$task_root/data/environments/x2-morph-human-20261112"}
cd "$task_root"
if [ ! -x "$task_env/bin/python" ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.11 "$task_env"
  else
    python3.11 -m venv "$task_env"
  fi
fi
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$task_env/bin/python" -r script/x2_morph_human/requirements.txt
else
  "$task_env/bin/python" -m pip install -r script/x2_morph_human/requirements.txt
fi
"$task_env/bin/python" -m script.x2_morph_human.artifacts --meshes
exec "$task_env/bin/python" -m script.x2_morph_human doctor
