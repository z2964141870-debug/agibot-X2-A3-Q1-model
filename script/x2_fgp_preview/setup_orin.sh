#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$project_root"
unset PYTHONPATH PYTHONHOME
test "$(uname -m)" = aarch64 || { echo 'This environment is for aarch64 SoC1.' >&2; exit 1; }
python3 -c 'import sys; assert sys.version_info[:2] == (3,10), "CPython 3.10 required"'
venv_path="$project_root/data/environments/fgp-preview-py310"
test -x "$venv_path/bin/python" || python3 -m venv "$venv_path"
"$venv_path/bin/python" -m pip install --no-index --find-links "$project_root/data/runtime/x2-fgp-viewer/wheels" -r "$script_dir/requirements-orin.txt"
"$venv_path/bin/python" -m pip check
"$venv_path/bin/python" -m script.x2_fgp_preview.doctor --install-incoming --load-models
"$venv_path/bin/python" -m unittest script.x2_fgp_preview.test_preview -v
echo 'Preparation finished. BLE and robot outputs have not been started.'
