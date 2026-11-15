#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$project_root"
unset PYTHONPATH PYTHONHOME

python_bin="$project_root/data/environments/fgp-preview-py310/bin/python"
test -x "$python_bin" || { echo 'Run bash script/x2_fgp_preview/setup_orin.sh first.' >&2; exit 1; }
runtime="$project_root/data/runtime/x2-fgp-viewer"
clotho_root="$project_root/data/runtime/clotho_fgp_latest/CLOTHO-mocap-main"
calibration="$project_root/script/x2_morph_human/config/clotho_fgp_yup.json"
output="$project_root/data/analysis/X2_FGP_LIVE_MORPH_$(date +%Y%m%d_%H%M%S)"
server_log="$output/fgp_server.log"
bridge_log="$output/bridge.log"
mkdir -p "$output"

cleanup() {
  if [[ -n "${server_pid:-}" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill -INT "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

flock -n "$runtime/preview.lock" "$python_bin" -u -m script.x2_fgp_preview.server \
  --mode ble --host 127.0.0.1 --port 8768 --scan-seconds 180 \
  --output "$output/preview" --three-root "$runtime/node_modules/three" \
  --lucide "$runtime/lucide.min.js" --reconstruct --clotho-root "$clotho_root" \
  >"$server_log" 2>&1 &
server_pid=$!

for _ in $(seq 1 180); do
  if curl -fsS --max-time 1 http://127.0.0.1:8768/api/state >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    cat "$server_log" >&2
    exit 1
  fi
  sleep 1
done

curl -fsS --max-time 2 http://127.0.0.1:8768/api/state >/dev/null
echo "FGP preview is live on 127.0.0.1:8768; bridge log: $bridge_log"
exec "$python_bin" -u -m script.x2_fgp_preview.live_morph_bridge \
  --preview-url http://127.0.0.1:8768 --endpoint tcp://127.0.0.1:5765 \
  --clotho-root "$clotho_root" --calibration "$calibration" \
  --output "$bridge_log"
