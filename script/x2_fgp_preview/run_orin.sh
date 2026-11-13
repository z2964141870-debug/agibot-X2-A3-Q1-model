#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$project_root"
unset PYTHONPATH PYTHONHOME
python_bin="$project_root/data/environments/fgp-preview-py310/bin/python"
test -x "$python_bin" || { echo 'Run bash script/x2_fgp_preview/setup_orin.sh first.' >&2; exit 1; }
runtime="$project_root/data/runtime/x2-fgp-viewer"
mode="${1:-smpl}"
extra=()
case "$mode" in
  smpl)
    extra=(--reconstruct --clotho-root "$project_root/data/runtime/clotho_fgp_latest/CLOTHO-mocap-main")
    echo 'FGP: EgoLocate default U100h denoiser + GlobalPose Full_GR_OV_10 (verified weights).'
    echo 'Current CPU backend is diagnostic only: recorded Orin median 781ms/output; live 30Hz is not validated.'
    ;;
  transport) extra=(--template "$runtime/smpl_template.json");;
  *) echo 'Usage: bash script/x2_fgp_preview/run_orin.sh [smpl|transport]' >&2; exit 1;;
esac
output="$project_root/data/analysis/X2_FGP_LIVE_$(date +%Y%m%d_%H%M%S)"
echo "Output: $output"
echo 'Browser on the robot LAN: http://10.0.1.41:8768'
echo 'BLE scan may take up to 180 seconds; each connection allows up to 60 seconds, two attempts.'
echo 'Robot feedback and motor output remain disabled. Ctrl-C stops this preview.'
exec flock -n "$runtime/preview.lock" "$python_bin" -u -m script.x2_fgp_preview.server --mode ble \
  --host 0.0.0.0 --port 8768 --scan-seconds 180 --output "$output" \
  --three-root "$runtime/node_modules/three" --lucide "$runtime/lucide.min.js" "${extra[@]}"
