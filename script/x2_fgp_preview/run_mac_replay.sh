#!/usr/bin/env bash
set -euo pipefail
project_root="/Users/yu/Documents/ChatGPT/元启"
cd "$project_root"
exec "$project_root/data/environments/clotho-fgp-20261112/bin/python" -u -m script.x2_fgp_preview.server \
  --mode replay --clotho-root "$project_root/data/runtime/clotho_fgp_latest/CLOTHO-mocap-main" \
  --recording "$project_root/data/analysis/X2_LATEST_FGP_PRETEST_20261112/fgp_exports/manual001_active4800_globalpose.npz" \
  --three-root "$project_root/data/runtime/x2-fgp-viewer/node_modules/three" \
  --lucide "$project_root/data/runtime/x2-fgp-viewer/lucide.min.js" \
  --output "$project_root/data/analysis/X2_FGP_REPLAY_$(date +%Y%m%d_%H%M%S)" --port "${1:-8768}"
