#!/usr/bin/env bash
set -euo pipefail

PYTHON="/opt/data/private/hechengzhu/ZHY/data/environments/a3-sonic/bin/python"
SIM2SIM="/opt/data/private/hechengzhu/ZHY/script/vendor/sonic_for_a3/gear_sonic/scripts/sim2sim_a3_mujoco.py"
MJCF="/opt/data/private/hechengzhu/ZHY/script/vendor/sonic_for_a3/gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml"
CKPT="/opt/data/private/yuanqi_ops/data/training/p2_main/m11c/checkpoints/milestone_128/model_step_000128.pt"
SEQ_DIR="/opt/data/private/hechengzhu/ZHY/data/experiments/a3_zhy_smpl_hull_20261013_E23/sequences"

OUT_DIR="/opt/data/private/yuanqi_ops/data/experiments/p2_main/m11c_eval"
mkdir -p "$OUT_DIR"

MOTIONS=(
  "G_0113_zj_train__183926_zj_walk"
  "G_0113_zj_train__192333_zj_warmUp"
  "G_20251226_fjw__fjw_140008_walk"
  "G_20251226_zls__zls_155013_walk"
  "G_20251229_yy__yy_125106_walk"
  "G_20251229_zzx__zzx_111505_walk"
)

echo "Starting evaluation of P2-M11C on 6 motions..."

for m in "${MOTIONS[@]}"; do
  CSV="$SEQ_DIR/$m/reference.csv"
  METRICS="$OUT_DIR/${m}.metrics.json"
  echo "--- Evaluating $m ---"
  $PYTHON "$SIM2SIM"     --checkpoint "$CKPT"     --motion "$CSV"     --encoder-mode a3_fast     --csv-source-fps 30     --csv-frame-stride 1     --mjcf "$MJCF"     --batch-once     --action-delay-ms 0     --metrics-out "$METRICS" > /dev/null 2>&1 || true
  echo "Done $m"
done

echo "M11 Evaluation complete."

