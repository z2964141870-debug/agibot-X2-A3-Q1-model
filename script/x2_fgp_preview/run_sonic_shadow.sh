#!/usr/bin/env bash
# Run the installed v16 SONIC with real feedback and a muted motor writer.
set -euo pipefail

base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
sonic=/home/agi/gear-sonic
legacy=/home/agi/projects/project_tools/x2_smpl_wholebody
python=/home/agi/.old_agi_home.bak/miniconda3/envs/teleop/bin/python
models="$sonic/policies/x2_sonic_v16ft8_45000"
binary="$sonic/ws/install/agi_x2_deploy_onnx_ref/lib/agi_x2_deploy_onnx_ref/x2_deploy_onnx_ref"

if (( $# )); then
    echo 'Usage: bash run_sonic_shadow.sh' >&2
    exit 2
fi

cd /home/agi/projects/smartwear_v2
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$base"
export LD_LIBRARY_PATH="$sonic/onnxruntime/lib:/home/agi/aimdk_msgs_pc2/lib:/opt/ros/humble/lib:${LD_LIBRARY_PATH:-}"
export AMENT_PREFIX_PATH="/home/agi/aimdk_msgs_pc2:/opt/ros/humble:${AMENT_PREFIX_PATH:-}"

"$python" - <<'PY'
from script.x2_fgp_preview.smpl_source import read_latest
from urllib.error import URLError
try:
    frame = read_latest("http://127.0.0.1:8768", 3.)
except (URLError, TimeoutError):
    raise SystemExit("FGP 网页未启动或未响应。请先执行终端一。")
if frame is None:
    raise SystemExit("FGP 没有实时 SMPL。请先检查衣服接收和 T-pose。")
PY
if pgrep -x x2_deploy_onnx_ >/dev/null || pgrep -f '[/]x2_deploy_onnx_ref( |$)|[/]x2_morph_native( |$)|[s]cript.x2_morph_backend.control_session' >/dev/null; then
    echo '已有 X2 控制后端，请先检查其状态。' >&2
    exit 1
fi
if ss -ltnH | awk '{print $4}' | grep -Eq ':(15556|15557|15558|15773|15774)$'; then
    echo '影子测试端口已被占用。' >&2
    exit 1
fi

out=$(mktemp -d "$base/data/analysis/X2_SONIC_V16_SHADOW_$(date +%Y%m%d_%H%M%S)_XXXXXX")
echo "日志目录 $out"
pids=()
cleanup() {
    if (( ${#pids[@]} )); then
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

timeout -k 3 38 "$sonic/venv/bin/python" -u "$sonic/gear_sonic_deploy/scripts/x2_pose_watchdog.py" \
    --upstream-host 127.0.0.1 --upstream-port 15556 --upstream-topic pose \
    --downstream-port 15558 --downstream-topic pose --idle-x2m2 "$sonic/data/idle_stand.x2m2" \
    --idle-stale-ms 100 --idle-mode blend --hold-last-secs 0.05 --blend-secs 0.15 \
    --x2-debug-port 15557 > "$out/stand.log" 2>&1 &
pids+=("$!")
timeout -k 3 38 "$python" -u -m script.x2_fgp_preview.sonic_shadow_intent --seconds 30 \
    > "$out/intent.log" 2>&1 &
pids+=("$!")
timeout -k 3 38 "$python" -u "$legacy/x2_smpl_token_service.py" \
    --tokenizer "${models}_smpl_tokenizer.onnx" --intent-port 15773 --token-port 15774 \
    --debug-port 15557 --planner-cmd-port 0 --span-dir '' > "$out/token.log" 2>&1 &
pids+=("$!")

echo '穿戴者先自然站立保持静止，等待 SONIC 接收真人 token。'
status=0
timeout -k 4 26 "$binary" --no-publish --input-type zmq --zmq-pose-host 127.0.0.1 \
    --zmq-pose-port 15558 --model "${models}_g1.onnx" --whole-body-teleop \
    --token-model "${models}_g1_token.onnx" --token-zmq-host 127.0.0.1 --token-zmq-port 15774 \
    --zmq-debug-port 15557 --autostart-after 5 --max-duration 12 --log-dir "$out" \
    > "$out/sonic.log" 2>&1 || status=$?
tail -n 4 "$out/token.log"
grep -E 'WHOLE-BODY ENGAGED|SAFE_IDLE|RAMP_OUT complete|FATAL' "$out/sonic.log" || true
echo "后端退出状态 $status"
if (( status != 0 )); then
    exit "$status"
fi
if ! grep -q 'WHOLE-BODY ENGAGED' "$out/sonic.log"; then
    echo '后端未切到真人 token。请检查 intent.log 和 token.log。' >&2
    exit 1
fi
