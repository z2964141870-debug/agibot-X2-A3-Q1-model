#!/usr/bin/env bash
# 终端 2：启动 Policy 链路（按需拉起 watchdog + kplanner + token service，前台点火 deploy_x2.sh）。
# 默认 --root-heading-band-deg 0（关闭 25° 航向锁，允许穿戴者迈步/转向；务必有人扶护）。
set -euo pipefail

PREFIX="/agibot/data/home/agi/gear-sonic"
TOOLS="/home/agi/projects/project_tools/x2_smpl_wholebody"
TELEOP_PY="/home/agi/.old_agi_home.bak/miniconda3/envs/teleop/bin/python"
GS_PY="$PREFIX/venv/bin/python"
PS="$PREFIX/planner_stack"

ENGAGE_FILE="${PICO_ENGAGE_FILE:-/home/agi/pico_intent.engage}"
INTENT_PORT="${PICO_INTENT_PORT:-5573}"
TOKEN_PORT="${PICO_TOKEN_PORT:-5574}"
BAND_DEG="${ROOT_HEADING_BAND_DEG:-0}"
MAX_DURATION="${X2_MAX_DURATION:-300}"
TUNING="${X2_RITUAL_TUNING:-$PREFIX/gear_sonic_deploy/configs/real_deploy_tuning/trained_gains_s0.yaml}"

if pgrep -f 'x2_deploy_onnx_re[f]' >/dev/null 2>&1; then
    echo "[term2] ERROR: 已有 x2_deploy_onnx_ref 在运行，请先在原终端停止后再启动。" >&2
    exit 1
fi

tok_pid=""
on_exit() {
    rm -f "$ENGAGE_FILE"
    if [[ -n "$tok_pid" ]] && kill -0 "$tok_pid" 2>/dev/null; then
        kill "$tok_pid" 2>/dev/null || true
        wait "$tok_pid" 2>/dev/null || true
    fi
    echo "[term2] 已撤 engage 并关闭 token service。"
}
trap on_exit EXIT

# 1. 确保先处于撤权状态，再清理并启动 token service
rm -f "$ENGAGE_FILE"
pkill -f 'x2_smpl_token_servic[e]' 2>/dev/null || true
pkill -f 'pc2_pico_token_servic[e]' 2>/dev/null || true
sleep 1

# 2. 按需拉起上游 watchdog (:5558) 与 pc2_kplanner (:5556 / :5563)
if ! ss -ltn | grep -q ':5558 '; then
    tmux kill-session -t x2_pose_watchdog 2>/dev/null || true
    echo "[term2] 启动 x2_pose_watchdog tmux 会话..."
    tmux new-session -d -s x2_pose_watchdog "bash $PREFIX/start_x2_pose_watchdog_local.sh"
fi

if ! ss -ltn | grep -q ':5556 '; then
    tmux kill-session -t pc2_kplanner 2>/dev/null || true
    echo "[term2] 启动 pc2_kplanner tmux 会话..."
    tmux new-session -d -s pc2_kplanner \
        "export PYTHONPATH=$PS && . $PREFIX/robot_env.env && stdbuf -oL -eL $GS_PY $PREFIX/pc2_kplanner_onnx.py --onnx $PS/models/planner_onnx/x2_planner_template.onnx --planner-mode slow_walk --cmd-bind --command-watchdog-s 0.3 --replan-threshold-frames \${KPLANNER_REPLAN_THRESHOLD_FRAMES:?} --warmup-qpos $PS/models/kplanner_idle_anchor_g1teleop_v3.pkl --dances-dir $PS/models/dances_x2m2 --ort-gpu --playing-yaw-resync-dps 10 2>&1 | tee -a $PREFIX/log/pc2_kplanner.log"
fi

echo "[term2] 等待 kplanner (:5556) 与 watchdog (:5558) 就绪..."
ready_up=0
for _ in $(seq 1 40); do
    if ss -ltn | grep -q ':5556 ' && ss -ltn | grep -q ':5558 '; then
        ready_up=1
        break
    fi
    sleep 0.5
done
[[ "$ready_up" == "1" ]] || { echo "[term2] ERROR: :5556/:5558 未就绪，请检查 tmux attach -t pc2_kplanner" >&2; exit 1; }

# 3. 启动 token service（--root-heading-band-deg 默认 0 允许迈步转向）
echo "[term2] 启动 x2_smpl_token_service (--root-heading-band-deg $BAND_DEG)..."
(
    cd "$TOOLS"
    exec "$TELEOP_PY" -u x2_smpl_token_service.py \
        --tokenizer "$PREFIX/policies/x2_smpl_tokenizer_v11release.onnx" \
        --intent-port "$INTENT_PORT" --token-port "$TOKEN_PORT" \
        --debug-host 127.0.0.1 --debug-port 5557 \
        --planner-cmd-port 5563 \
        --root-heading-band-deg "$BAND_DEG" \
        --span-dir ''
) > /home/agi/pico_token_service.log 2>&1 &
tok_pid=$!

tok_up=0
for _ in $(seq 1 20); do
    if ! kill -0 "$tok_pid" 2>/dev/null; then
        echo "[term2] ERROR: x2_smpl_token_service 启动失败：" >&2
        cat /home/agi/pico_token_service.log >&2 || true
        exit 1
    fi
    if ss -ltn | grep -q ":${TOKEN_PORT} "; then
        tok_up=1
        break
    fi
    sleep 0.5
done
[[ "$tok_up" == "1" ]] || { echo "[term2] ERROR: token :$TOKEN_PORT 未监听" >&2; exit 1; }

echo "[term2] 全链路底层就绪。准备在前台点火 SONIC Whole-Body Policy (时长上限 ${MAX_DURATION}s)..."
echo "[term2] 注意：若开启迈步 (band=$BAND_DEG)，务必确认现场有人扶护、手握急停；按 Ctrl-C 可软退回 MC。"

cd "$PREFIX/gear_sonic_deploy"
export PC2_HOST=10.0.1.41
X2_WB_TELEOP=1 \
X2_TOKEN_MODEL="$PREFIX/policies/x2_sonic_s1ft16000_g1_token.onnx" \
X2_TOKEN_ZMQ_HOST=127.0.0.1 \
X2_TOKEN_ZMQ_PORT="$TOKEN_PORT" \
X2_MAX_DURATION="$MAX_DURATION" \
./deploy_x2.sh onbot --no-docker --vla \
    --vla-zmq-host localhost --vla-zmq-port 5558 --vla-zmq-topic pose \
    --vla-debug-port 5557 --vla-debug-topic x2_debug \
    --wrist-bypass ik --head-bypass ref \
    --model "$PREFIX/policies/x2_sonic_s1ft16000_g1.onnx" \
    --onbot-prefix "$PREFIX" \
    --onbot-ws "$PREFIX/ws" \
    --onbot-venv "$PREFIX/venv" \
    --onbot-onnxruntime "$PREFIX/onnxruntime" \
    --onbot-aimdk-prefix /agibot/data/home/agi/aimdk_msgs_pc2 \
    --tuning-config "$TUNING" \
    --max-target-dev-head 0.40 \
    --intra-op-threads 3 \
    --deploy-extra-arg --flight-recorder-seconds --deploy-extra-arg 60 \
    --deploy-extra-arg --disable-pose-ref-watchdog
