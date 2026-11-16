#!/usr/bin/env bash
# 终端 3：切换机器人运控（按 Enter 在 kplanner 站立 <-> 自己的全身 SONIC policy 间切换），并实时刷状态。
set -uo pipefail

TOOLS="/home/agi/projects/project_tools/x2_smpl_wholebody"
PY="/home/agi/.old_agi_home.bak/miniconda3/envs/teleop/bin/python"
ENGAGE_FILE="${PICO_ENGAGE_FILE:-/home/agi/pico_intent.engage}"
INTENT_PORT="${PICO_INTENT_PORT:-5573}"
TOKEN_PORT="${PICO_TOKEN_PORT:-5574}"

pkill -f 'pipeline_monito[r]' 2>/dev/null || true
rm -f "$ENGAGE_FILE"

mon_pid=""
cleanup() {
    rm -f "$ENGAGE_FILE"
    if [[ -n "$mon_pid" ]] && kill -0 "$mon_pid" 2>/dev/null; then
        kill "$mon_pid" 2>/dev/null || true
        wait "$mon_pid" 2>/dev/null || true
    fi
    echo
    echo "[term3] 已安全撤出全身 policy（回 kplanner 站立）并关闭监视器。"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "=================================================================="
echo " 终端 3：全身 SONIC 运控切换台 + 1 Hz 实时状态"
echo " 当前状态: [kplanner 站立 / 未接管]"
echo " 操作方法:"
echo "   直接按 [Enter]   -> 在 [kplanner 站立] 与 [全身 SONIC] 间切换"
echo "   输入 1 + Enter   -> 接入自己的全身 policy (touch engage)"
echo "   输入 0 + Enter   -> 切回 kplanner 站立 (rm engage)"
echo "   按 Ctrl-C        -> 立刻撤出全身 policy 并退出"
echo "=================================================================="

(
    cd "$TOOLS"
    exec "$PY" -u pipeline_monitor.py \
        --intent-port "$INTENT_PORT" --token-port "$TOKEN_PORT" \
        --debug-host 127.0.0.1 --debug-port 5557 \
        --log /home/agi/monitor.csv
) &
mon_pid=$!

while true; do
    if ! IFS= read -r cmd < /dev/tty; then
        break
    fi
    case "$cmd" in
        1|on|ON)
            touch "$ENGAGE_FILE"
            echo ">>> [已接入] SONIC 全身 policy 接管中（按 Enter 或 Ctrl-C 切回站立）"
            ;;
        0|off|OFF|q|Q)
            rm -f "$ENGAGE_FILE"
            echo ">>> [已撤出] 切回 kplanner 站立（按 Enter 可再次接入）"
            ;;
        "")
            if [[ -f "$ENGAGE_FILE" ]]; then
                rm -f "$ENGAGE_FILE"
                echo ">>> [已撤出] 切回 kplanner 站立（按 Enter 可再次接入）"
            else
                touch "$ENGAGE_FILE"
                echo ">>> [已接入] SONIC 全身 policy 接管中（按 Enter 或 Ctrl-C 切回站立）"
            fi
            ;;
        *)
            echo ">>> 未识别指令 '$cmd'：直接按 Enter 切换，或输入 1=接入 / 0=切回"
            ;;
    esac
done
