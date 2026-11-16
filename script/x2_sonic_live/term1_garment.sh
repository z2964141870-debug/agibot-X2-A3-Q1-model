#!/usr/bin/env bash
# 终端 1：控制衣服（前台运行；按 Enter 触发 3-2-1 T-Pose 标定；Ctrl-C 退出并释放蓝牙）。
set -uo pipefail

UP="F5:C6:03:E6:4B:30"
DOWN="DA:1B:78:C1:56:1F"
ENGAGE_FILE="${PICO_ENGAGE_FILE:-/home/agi/pico_intent.engage}"
INTENT_PORT="${PICO_INTENT_PORT:-5573}"
PY="/home/agi/.old_agi_home.bak/miniconda3/envs/teleop/bin/python"

if [[ ! -t 0 || ! -t 1 ]]; then
    echo "请在交互式终端运行（需要按 Enter 开始 T-Pose 标定）。" >&2
    exit 2
fi

release_ble() {
    bluetoothctl disconnect "$UP" >/dev/null 2>&1 || true
    bluetoothctl disconnect "$DOWN" >/dev/null 2>&1 || true
}

cleanup_stale() {
    rm -f "$ENGAGE_FILE"
    if pgrep -f '[g]arment_udp_v2_reconnect_safe.py' >/dev/null 2>&1; then
        echo "[term1] 清理残留衣服进程..."
        pkill -INT -f '[g]arment_udp_v2_reconnect_safe.py' || true
        sleep 2
        pkill -KILL -f '[g]arment_udp_v2_reconnect_safe.py' 2>/dev/null || true
    fi
    release_ble
}

child=""
on_exit() {
    rm -f "$ENGAGE_FILE"
    if [[ -n "$child" ]] && kill -0 "$child" 2>/dev/null; then
        kill -INT "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    release_ble
    echo "[term1] 已退出并释放蓝牙 ($UP, $DOWN)"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cleanup_stale
echo "[term1] 启动衣服前台采集 (intent :$INTENT_PORT)..."
echo "[term1] 两件衣服连上后，摆好 T-Pose 按 Enter 开始标定；按 Ctrl-C 安全退出。"
cd /home/agi/projects/smartwear_v2
env -u PYTHONHOME "$PY" -B -u garment_udp_v2_reconnect_safe.py \
    --input ble --addr-up "$UP" --addr-down "$DOWN" \
    --lfp-backend local-cpu --stream-mode none --no-viewer \
    --pico-intent-port "$INTENT_PORT" \
    --pico-intent-engage-file "$ENGAGE_FILE" < /dev/tty &
child=$!
wait "$child"
