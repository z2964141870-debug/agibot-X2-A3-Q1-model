#!/usr/bin/env bash
# Foreground garment capture for the isolated SONIC shadow test.
# Same receiver and arguments as the production foreground entry, but the
# pico_intent port and engage file are isolated (15573 / /tmp/x2_shadow_...).
# Releases both BLE links before start and on any exit, because the unpatched
# receiver aborts on Ctrl-C and leaves BlueZ links connected.
set -uo pipefail
UP=F5:C6:03:E6:4B:30
DOWN=DA:1B:78:C1:56:1F

if [[ ! -t 0 || ! -t 1 ]]; then
    echo "请在 X2 的 agi 终端前台运行（需要按 Enter 标定）。" >&2
    exit 2
fi
if pgrep -f '[g]arment_udp_v2_reconnect_safe.py' >/dev/null; then
    echo "板上已有衣服采集进程，请先退出旧采集。" >&2
    exit 1
fi

release_ble() {
    bluetoothctl disconnect "$UP" >/dev/null 2>&1 || true
    bluetoothctl disconnect "$DOWN" >/dev/null 2>&1 || true
}
child=""
cleanup() {
    if [[ -n $child ]] && kill -0 "$child" 2>/dev/null; then
        kill -INT "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    release_ble
    echo "[run_garment_fg] BLE released ($UP, $DOWN)"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

release_ble
cd /home/agi/projects/smartwear_v2
env -u PYTHONHOME /home/agi/.old_agi_home.bak/miniconda3/envs/teleop/bin/python -B -u \
    garment_udp_v2_reconnect_safe.py \
    --input ble --addr-up "$UP" --addr-down "$DOWN" \
    --lfp-backend local-cpu --stream-mode none --no-viewer \
    --pico-intent-port 15573 --pico-intent-engage-file /tmp/x2_shadow_pico_intent.engage < /dev/tty &
child=$!
wait "$child"
