#!/usr/bin/env bash
# The existing SONIC garment receiver, with interactive T-pose on this terminal.
set -euo pipefail

if [[ ${1:-} == --help ]]; then
    printf '%s\n' 'SONIC 衣服前台采集。收到上下衣后按 Enter，看到倒计时后保持 T-pose。'
    exit 0
fi
if (( $# )) || [[ ! -t 0 || ! -t 1 ]]; then
    printf '%s\n' '请直接在 X2 的 agi 终端前台运行。' >&2
    exit 2
fi

wear=/home/agi/projects/smartwear_v2
python=/home/agi/.old_agi_home.bak/miniconda3/envs/teleop/bin/python
cd "$wear"
if pgrep -f '[/]garment_udp_v2_reconnect_safe.py|[ ]garment_udp_v2_reconnect_safe.py' >/dev/null; then
    printf '%s\n' '板上已有衣服采集进程，请先退出旧采集。' >&2
    exit 1
fi
printf '%s\n' '等待上下衣数据。出现校准提示后请按 Enter，终端会显示倒计时。'
exec env -u PYTHONHOME "$python" -B -u garment_udp_v2_reconnect_safe.py \
    --input ble --addr-up F5:C6:03:E6:4B:30 --addr-down DA:1B:78:C1:56:1F \
    --lfp-backend local-cpu --stream-mode none --no-viewer --pico-intent-port 5573 \
    --pico-intent-engage-file /home/agi/pico_intent.engage
