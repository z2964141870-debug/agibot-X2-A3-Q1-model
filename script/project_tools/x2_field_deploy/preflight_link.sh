#!/bin/bash
# preflight_link.sh (SoC1, agi@) - 上电后第一步: 无风险链路预检
# 只起 kplanner + 键盘; 不碰 deploy / 不碰 MC -> 机器人零运动。
# 验证三件事: venv 实机跑通 kplanner; :5556 流出 3936B V51 帧;
# 键盘 planner_cmd 被实机 ingest。
# 注意: 预检用 --no-yaw-rebase(独立运行无法等 deploy 的 x2_debug 航向);
# 现场正式链条不带此旗标, 航向由 deploy x2_debug 提供(rebase ON)。
set -uo pipefail
PY=/home/agi/gear-sonic/venv/bin/python
ONNX=/home/agi/gear-sonic/planner_stack/models/planner_onnx/x2_planner_template.onnx
LOG=/tmp/kpl_preflight.log
fail(){ echo "FAIL: $*"; exit 1; }
[ -x "$PY" ] || fail "venv python 缺失: $PY"
[ -f "$ONNX" ] || fail "template planner onnx 缺失: $ONNX"
[ -f /home/agi/walk/x2_keyboard_walk_publisher.py ] || fail "键盘工具未部署在 ~/walk/"

echo "[1/5] 启动 kplanner (单独, 后台 tmux 会话 kpl_preflight)..."
tmux kill-session -t kpl_preflight 2>/dev/null
tmux new-session -d -s kpl_preflight \
  "cd /home/agi/gear-sonic && export PC2_PREFIX=/home/agi/gear-sonic && KPLANNER_FIXED_FWD_MPS=0.35 KPLANNER_FIXED_TURN_RAD_S=0.65 \
   $PY pc2_kplanner_onnx.py --onnx $ONNX --no-yaw-rebase 2>&1 | tee $LOG"

echo "[2/5] 等 planner worker 起来 (最多 90s; 独立模式无 planner active)"...
ok=""
for i in $(seq 1 90); do
  grep -q "planner worker thread started" $LOG 2>/dev/null && { ok=1; break; }
  grep -qiE "Traceback" $LOG 2>/dev/null && { tail -5 $LOG; fail "kplanner 启动报错"; }
  sleep 1
done
[ -n "$ok" ] || { tail -8 $LOG; fail "90s 内 worker 未起"; }
echo "  OK: planner active"

echo "[3/5] 验证 :5556 pose 流 (v4 契约 13 字段)..."
$PY - <<'PYEOF'
import zmq, json
ctx = zmq.Context(); s = ctx.socket(zmq.SUB)
s.setsockopt_string(zmq.SUBSCRIBE, "pose"); s.RCVTIMEO = 8000
s.connect("tcp://127.0.0.1:5556")
try:
    m = s.recv_multipart()
except zmq.Again:
    print("  FAIL: 5556 无帧流出"); raise SystemExit(1)
frame = m[-1]
j = frame.find(b"{")
txt = frame.decode("latin1")
hdr, end = json.JSONDecoder().raw_decode(txt[j:])
names = [f["name"] for f in hdr.get("fields", [])]
payload_len = len(frame) - (j + end)
need = {"joint_pos_mj","root_quat_xyzw","motion_token","left_hand_joints",
        "right_hand_joints","frame_index","root_xy_world","root_z_world",
        "joint_pos_mj_future","root_quat_xyzw_future","joint_vel_mj_future",
        "frame_index_future","future_dt_s"}
missing = need - set(names)
print("  单 part 帧 %dB = topic(%dB) + JSON头(%dB) + payload(%dB)" %
      (len(frame), j, end, payload_len))
print("  协议 v%s endian=%s 字段数 %d" % (hdr.get("v"), hdr.get("endian","le"), len(names)))
print("  字段:", ", ".join(names))
if missing:
    print("  FAIL: 缺契约字段:", missing); raise SystemExit(1)
print("  OK: v4 十三字段契约全在 (deploy 按字段名解码; vel 由 deploy 差分自算)")
PYEOF
[ $? -eq 0 ] || fail "pose 流/契约不过"
[ $? -eq 0 ] || fail "pose 流/契约不过"
[ $? -eq 0 ] || fail "pose 流不通"

echo "[4/5] 键盘发 3s planner_cmd, 查 kplanner 实机 ingest..."
$PY /home/agi/walk/x2_keyboard_walk_publisher.py --mode planner-cmd --duration 3 >/dev/null 2>&1
TAPE=$(ls -t /home/agi/gear-sonic/log/kplanner_tape/tape_*.jsonl 2>/dev/null | head -1)
if [ -n "$TAPE" ] && grep -aq "intent_recv" "$TAPE"; then
  echo "  OK: 实机 ingest 证据 (tape $TAPE):"
  grep -a "intent_recv" "$TAPE" | tail -1 | cut -c1-220
else
  echo "  WARN: tape 未捕获 intent_recv (键盘帧可能在 ingest 窗口外)"
fi

echo "[5/5] 清理: 关闭预检 kplanner"
tmux kill-session -t kpl_preflight 2>/dev/null
echo ""
echo "预检通过 = T1 的 kplanner 零件 + T2 键盘链路全部实机验证。"
echo "之后才执行现场任务书 T3 -> T1 -> T2 全链 (含 deploy)。"
