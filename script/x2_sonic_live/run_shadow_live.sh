#!/usr/bin/env bash
# Isolated read-only SONIC tracking capture with the live garment stream.
#
#   live garment (pico_intent PUB :15573, engage file /tmp/x2_shadow_pico_intent.engage)
#     -> [LEGACY=1 only: Y-up -> Z-up relay :15573 -> :15575]
#     -> isolated token service (planner command output disabled) :15574
#     -> diagnostic SONIC deploy --no-publish --cycle-trace, pose wire via the
#        provenance-tagging watchdog that SUBSCRIBES to the live kplanner :5556 -> :15558
#
# Nothing here publishes motor commands, touches MC, kplanner, the production
# deploy or the production token service ports (5557/5558/5573/5574/5563).
# Usage on X2 SoC1 (as agi), after build_diag.sh and with the garment running:
#   bash run_shadow_live.sh            # patched on-board adapter/token service
#   LEGACY=1 bash run_shadow_live.sh   # unpatched on-board files (relay + clock wrapper)
# Env: ENGAGE_AFTER=2.5 ENGAGE_FOR=12.5 DURATION=19.5 (seconds)
set -eo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
preview="$here/../x2_fgp_preview"
ws=${X2_SONIC_DIAG_WS:-/home/agi/x2_sonic_live/diag_ws}
sonic=/agibot/data/home/agi/gear-sonic
tools=/home/agi/projects/project_tools/x2_smpl_wholebody
py=/home/agi/.old_agi_home.bak/miniconda3/envs/teleop/bin/python
binary="$ws/install/agi_x2_deploy_onnx_ref/lib/agi_x2_deploy_onnx_ref/x2_deploy_onnx_ref"
engage=/tmp/x2_shadow_pico_intent.engage
ENGAGE_AFTER=${ENGAGE_AFTER:-2.5}
ENGAGE_FOR=${ENGAGE_FOR:-12.5}
DURATION=${DURATION:-19.5}
LEGACY=${LEGACY:-0}

[[ -x $binary && -f $ws/x2_pose_watchdog_trace.py ]] || { echo "run build_diag.sh first" >&2; exit 1; }
pgrep -f 'garment_udp_v2_reconnect_safe.py.*--pico-intent-port 15573' >/dev/null \
    || { echo "isolated garment capture (port 15573) is not running; start run_garment_fg.sh" >&2; exit 1; }
if ss -ltnH | awk '{print $4}' | grep -Eq ':(15557|15558|15566|15574|15575)$'; then
    echo "isolated diagnostic port already in use" >&2; exit 1
fi
pgrep -f pc2_kplanner_onnx.py >/dev/null || echo "WARNING: kplanner not running; pose provenance will be cold_idle" >&2

out=/home/agi/x2_sonic_live/captures/$(date +%Y%m%d_%H%M%S)
mkdir -p "$out"
echo "CAPTURE_DIR=$out"
rm -f "$engage"

pids=()
cleanup() {
    rm -f "$engage"
    for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

export LD_LIBRARY_PATH="$sonic/onnxruntime/lib:/agibot/data/home/agi/aimdk_msgs_pc2/lib:/opt/ros/humble/lib:${LD_LIBRARY_PATH:-}"
export AMENT_PREFIX_PATH="$ws/install/agi_x2_deploy_onnx_ref:/agibot/data/home/agi/aimdk_msgs_pc2:/opt/ros/humble:${AMENT_PREFIX_PATH:-}"
life=$(python3 -c "print(int($DURATION + 8))")

intent_port=15573
token_svc="$tools/x2_smpl_token_service.py"
if [[ $LEGACY == 1 ]]; then
    intent_port=15575
    timeout -k 3 "$life" "$py" -u "$here/garment_sonic_relay.py" > "$out/relay.log" 2>&1 &
    pids+=($!)
    token_svc="$out/token_svc_clockfix.py"
    "$py" - "$tools/x2_smpl_token_service.py" "$token_svc" "$tools" <<'PY'
import sys
from pathlib import Path
src = Path(sys.argv[1]).read_text()
old = "        res = core.tick()"
assert src.count(old) == 1, "token service layout changed"
src = src.replace(old, "        core._now = now  # legacy clock fix\n" + old)
src = src.replace("from token_service_core import",
                  f"import sys; sys.path.insert(0, {sys.argv[3]!r}); from token_service_core import", 1)
Path(sys.argv[2]).write_text(src)
PY
fi

PYTHONPATH="$sonic:$ws" timeout -k 3 "$life" "$sonic/venv/bin/python" -u "$ws/x2_pose_watchdog_trace.py" \
    --upstream-host 127.0.0.1 --upstream-port 5556 --upstream-topic pose \
    --downstream-port 15558 --downstream-topic pose \
    --idle-x2m2 "$sonic/data/idle_stand.x2m2" \
    --idle-stale-ms 100 --idle-mode blend --hold-last-secs 0.05 --blend-secs 0.15 \
    --x2-debug-port 15557 --trace-reference-source > "$out/watchdog.log" 2>&1 &
pids+=($!)

timeout -k 3 "$life" "$py" -u "$token_svc" \
    --tokenizer /home/agi/gear-sonic/policies/x2_smpl_tokenizer_v11release.onnx \
    --intent-port "$intent_port" --token-port 15574 \
    --debug-host 127.0.0.1 --debug-port 15557 \
    --planner-cmd-port 0 --span-dir "$out/spans" --record-dir "$out/intent_rec" > "$out/token.log" 2>&1 &
pids+=($!)

python3 - "$out/requests.jsonl" "$engage" "$ENGAGE_AFTER" "$ENGAGE_FOR" <<'PY' &
import json, sys, time
from pathlib import Path
log, eng, after, hold = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
def mark(mode):
    with log.open("a") as f:
        f.write(json.dumps({"t": time.monotonic(), "mode": mode}) + "\n")
    print(f"REQUEST {mode}", flush=True)
time.sleep(after); eng.touch(); mark("tracking")
time.sleep(hold); eng.unlink(missing_ok=True); mark("planner")
PY
pids+=($!)

status=0
timeout -k 4 "$(python3 -c "print(int($DURATION + 6))")" "$binary" \
    --no-publish --cycle-trace --log-dir "$out/deploy" \
    --model /home/agi/gear-sonic/policies/x2_sonic_s1ft16000_g1.onnx \
    --whole-body-teleop --token-model /home/agi/gear-sonic/policies/x2_sonic_s1ft16000_g1_token.onnx \
    --token-zmq-host 127.0.0.1 --token-zmq-port 15574 \
    --input-type zmq --zmq-pose-host 127.0.0.1 --zmq-pose-port 15558 --zmq-pose-topic pose \
    --zmq-debug-port 15557 --zmq-debug-topic x2_debug \
    --zmq-resume-host 127.0.0.1 --zmq-resume-port 15566 --mc-mode-poll-s 0 \
    --action-clip 20 --max-target-dev 0.3 --max-target-dev-leg 0.7 --max-target-dev-waist 0.45 \
    --max-target-dev-arm 2 --max-target-dev-head 0.40 \
    --ramp-seconds 2 --return-seconds 2 --target-lpf-hz 8 --target-lpf-hz-leg 16 \
    --target-lpf-hz-waist 0 --target-lpf-hz-arm 0 \
    --wrist-bypass ik --head-bypass ref --intra-op-threads 3 --disable-pose-ref-watchdog \
    --autostart-after 0.5 --max-duration "$DURATION" > "$out/sonic.log" 2>&1 || status=$?

echo "SONIC_STATUS=$status"
grep -E "WHOLE-BODY (ENGAGED|RELEASED)|FATAL" "$out/sonic.log" || true
tail -n 8 "$out/token.log"
mkdir -p "$out/pkg"
cp "$preview"/{audit_sonic_handoff.py,review_sonic_cycles.py,sonic_reference_provenance.py} "$out/pkg/"
touch "$out/pkg/__init__.py"
PYTHONPATH="$out" python3 -m pkg.audit_sonic_handoff "$out/deploy/cycle_trace.csv" "$out/requests.jsonl" \
    --max-gap-ms 60 --max-confirmation-ms 6000 | tee "$out/handoff_audit.json"
