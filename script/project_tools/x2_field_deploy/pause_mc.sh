#!/bin/bash
# pause_mc.sh v2 (SoC0, run@) - MC freeze wrapper
# 模式:
#   hold <seconds>    会话级冻结: SIGSTOP 一次, 持续监视身份+状态, 到期限才自动 CONT。
#                     断 ssh 不放行(SIGHUP忽略); Ctrl-C = 立即 CONT; 期限硬上限 4h。
#   release           主动解冻(身份核对后才 CONT)
#   <seconds> (<=120) 旧版单次有界暂停(走原探针, 到点自动 CONT)
#   --print-identity  只打印身份(dry)
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
STATE=/tmp/mc_hold_state.json
RELEASE=/tmp/mc_pause_release

identity() {
  PID="$(pgrep -x mc_main | head -1 || true)"
  [ -n "$PID" ] || { echo "FATAL: no mc_main on this host" >&2; return 1; }
  BOOT="$(cat /proc/sys/kernel/random/boot_id)"
  ASHA="$(sha256sum "/proc/$PID/cmdline" | cut -d' ' -f1)"
  TICKS="$(sed 's/^[0-9]* (.*) //' "/proc/$PID/stat" | awk '{print $20}')"
  echo "MC identity: pid=$PID start_ticks=$TICKS argv_sha256=${ASHA:0:16}... boot=${BOOT:0:8}..."
}

case "${1:-}" in
  --print-identity) identity; echo "(dry -- nothing paused)"; exit 0;;
  release)
    [ -f "$STATE" ] || { echo "no hold state file - nothing to release"; exit 0; }
    python3 - "$STATE" <<'PYEOF'
import json,sys,os,signal,hashlib
st=json.load(open(sys.argv[1]))
pid=st["pid"]; boot=open("/proc/sys/kernel/random/boot_id").read().strip()
try:
    asha=hashlib.sha256(open(f"/proc/{pid}/cmdline","rb").read()).hexdigest()
except FileNotFoundError:
    print(f"CRITICAL: held pid {pid} is GONE (MC died or rebooted). Nothing to CONT."); sys.exit(2)
if boot!=st["boot_id"] or asha!=st["argv_sha256"]:
    print("CRITICAL: identity mismatch - refusing to CONT a different process. Check manually."); sys.exit(2)
os.kill(pid, signal.SIGCONT)
os.remove(sys.argv[1])
print(f"RELEASED pid={pid} (identity verified)")
PYEOF
    exit $?;;
  hold)
    SECS="${2:-}"; case "$SECS" in (*[!0-9]*|'') echo "usage: pause_mc.sh hold <seconds 1-14400>" >&2; exit 1;; esac
    [ "$SECS" -le 14400 ] || { echo "cap 14400s (4h)" >&2; exit 1; }
    [ -f "$STATE" ] && { echo "ALREADY HOLDING (state file exists). 先 release 或查看 $STATE" >&2; exit 1; }
    identity
    python3 - "$PID" "$BOOT" "$ASHA" "$TICKS" "$SECS" "$STATE" "$RELEASE" <<'PYEOF'
import sys,os,json,time,hashlib,signal
pid=int(sys.argv[1]); boot=sys.argv[2]; asha=sys.argv[3]; ticks=sys.argv[4]
secs=int(sys.argv[5]); state_f=sys.argv[6]; release_f=sys.argv[7]
def sha_of(p):
    try: return hashlib.sha256(open(f"/proc/{p}/cmdline","rb").read()).hexdigest()
    except FileNotFoundError: return None
def mc_state(p):
    try:
        d=open(f"/proc/{p}/stat").read()
        return d[d.rindex(")")+2:].split()[0]
    except Exception: return None
deadline=time.time()+secs
os.kill(pid, signal.SIGSTOP)
time.sleep(0.3)
st=mc_state(pid)
json.dump({"pid":pid,"boot_id":boot,"argv_sha256":asha,"start_ticks":ticks,
           "deadline_epoch":deadline,"utc_stop":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())},
          open(state_f,"w"))
print(f"FROZEN pid={pid} state={st} for {secs}s (deadline {time.strftime('%H:%M:%S',time.localtime(deadline))})")
print("  release: bash ~/walk/pause_mc.sh release   | Ctrl-C = 立即解冻 | 断 ssh 不解冻")
def on_sig(sig,frm):
    try: os.kill(pid, signal.SIGCONT)
    except Exception: pass
    if os.path.exists(state_f): os.remove(state_f)
    print("\nINTERRUPTED -> CONT sent"); sys.exit(0)
signal.signal(signal.SIGINT,on_sig)
signal.signal(signal.SIGTERM,on_sig)
signal.signal(signal.SIGHUP, signal.SIG_IGN)
rel=False
try:
    while True:
        time.sleep(5)
        cur=sha_of(pid)
        if cur is None:
            print(f"{time.strftime('%H:%M:%S')} CRITICAL: held pid {pid} vanished (MC died). NOT touching anything. Operator must inspect.")
            if os.path.exists(state_f): os.remove(state_f)
            sys.exit(2)
        if cur!=asha:
            print(f"{time.strftime('%H:%M:%S')} CRITICAL: cmdline changed under held pid. Operator must inspect.")
            os.remove(state_f); sys.exit(2)
        s=mc_state(pid)
        if s=="S":
            print(f"{time.strftime('%H:%M:%S')} WARN: MC woke unexpectedly (state S) -> re-STOP")
            os.kill(pid, signal.SIGSTOP); time.sleep(0.2)
        rel=os.path.exists(release_f)
        if rel or time.time()>=deadline: break
    os.kill(pid, signal.SIGCONT)
    if os.path.exists(state_f): os.remove(state_f)
    print(f"{time.strftime('%H:%M:%S')} RELEASED pid={pid} (reason: {'stop-file' if rel else 'deadline'})")
except BaseException as e:
    try: os.kill(pid, signal.SIGCONT)
    except Exception: pass
    if os.path.exists(state_f): os.remove(state_f)
    print(f"{time.strftime('%H:%M:%S')} EMERGENCY CONT after: {e!r}")
    raise
PYEOF
    exit $?;;
  *)
    SECS="${1:-7200}"
    case "$SECS" in (*[!0-9]*|'') echo "usage: pause_mc.sh hold <sec>|release|<sec <=120>|--print-identity" >&2; exit 1;; esac
    [ "$SECS" -le 120 ] || { echo "单次上限 120s。会话级请用: pause_mc.sh hold <seconds>" >&2; exit 1; }
    identity
    exec python3 "$DIR/x2_mc_bounded_pause_probe.py" \
      --pid "$PID" --boot-id "$BOOT" --start-ticks "$TICKS" --argv-sha256 "$ASHA" \
      --execute SUSPENDED_MC_PAUSE_2S --pause-seconds "$SECS";;
esac
