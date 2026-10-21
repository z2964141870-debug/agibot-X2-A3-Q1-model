#!/usr/bin/env bash
# post_unpack_pipeline.sh -- autonomous smoke + triage after unpack_DONE.
# Gates on unpack_DONE (poll 120s), then:
#   1) pick 3 npz from 3 distinct dataset dirs
#   2) smplx2smpl24.py convert (single-file runs, --verify) -> data/smoke_out + smoke_report_*.json
#   3) triage.py --scan-frames 50 -> triage_report.json
#   4) PIPELINE_DONE marker + pipeline_summary.json
# Red lines: CPU only, no training dirs, stdin-null guarded.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="/opt/data/private/hechengzhu/ZHY/data/env/bin/python"
RAW="$HERE/data/amass_raw"
LOG="$HERE/pipeline.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "waiting for unpack_DONE"
until [ -f "$HERE/unpack_DONE" ]; do sleep 120; done
log "unpack_DONE seen: $(cat "$HERE/unpack_DONE")"

N=$(find "$RAW" -name '*.npz' | wc -l)
log "npz count: $N"
"$PY" - "$RAW" "$HERE" <<'PYEOF' </dev/null
import json, os, subprocess, sys, glob
raw, here = sys.argv[1], sys.argv[2]
py = "/opt/data/private/hechengzhu/ZHY/data/env/bin/python"
files = sorted(glob.glob(os.path.join(raw, "**", "*.npz"), recursive=True))
# 3 files from 3 distinct dataset dirs (component right after raw)
seen, picks = set(), []
for f in files:
    ds = os.path.relpath(f, raw).split(os.sep)[0]
    if ds not in seen:
        seen.add(ds); picks.append(f)
    if len(picks) == 3:
        break
while len(picks) < 3 and len(files) > len(picks):
    picks.append(files[len(picks)])
print("picks:", picks, flush=True)
smoke_dir = os.path.join(here, "data", "smoke_out")
reports = []
for i, f in enumerate(picks):
    rep = os.path.join(here, "smoke_report_%d.json" % i)
    r = subprocess.run([py, os.path.join(here, "smplx2smpl24.py"), "convert",
                        "--input", f, "--out-dir", smoke_dir, "--verify", "1",
                        "--report", rep], capture_output=True, text=True, timeout=600)
    print("smoke[%d] rc=%d %s" % (i, r.returncode, (r.stdout or "").strip()[-300:]), flush=True)
    if os.path.exists(rep):
        reports.append(json.load(open(rep)))
json.dump({"picks": picks, "smoke_reports": reports},
          open(os.path.join(here, "smoke_summary.json"), "w"), indent=1, ensure_ascii=False)
print("smoke summary written", flush=True)
PYEOF
RC1=$?
log "smoke rc=$RC1"

log "triage (scan-frames 50)"
timeout 3000 "$PY" "$HERE/triage.py" --root "$RAW" --out "$HERE/triage_report.json" --scan-frames 50 </dev/null >>"$LOG" 2>&1
RC2=$?
log "triage rc=$RC2"

"$PY" - "$HERE" <<'PYEOF2' </dev/null
import json, os, sys
here = sys.argv[1]
s = {"smoke_rc": None, "triage_rc": None}
try:
    s["smoke"] = json.load(open(os.path.join(here, "smoke_summary.json")))
except Exception as e:
    s["smoke_error"] = str(e)
try:
    t = json.load(open(os.path.join(here, "triage_report.json")))
    s["triage"] = {"n_files": t["n_files"], "buckets": {b: v["count"] for b, v in t["buckets"].items()},
                   "hours_at_30fps": t.get("frame_scan", {}).get("hours_at_30fps"),
                   "basis": t["decision_basis"]}
except Exception as e:
    s["triage_error"] = str(e)
json.dump(s, open(os.path.join(here, "pipeline_summary.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(s, ensure_ascii=False)[:800])
PYEOF2
echo "$(date '+%F %T') PIPELINE_DONE smoke_rc=$RC1 triage_rc=$RC2" > "$HERE/PIPELINE_DONE"
log "PIPELINE_DONE smoke_rc=$RC1 triage_rc=$RC2"
