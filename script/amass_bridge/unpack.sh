#!/usr/bin/env bash
# unpack.sh -- AMASS SMPL-X G split-zip gated unpacker (A3 data bridge).
#
#   unpack.sh run       gate1: ALL_DOWNLOADS_DONE in dl_progress.log (poll, default 600s)
#                       gate2: volume set + sizes stable (double sample, 90s apart)
#                       -> extract via split_zip_extract.py (THE ONLY extractor)
#                       -> data/amass_raw/, manifest + unpack_DONE marker
#   unpack.sh validate  small-scale round-trip proof of exactly that path (in /tmp)
#   unpack.sh status    state printout
#
# HARD RULES (lead directive 2026-10-20):
#   * System zip repair modes, merge mode, and the Info-ZIP extractor are all BANNED
#     merge output corrupts entries spanning volumes; repair mode hangs interactively
#     (left stuck PIDs once); naive cat misreads disk-relative offsets. All proven
#     empirically -- see README.md. split_zip_extract.py is the sole extractor.
#   * Every external invocation is stdin-null guarded (</dev/null) and fixture builds
#     are timeout-wrapped: no interactive prompt can ever block this script.
# Env overrides: AMASS_BASE, AMASS_ZIPBASE, AMASS_OUT, AMASS_POLL_SEC, PY
# Red lines: no training dirs / gate_log access; CPU only; zero unpack IO before gates.
set -uo pipefail

AMASS_BASE="${AMASS_BASE:-/opt/data/private/yuanqi_ops/data/motionx}"
AMASS_ZIPBASE="${AMASS_ZIPBASE:-AMASS_smplxg}"
HERE="$(cd "$(dirname "$0")" && pwd)"
AMASS_OUT="${AMASS_OUT:-$HERE/data/amass_raw}"
AMASS_POLL_SEC="${AMASS_POLL_SEC:-600}"
PY="${PY:-/opt/data/private/hechengzhu/ZHY/data/env/bin/python}"
LOG="$HERE/unpack.log"
MARKER="ALL_DOWNLOADS_DONE"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

volumes() { ls "$AMASS_BASE/$AMASS_ZIPBASE".z[0-9][0-9] 2>/dev/null | sort; }

all_parts() { volumes; [ -f "$AMASS_BASE/$AMASS_ZIPBASE.zip" ] && echo "$AMASS_BASE/$AMASS_ZIPBASE.zip"; }

cmd_status() {
  echo "base=$AMASS_BASE zipbase=$AMASS_ZIPBASE"
  echo "out=$AMASS_OUT poll=${AMASS_POLL_SEC}s py=$PY"
  if grep -q "$MARKER" "$AMASS_BASE/dl_progress.log" 2>/dev/null; then
    echo "gate1 marker: PRESENT"
  else
    echo "gate1 marker: absent (download still running)"
  fi
  echo "volumes: $(volumes | wc -l) split parts; final .zip: $([ -f "$AMASS_BASE/$AMASS_ZIPBASE.zip" ] && echo yes || echo no)"
  [ -f "$HERE/unpack_DONE" ] && echo "unpack_DONE: $(cat "$HERE/unpack_DONE")"
  df -h "$AMASS_BASE" | sed 1d
  tail -5 "$LOG" 2>/dev/null
  return 0
}

cmd_validate() {
  # Small-scale proof of the exact production path: create split fixture (CREATION
  # ONLY via 'zip -r -s', timeout+stdin-null guarded) -> split_zip_extract.py.
  local T; T=$(mktemp -d /tmp/amass_split_test.XXXXXX)
  log "validate: sandbox $T"
  mkdir -p "$T/src"
  head -c 3000000 /dev/urandom > "$T/src/a.bin" 2>/dev/null
  head -c 2000000 /dev/urandom > "$T/src/b.bin" 2>/dev/null
  echo "hello amass bridge" > "$T/src/c.txt"
  if ! ( cd "$T/src" && timeout -k 3 60 zip -q -r -s 1m ../test.zip . </dev/null ); then
    log "validate FAIL: fixture creation"; rm -rf "$T"; return 1
  fi
  ls "$T"/test.z[0-9][0-9] >/dev/null 2>&1 || { log "validate FAIL: no split volumes"; rm -rf "$T"; return 1; }
  head -c 4096 /dev/urandom > "$T/test.z05" 2>/dev/null   # ghost extra volume (z29 incident)
  log "validate: volumes $( ls "$T" | grep '^test.z' | tr '\n' ' ') (+injected ghost test.z05)"
  if ! timeout -k 3 120 "$PY" "$HERE/split_zip_extract.py" "$T/test.zip" "$T/out" </dev/null >>"$LOG" 2>&1; then
    log "validate FAIL: split_zip_extract"; rm -rf "$T"; return 1
  fi
  if ( cd "$T/src" && find . -type f -exec sha256sum {} + ) </dev/null | sed 's| \./| |' | sort > "$T/g.txt" \
     && ( cd "$T/out" && find . -type f -exec sha256sum {} + ) </dev/null | sed 's| \./| |' | sort > "$T/g2.txt" \
     && diff -q "$T/g.txt" "$T/g2.txt" >/dev/null; then
    log "validate PASS: split -> split_zip_extract sha256 identical ($( wc -l < "$T/g.txt" ) files)"
    rm -rf "$T"
    return 0
  fi
  log "validate FAIL: sha256 mismatch"
  rm -rf "$T"
  return 1
}

wait_marker() {
  until grep -q "$MARKER" "$AMASS_BASE/dl_progress.log" 2>/dev/null; do
    log "gate1 waiting: $MARKER not yet (volumes: $(volumes | wc -l)); next poll ${AMASS_POLL_SEC}s"
    sleep "$AMASS_POLL_SEC"
  done
  log "gate1 open: $MARKER present"
}

wait_stable() {
  while [ ! -f "$AMASS_BASE/$AMASS_ZIPBASE.zip" ]; do
    log "gate2 waiting: final .zip not present yet"
    sleep "$AMASS_POLL_SEC"
  done
  while :; do
    local parts; parts=$(all_parts)
    local h1; h1=$(stat -c '%n %s' $parts 2>/dev/null | sha256sum | cut -d' ' -f1)
    sleep 90
    local parts2; parts2=$(all_parts)
    local h2; h2=$(stat -c '%n %s' $parts2 2>/dev/null | sha256sum | cut -d' ' -f1)
    if [ "$h1" = "$h2" ] && [ "$parts" = "$parts2" ]; then
      log "gate2 open: volume set stable ($( echo "$parts" | wc -l ) parts, $( du -scb $parts | tail -1 | cut -f1 ) bytes)"
      return 0
    fi
    log "gate2 re-check: volume set still changing"
  done
}

check_disk() {
  local need have
  need=$(( $(du -scb $(all_parts) | tail -1 | cut -f1) * 115 / 100 ))
  have=$(df -B1 "$AMASS_BASE" | awk 'NR==2{print $4}')
  if [ "$have" -lt "$need" ]; then
    log "ABORT: need ~$need bytes free, have $have"
    return 1
  fi
  log "disk ok: need ~$need, have $have"
  return 0
}

cmd_run() {
  wait_marker || return 1
  wait_stable || return 1
  check_disk || return 1
  mkdir -p "$AMASS_OUT"
  log "extracting split set -> $AMASS_OUT (split_zip_extract.py, per-entry CRC32)"
  if ! "$PY" "$HERE/split_zip_extract.py" "$AMASS_BASE/$AMASS_ZIPBASE.zip" "$AMASS_OUT" </dev/null >>"$LOG" 2>&1; then
    log "FAIL: extraction failed; volumes untouched; rerun resumes (existing correct-size files skipped)"
    return 1
  fi
  local N; N=$(find "$AMASS_OUT" -name '*.npz' | wc -l)
  find "$AMASS_OUT" -name '*.npz' | sort > "$HERE/unpack_manifest.txt"
  echo "$(date '+%F %T') DONE npz=$N" > "$HERE/unpack_DONE"
  log "DONE npz=$N -> $AMASS_OUT"
}

case "${1:-run}" in
  run) cmd_run ;;
  validate) cmd_validate ;;
  status) cmd_status ;;
  *) echo "usage: $0 run|validate|status"; exit 2 ;;
esac
