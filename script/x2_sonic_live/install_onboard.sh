#!/usr/bin/env bash
set -euo pipefail

# Verify onboard SHA-256s, create .bak_20261116_pre_sonic_fix backups, install
# patched production files + offline tests on X2 SoC1, and run verification.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHED="$HERE/onboard_patched"
TELEOP_PY="${TELEOP_PY:-/home/agi/.old_agi_home.bak/miniconda3/envs/teleop/bin/python}"
BACKUP_SUFFIX=".bak_20261116_pre_sonic_fix"
ALLOW_REINSTALL=0
if [ "${1:-}" = "--update" ]; then
  ALLOW_REINSTALL=1
fi

X2_SW_DST="/home/agi/projects/project_tools/x2_smpl_wholebody"
SMARTWEAR_DST="/home/agi/projects/smartwear_v2"

if [ ! -x "$TELEOP_PY" ]; then
  echo "ERROR: teleop python not found at $TELEOP_PY (run this script on X2 SoC1)" >&2
  exit 2
fi

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

install_one() {
  local src="$1"
  local dst="$2"
  local expected_orig_sha="$3"
  local live_pattern="${4:-}"

  if [ ! -f "$src" ]; then
    echo "ERROR: missing patched source: $src" >&2
    exit 2
  fi
  if [ ! -f "$dst" ]; then
    echo "ERROR: missing target on board: $dst" >&2
    exit 2
  fi

  local src_sha dst_sha bak
  src_sha="$(sha256_of "$src")"
  dst_sha="$(sha256_of "$dst")"
  bak="${dst}${BACKUP_SUFFIX}"

  if [ -n "$live_pattern" ] && pgrep -af "$live_pattern" >/dev/null 2>&1; then
    echo "SKIP $dst (live process matches '$live_pattern')"
    pgrep -af "$live_pattern"
    return 0
  fi

  if [ "$dst_sha" = "$src_sha" ] && [ -f "$bak" ]; then
    local bak_sha
    bak_sha="$(sha256_of "$bak")"
    if [ "$bak_sha" = "$expected_orig_sha" ]; then
      echo "OK   $dst already matches patched sha ($src_sha); backup verified ($bak_sha)"
      return 0
    fi
  fi

  if [ "$dst_sha" != "$expected_orig_sha" ]; then
    if [ "$ALLOW_REINSTALL" -eq 1 ] && [ -f "$bak" ] && [ "$(sha256_of "$bak")" = "$expected_orig_sha" ]; then
      echo "UPD  $dst (verified backup $bak exists)"
    else
      echo "ERROR: SHA-256 mismatch on $dst" >&2
      echo "  found:    $dst_sha" >&2
      echo "  expected: $expected_orig_sha" >&2
      exit 3
    fi
  fi

  if [ ! -e "$bak" ]; then
    cp -p "$dst" "$bak"
    echo "BAK  $dst -> $bak ($dst_sha)"
  else
    local bak_sha
    bak_sha="$(sha256_of "$bak")"
    if [ "$bak_sha" != "$expected_orig_sha" ]; then
      echo "ERROR: existing backup $bak has unexpected SHA $bak_sha" >&2
      exit 3
    fi
    echo "BAK  reusing verified $bak ($bak_sha)"
  fi

  cp "$src" "${dst}.tmp_install"
  mv -f "${dst}.tmp_install" "$dst"
  local new_sha
  new_sha="$(sha256_of "$dst")"
  if [ "$new_sha" != "$src_sha" ]; then
    echo "ERROR: post-copy SHA mismatch on $dst ($new_sha != $src_sha)" >&2
    exit 4
  fi
  echo "INST $dst ($expected_orig_sha -> $new_sha)"
}

echo "=== [1/3] Checking & installing patched files on X2 ==="
install_one \
  "$PATCHED/x2_smpl_wholebody/garment_intent_adapter.py" \
  "$X2_SW_DST/garment_intent_adapter.py" \
  "9c9d4386c8779650e702946a6ef3e52dc26062836277953971794382243d1b39"

install_one \
  "$PATCHED/x2_smpl_wholebody/x2_smpl_token_service.py" \
  "$X2_SW_DST/x2_smpl_token_service.py" \
  "847a4915bb9b6d44b589544009e94dc4807511d2d4725635ac6008848e2b3526" \
  "x2_smpl_token_service\.py"

install_one \
  "$PATCHED/smartwear_v2/garment_udp_v2_reconnect_safe.py" \
  "$SMARTWEAR_DST/garment_udp_v2_reconnect_safe.py" \
  "d7c517605b1ed7c3970e8ff31a3eac87f646681821561bfde586e111408bdf4b"

install_one \
  "$PATCHED/x2_smpl_wholebody/tests/test_garment_adapter.py" \
  "$X2_SW_DST/tests/test_garment_adapter.py" \
  "b3fff196654dd8ef4dc2147ecc44f08eb479babc1fbf4c3ef79989e543407742"

install_one \
  "$PATCHED/x2_smpl_wholebody/tests/test_token_chain.py" \
  "$X2_SW_DST/tests/test_token_chain.py" \
  "19bdb69a3cbb6e5dd40135532a92e92c4a9f677b139cb58d0d2ab7bce55621b2"

echo "=== [2/3] Running installed x2_smpl_wholebody test suite + fix suite ==="
(
  cd "$X2_SW_DST"
  PYTHONPATH="$X2_SW_DST" "$TELEOP_PY" tests/test_garment_adapter.py
  PYTHONPATH="$X2_SW_DST" "$TELEOP_PY" tests/test_heading_guards.py
  PYTHONPATH="$X2_SW_DST" "$TELEOP_PY" tests/test_token_chain.py
)
"$TELEOP_PY" "$HERE/test_onboard_fixes.py"

echo "=== [3/3] Checking smartwear_v2 garment script syntax & --help ==="
(
  cd "$SMARTWEAR_DST"
  "$TELEOP_PY" -m py_compile "$SMARTWEAR_DST/garment_udp_v2_reconnect_safe.py"
  "$TELEOP_PY" "$SMARTWEAR_DST/garment_udp_v2_reconnect_safe.py" --help >/dev/null
  echo "OK   garment_udp_v2_reconnect_safe.py py_compile & --help passed"
)

echo "=== Installed SHA-256 summary ==="
sha256sum \
  "$X2_SW_DST/garment_intent_adapter.py" \
  "$X2_SW_DST/x2_smpl_token_service.py" \
  "$X2_SW_DST/tests/test_garment_adapter.py" \
  "$X2_SW_DST/tests/test_token_chain.py" \
  "$SMARTWEAR_DST/garment_udp_v2_reconnect_safe.py"
