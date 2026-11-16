#!/usr/bin/env bash
# Build the read-only diagnostic SONIC deploy (cycle trace + watchdog provenance)
# and the provenance-tagging watchdog into a persistent, isolated workspace.
# Never touches /home/agi/gear-sonic itself. Run on X2 SoC1 as agi.
set -eo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
preview="$here/../x2_fgp_preview"
ws=${X2_SONIC_DIAG_WS:-/home/agi/x2_sonic_live/diag_ws}
sonic=/home/agi/gear-sonic

expected_cpp=e88baa7e08ead6062030fae75ec969ef74ac9144fb72e732552e2e7344e9fb7d
actual_cpp=$(sha256sum "$sonic/ws/src/x2/agi_x2_deploy_onnx_ref/src/x2_deploy_onnx_ref.cpp" | cut -d' ' -f1)
if [[ $actual_cpp != "$expected_cpp" ]]; then
    echo "x2_deploy_onnx_ref.cpp differs from the audited source ($actual_cpp)" >&2
    exit 1
fi

rm -rf "$ws"
mkdir -p "$ws/src/x2"
cp -a "$sonic/ws/src/x2/agi_x2_deploy_onnx_ref" "$ws/src/x2/"
cp -a "$sonic/ws/src/common" "$ws/src/"
( cd "$ws/src/x2/agi_x2_deploy_onnx_ref"
  patch -p3 < "$preview/sonic_cycle_trace.patch"
  patch -p3 < "$preview/sonic_reference_trace.patch" )

cp "$preview/sonic_reference_provenance.py" "$ws/"
python3 "$preview/build_watchdog_reference_trace.py" \
    "$sonic/gear_sonic_deploy/scripts/x2_pose_watchdog.py" "$ws/x2_pose_watchdog_trace.py"

set +u
source /opt/ros/humble/setup.bash
set -u
export CMAKE_PREFIX_PATH="/agibot/data/home/agi/aimdk_msgs_pc2:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$sonic/onnxruntime/lib:/agibot/data/home/agi/aimdk_msgs_pc2/lib:/opt/ros/humble/lib:${LD_LIBRARY_PATH:-}"
cd "$ws"
colcon build --packages-select agi_x2_deploy_onnx_ref \
    --cmake-args -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT="$sonic/onnxruntime"
binary="$ws/install/agi_x2_deploy_onnx_ref/lib/agi_x2_deploy_onnx_ref/x2_deploy_onnx_ref"
sha256sum "$binary" "$ws/x2_pose_watchdog_trace.py"
echo "DIAG_READY $binary"
