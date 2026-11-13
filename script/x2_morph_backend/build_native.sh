#!/bin/bash
set -euo pipefail
native_root=$(cd -- "$(dirname -- "$0")/../.." && pwd)
native_build=${1:?Usage: build_native.sh NEW_BUILD_DIRECTORY}
native_sonic=${X2_SONIC_SOURCE:-/home/agi/gear-sonic/ws/src/x2/agi_x2_deploy_onnx_ref}
native_sdk=${X2_BACKEND_SDK:-/home/agi/aimdk_msgs_pc2}
set +u
source /opt/ros/humble/setup.bash
set -u
export AMENT_PREFIX_PATH="$native_sdk:${AMENT_PREFIX_PATH:-}"
export CMAKE_PREFIX_PATH="$native_sdk:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$native_sdk/lib:${LD_LIBRARY_PATH:-}"
cmake -S "$native_root/script/x2_morph_backend/cpp" -B "$native_build" \
  -DCMAKE_BUILD_TYPE=Release -DSONIC_SOURCE="$native_sonic" \
  -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DMORPH_RUNTIME="$native_root/data/runtime/x2_morph_human"
cmake --build "$native_build" --parallel 2
ctest --test-dir "$native_build" --output-on-failure
