#!/bin/bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$project_root"
output=${1:?Pass the new experiment root}
python3 -m script.x2_morph_rpc_budget.prepare --output "$output" --profile data/analysis/X2_MORPH_NATIVE_SENSOR_V2_20261113.json
set +u
source /opt/ros/humble/setup.bash
set -u
sdk=/home/agi/aimdk_msgs_pc2
export AMENT_PREFIX_PATH="$sdk:${AMENT_PREFIX_PATH:-}"
export CMAKE_PREFIX_PATH="$sdk:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$sdk/lib:${LD_LIBRARY_PATH:-}"
cmake -S "$output/source" -B "$output/build" -DCMAKE_BUILD_TYPE=Release \
  -DSONIC_SOURCE=/home/agi/gear-sonic/ws/src/x2/agi_x2_deploy_onnx_ref \
  -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DMORPH_RUNTIME="$project_root/data/runtime/x2_morph_human"
cmake --build "$output/build" --parallel 2
ctest --test-dir "$output/build" --output-on-failure
