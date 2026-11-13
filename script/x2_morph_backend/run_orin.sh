#!/bin/sh
# Process-local policy + ROS environment; control remains explicit opt-in.
set -eu
backend_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
backend_python=${X2_BACKEND_PYTHON:-"$backend_root/data/environments/orin-py310/bin/python"}
backend_sdk=${X2_BACKEND_SDK:-"/home/agi/aimdk_msgs_pc2"}
if [ ! -x "$backend_python" ]; then
    echo "Project-local Orin Python 3.10 is required." >&2
    exit 1
fi
cd "$backend_root"
backend_module=script.x2_morph_backend
case "${1:-}" in
  sensor-profile) backend_module=script.x2_morph_backend.sensor_profile; shift ;;
  mc) backend_module=script.x2_morph_backend.mc_handoff; shift ;;
  control) backend_module=script.x2_morph_backend.control_session; shift ;;
  native-probe) backend_module=script.x2_morph_backend.native_probe; shift ;;
esac
exec env -u PYTHONHOME PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="$backend_root:$backend_sdk/local/lib/python3.10/dist-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages" \
    LD_LIBRARY_PATH="$backend_sdk/lib:/opt/ros/humble/lib:/agibot/software/common/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    AMENT_PREFIX_PATH="$backend_sdk:/opt/ros/humble:/agibot/software/common" \
    "$backend_python" -B -m "$backend_module" "$@"
