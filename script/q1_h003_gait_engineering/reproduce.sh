#!/usr/bin/env bash
# Run from this source directory on the existing SuperMimicHP evaluation installation.
set -euo pipefail
project=${1:?Usage: reproduce.sh SUPER_MIMIC_HP_ROOT NEW_RUN_NAME}
name=${2:?Supply a new run name under runs/}
[[ "$name" =~ ^[[:alnum:]_-]+$ ]] || { echo "Invalid run name" >&2; exit 2; }
project=$(cd "$project" && pwd)
src=$(cd "$(dirname "$0")" && pwd)
original="$project/runs/h003_recovery_expert"
previous="$project/runs/h003_recovery_handoff_study"
dest="$project/runs/$name"
py="$project/.venv/bin/python"
test -x "$py"
test -f "$original/formal_03/q1_recovery_expert.onnx"
test -f "$previous/blind01/cases.json"
# An existing evidence directory must never receive another run's results.
mkdir "$dest"
cd "$dest"
cp "$original"/{bench,check_base,delivery,leg_height,q1_recovery_runtime,recovery}.py .
cp -r "$original/dependencies" .
cp "$previous/score_gait.py" .
cp "$src"/{assess,check_reference,reference,regression,relay,render_engineering,repeated,restart_profile,run_suite}.py .
ln -s "$original/formal_03" formal_03
ln -s "$previous" previous_study
export Q1_V2B_BUNDLE="$dest/formal_03/pinned/q1_v2b_gantry_deploy_bundle"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

run_group() {
    local suite=$1 output=$2
    shift 2
    local profile pid failed=0
    local -a pids=()
    mkdir "$output"
    for profile in "$@"; do
        "$py" run_suite.py "$suite" "$profile" "$output/$profile" > "$output/$profile.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    if (( failed )); then echo "Check $dest/$output/*.log" >&2; return 1; fi
}
arms=(published fast damped_moderate)
run_group dev dev "${arms[@]}" moderate damped_fast
"$py" assess.py dev
run_group confirm confirm "${arms[@]}"
"$py" assess.py confirm
run_group flat flat "${arms[@]}"
run_group repeat repeat_checked "${arms[@]}"
"$py" bench.py static --expert formal_03/q1_recovery_expert.onnx --public-runtime > static.log
mkdir direct
for profile in published damped_moderate; do
    "$py" relay.py formal_03/q1_recovery_expert.onnx "direct/$profile" --tracks formal_03/input_tracks \
        --public-runtime --gait-profile "$profile" > "direct/$profile.log"
done
"$py" check_reference.py
"$py" regression.py
echo "Review $dest/regression.json"
