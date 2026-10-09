#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/media/yu/FAFF-E9771/YUANQI}"
RUN_ID="${RUN_ID:-smoke_16env_2iter}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid RUN_ID' >&2; exit 2; }
export REPO_DIR="$PROJECT_ROOT/script/vendor/sonic_for_a3"
export ISAAC_PYTHON="$PROJECT_ROOT/data/environments/a3-sonic/bin/python"
export MOTION_FILE="$PROJECT_ROOT/data/training/a3_20261009/motionlib"
export EXPERIMENT_DIR="$PROJECT_ROOT/data/training/a3_20261009/$RUN_ID"
export NUM_ENVS="${NUM_ENVS:-16}"
export NUM_MINI_BATCHES="${NUM_MINI_BATCHES:-1}"
export NUM_LEARNING_ITERATIONS="${NUM_LEARNING_ITERATIONS:-2}"
export NUM_PROCESSES=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export ISAACLAB_APP_LAUNCHER_LOCK_TIMEOUT_SECONDS=180
export HYDRA_FULL_ERROR=1
export WANDB_MODE=disabled

cd "$PROJECT_ROOT"
LOG_DIR="$PROJECT_ROOT/logs/a3_training_20261009/$RUN_ID"
mkdir -p "$LOG_DIR" "$EXPERIMENT_DIR"
if [[ -e "$LOG_DIR/exit_code" || -e "$EXPERIMENT_DIR/config.yaml" ]]; then
    echo "Run already exists; choose another RUN_ID: $RUN_ID" >&2
    exit 2
fi
date -Is > "$LOG_DIR/started_at.txt"
echo "$$" > "$LOG_DIR/wrapper.pid"
ln -s "$LOG_DIR/launcher.log" "$EXPERIMENT_DIR/train.rank0.log"
set +e
bash "$REPO_DIR/train_a3_035_fromscratch.sh" \
    seed=0 callbacks.model_save.save_frequency="${SAVE_FREQUENCY:-1}" \
    +callbacks.model_save.save_last_frequency="${SAVE_LAST_FREQUENCY:-1}" \
    "$@" > "$LOG_DIR/console.log" 2>&1
result=$?
set -e
echo "$result" > "$LOG_DIR/exit_code"
date -Is > "$LOG_DIR/finished_at.txt"
exit "$result"
