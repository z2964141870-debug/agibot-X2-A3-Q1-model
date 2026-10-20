#!/usr/bin/env bash
set -euo pipefail
export HYDRA_FULL_ERROR=1
export YUANQI_RESUME_LOG_DIR=/opt/data/private/yuanqi_ops/logs/p2_main/m11_resume
export MOTION_FILE=/opt/data/private/yuanqi_ops/data/training/p2_main/m11_motionlib
export PYTHONPATH=/opt/data/private/yuanqi_ops:/opt/data/private/yuanqi_ops/script:/opt/data/private/hechengzhu/ZHY:$PYTHONPATH
mkdir -p /opt/data/private/yuanqi_ops/logs/p2_main/m11_resume
mkdir -p /opt/data/private/yuanqi_ops/data/training/p2_main/m11/checkpoints
mkdir -p /opt/data/private/yuanqi_ops/data/training/p2_main/m11/expdir
bash /opt/data/private/hechengzhu/ZHY/script/vendor/sonic_for_a3/train_a3_035_fromscratch.sh   --isaac-python /opt/data/private/hechengzhu/ZHY/data/env/bin/python   algo.config.num_learning_iterations=32   num_envs=16   checkpoint=/opt/data/private/yuanqi_ops/data/training/p2_main/m9c/checkpoints/milestone_128/model_step_000128.pt   exp_var=a3_035_finetune_m11_truetarget   experiment_dir=/opt/data/private/yuanqi_ops/data/training/p2_main/m11/expdir   seed=0   trainer._target_=script.a3.finetune_trainer.FineTuneTrainer   algo.config.num_learning_epochs=1   algo.config.actor_learning_rate=2e-5   algo.config.adaptive_lr_min=1e-6   algo.config.adaptive_lr_max=2e-5   callbacks.model_save._target_=e54_callback.E54MilestoneCallback   callbacks.model_save.save_frequency=8   +callbacks.model_save.target_global_step=32   callbacks.model_save.save_dir=/opt/data/private/yuanqi_ops/data/training/p2_main/m11/checkpoints   ++algo.trl.bf16=false ++algo.trl.fp16=false ++algo.trl.tf32=false   > /opt/data/private/yuanqi_ops/logs/p2_main/m11_train.log 2>&1 &
echo P2_M11_PID=$!
