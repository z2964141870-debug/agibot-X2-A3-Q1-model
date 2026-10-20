cd /opt/data/private/hechengzhu/ZHY
export YUANQI_CLOTH_ANCHOR_ROLLOUT=/opt/data/private/hechengzhu/ZHY/data/experiments/a3_zhy_gradient_20261012_E11/rollout.pt
export YUANQI_CLOTH_ANCHOR_OUTPUTS=/opt/data/private/yuanqi_ops/data/experiments/a3_zhy_rootchannel_20261019_E53/anchors_035.pt
export YUANQI_E55_ARM=Q
export YUANQI_E55_TIER=64
export YUANQI_E55=1
export ISAAC_PYTHON=/opt/data/private/hechengzhu/ZHY/data/environments/a3-sonic/bin/python
export MOTION_FILE=/opt/data/private/yuanqi_ops/data/training/a3_zhy_residual_p1_20261019_E55/arm_q_tier64/motionlib
export PYTHONPATH=/opt/data/private/yuanqi_ops:/opt/data/private/hechengzhu/ZHY:$PYTHONPATH
export YUANQI_RESUME_LOG_DIR=/opt/data/private/yuanqi_ops/logs/p12_p12run/q
export E54_CHANNEL_STATS=/opt/data/private/yuanqi_ops/data/experiments/a3_zhy_channel_probe_20261019_E54/e54_channel_stats.json
export HYDRA_FULL_ERROR=1
bash /opt/data/private/hechengzhu/ZHY/script/vendor/sonic_for_a3/train_a3_035_fromscratch.sh --isaac-python /opt/data/private/hechengzhu/ZHY/data/env/bin/python algo.config.num_learning_iterations=64 num_envs=16 checkpoint=/opt/data/private/hechengzhu/ZHY/data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt exp_var=a3_035_finetune_selected20 experiment_dir=/opt/data/private/yuanqi_ops/data/training/p12_p12run/q/expdir seed=0 trainer._target_=e55_trainer.E55ResidualTrainer algo.config.num_learning_epochs=1 algo.config.actor_learning_rate=4e-5 algo.config.adaptive_lr_min=4e-5 algo.config.adaptive_lr_max=4e-5 callbacks.model_save._target_=e54_callback.E54MilestoneCallback callbacks.model_save.save_frequency=8 +callbacks.model_save.target_global_step=64 callbacks.model_save.save_dir=/opt/data/private/yuanqi_ops/data/training/p12_p12run/q/checkpoints ++algo.trl.bf16=false ++algo.trl.fp16=false ++algo.trl.tf32=false manager_env.observations.tokenizer.motion_anchor_ori_b_mf_nonflat.func=e55_obs_terms:e55_root_channels_q > /opt/data/private/yuanqi_ops/p12_run_q.log 2>&1 &
echo PID=$!
