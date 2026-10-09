# A3 官方训练工程首次尝试

记录日期：2026-10-09（北京时间）。阶段：a3-training-20261009。状态：16 环境训练 / 保存 / 重载通过；E001 和 E002 均遇服务器重启中断。E002 第 150 次更新模型可恢复，重载通过；未完成 2,000 更新，当前无训练作业。

## 目标与授权

用户要求直接依照官方工程尝试训练，使用 hp3090 的 RTX 3090。先完成官方小规模训练验证，检查 checkpoint 与重载，再决定扩大训练或微调。本次授权包含训练所需的软件与数据准备，不包含真机部署。现有 Q1 衣服数据继续为 `TEST_ONLY`，使用 A3 官方示例动作。

## 环境与初步核验

- 主项目：`hp3090:/media/yu/FAFF-E9771/YUANQI`，分支 `main`；读取时保留六项上一学习阶段未提交修改。
- GPU 本轮可见 RTX 3090 24576 MiB，查询时显存使用 92 MiB、利用率 0%。项目盘可用约 591GB。
- 候选 Python：`/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python`，Python 3.11.15、PyTorch 2.8.0+cu128，CUDA 12.8 可用；Isaac Lab、MuJoCo、Hydra、OmegaConf 与 huggingface_hub 模块存在。存在不表示完整 A3 任务通过。
- 新第三方源码拟保存 `script/vendor/sonic_for_a3/`，主仓库忽略整份 vendor 副本；配置版本、来源与必要适配由小型报告 / 清单保存。大型依赖、资产和产物留服务器。

## 当前检查与下一步

1. 核验官方仓库当前版本、许可、训练脚本与必需资产。
2. 准备官方示例动作库，执行入口检查与小规模训练。
3. 实际发生 Actor 参数更新、checkpoint 保存和重载后记录结果；不能用入口检查或仿真加载替代训练成功。
4. 根据峰值显存与运行性能扩大训练；长作业需保留独立日志、作业标识、恢复方式和状态。

## 已完成的软件与数据准备

- 官方源码：`script/vendor/sonic_for_a3`，来源 `AgibotTech/sonic_for_a3`，固定提交 `fe6868ba37034f89b912f0fb851bce19f120266d`；SSH clone 成功，219 个 LFS 资产已实化。HTTPS GitHub 连接失败，本轮不继续重复 HTTPS clone。
- 已读根 LICENSE、NOTICE 与 `docs/THIRD_PARTY_NOTICES.md`：代码 Apache-2.0；A3 资产保留 Mulan PSL v2；selected20 示例发行授权限于公开的 20 个 CSV，不能推及其他数据。第三方代码和数据留被忽略的服务器目录，不提交本项目 Git。
- Python overlay：`data/environments/a3-sonic`，`venv --system-site-packages` 复用既有 Isaac Lab / CUDA；局部安装 SONIC editable、NumPy 1.26.4、Hydra 1.3.2。原 conda 环境不改动。官方 `check_environment.py --training` 全部通过。
- pip 提示既有 `mink/mujoco`、Isaac Sim 的 click/psutil 版本及 NumPy 精确锁定不一致；此处记录而不视为检查通过即可保证运行，实际训练用于核验。安装日志 `logs/a3_training_20261009/install_*.log`。
- 使用官方 converter 将 120 Hz CSV 转为 30 Hz 动作库：20/20 成功，0 失败；输出 `data/training/a3_20261009/motionlib/agibot_a3/`，日志 `logs/a3_training_20261009/convert_motion.log`。输出包含米 / 弧度、29 维可驱动关节参考；头两关节不进入 action。
- 包装入口 `script/a3/run_official_training.sh`：保留上游训练算法、奖励和物理配置；将产物放 data、stdout 放 logs；上游 `train.rank0.log` 为指向 logs 的符号链接。设置种子 0，首次检查每步保存 checkpoint，正式实验可调整保存间隔。

复现准备与首次训练（仅 hp3090；不操作真机）：

```bash
cd /media/yu/FAFF-E9771/YUANQI
data/environments/a3-sonic/bin/python script/vendor/sonic_for_a3/check_environment.py --training
data/environments/a3-sonic/bin/python script/vendor/sonic_for_a3/gear_sonic/data_process/convert_soma_csv_to_motion_lib.py --input script/vendor/sonic_for_a3/a3_data/agibot_a3 --output data/training/a3_20261009/motionlib --individual --robot a3_29 --fps 30 --fps_source 120 --num_workers 4
bash script/a3/run_official_training.sh
```

首次 run ID `smoke_16env_2iter`，16 环境、2 次 PPO 采集 / 更新，1 mini-batch、每批 5 epochs，随机初始化，无预训练 checkpoint。09:25 已正常完成，退出码 0，保存 step 1、step 2 和 last.pt；Actor+Critic 参数总量 33,496,891。比对 step 1/2：两个网络实际变化，所有浮点张量有限，optimizer step 已到 10。原始 state dict 包含运行均值 / 方差 / 计数，不能把统计缓冲变化称为权重变化；独立检查脚本只统计 weight/bias/std。入口拒绝重用已经保存配置的 run ID。

新进程使用官方 `sim2sim_a3_mujoco.py`，加载 step 2 的 PT 及同目录 config.yaml，a3_fast encoder、官方匹配的被动脚 MJCF、`001_walk_front_slow.csv`，batch-once、10 policy steps、零执行延迟；退出 0、产生 metrics JSON。0.2 秒未触发跌倒不代表会走路；该项只证明重载与短闭环可运行，没有做长期性能验收。

```bash
data/environments/a3-sonic/bin/python script/vendor/sonic_for_a3/gear_sonic/scripts/sim2sim_a3_mujoco.py --checkpoint data/training/a3_20261009/smoke_16env_2iter/model_step_000002.pt --motion script/vendor/sonic_for_a3/a3_data/agibot_a3/001_walk_front_slow.csv --encoder-mode a3_fast --mjcf script/vendor/sonic_for_a3/gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml --batch-once --max-policy-steps 10 --metrics-out data/training/a3_20261009/smoke_16env_2iter/mujoco_reload_metrics.json
data/environments/a3-sonic/bin/python script/a3/inspect_checkpoints.py data/training/a3_20261009/smoke_16env_2iter/model_step_000001.pt data/training/a3_20261009/smoke_16env_2iter/model_step_000002.pt --output data/manifests/a3_smoke_20261009.json
```

日志：`logs/a3_training_20261009/smoke_16env_2iter/console.log` 与 `mujoco_reload.log`；checkpoint 每份约 402 MB，仅留服务器 data，备份为 LOCAL_ONLY。上游 Hydra 自身仍生成少量 train.log / .hydra 记录于 experiment_dir，属于上游输出布局，后续整理时保留配置和来源。

下一步扩大规模。随机初始化 selected20 的训练用于建立自主训练闭环；官方明确这个小样本集不是广泛动作跟踪的完整训练配方，后续有针对性微调或大规模动作数据路线仍需评测后判断。

## E001 扩大规模的训练作业

作业 `E001_512env_2000iter`：随机权重开始，512 环境、2,000 次更新、4 mini-batches、5 epochs / 批、种子 0，保留官方 035 奖励、机器人 / 被动脚和域随机化。每 500 步保存独立 checkpoint、每 50 步更新 last.pt。此次主要改变训练规模，不能与两步检查作为算法对照实验。

通过 `systemd-run --user` 启动，单元 `yuanqi-a3-20261009-e001.service`。首次检查显示服务 active/running，512 个物理环境创建完成；随后服务器重启，此作业已中断，不能凭此前 active/running 推断仍在跑，也没有完成 2,000 次更新。

```bash
systemd-run --user --unit=yuanqi-a3-20261009-e001 --description="YUANQI A3 SONIC selected20 training" --property=WorkingDirectory=/media/yu/FAFF-E9771/YUANQI --setenv=RUN_ID=E001_512env_2000iter --setenv=NUM_ENVS=512 --setenv=NUM_MINI_BATCHES=4 --setenv=NUM_LEARNING_ITERATIONS=2000 --setenv=SAVE_FREQUENCY=500 --setenv=SAVE_LAST_FREQUENCY=50 /bin/bash /media/yu/FAFF-E9771/YUANQI/script/a3/run_official_training.sh
systemctl --user show yuanqi-a3-20261009-e001 -p ActiveState -p SubState -p MainPID -p Result
tail -f /media/yu/FAFF-E9771/YUANQI/logs/a3_training_20261009/E001_512env_2000iter/console.log
```

服务器产物：`data/training/a3_20261009/E001_512env_2000iter/`；纯终端日志 / 退出码：`logs/a3_training_20261009/E001_512env_2000iter/`。恢复先读服务实时状态、console.log 最新迭代和 exit_code，不能凭旧 PID 推断仍在跑。wrapper 避免复用旧 run ID；重新实验需新 ID。

后续验收：确认 2,000 次更新结束且模型有限；用相同 CSV、a3_fast 与物理 / 时间配置做较长 MuJoCo 闭环评测，比较训练早期与后期的失败时间和跟踪误差，登记模型 SHA-256。当前不提供尚未测量的预计完成时间，也不把官方预训练模型结果当作本次训练成果。

## E001 中断定位与继续决定

09:31 SSH 恢复后 uptime 仅约 1 分钟，原 transient unit 不存在，GPU 无训练进程；日志停在 09:28:58 motion data loading，没有正常 exit_code、finished_at 或 checkpoint。这是服务器重启证据，不只是 SSH 断线。重启原因尚未确定：上一次启动内核 / 系统日志没有发现明确 OOM、NVIDIA Xid、panic 或正常 shutdown 记录，不能写成显存不足 / OOM 已确认。

保留 `logs/a3_training_20261009/E001_512env_2000iter/kernel_previous_boot.log` 和 `system_before_restart.log`。没有进行 reboot、改驱动、改电源限制或自动反复重试 512 环境。

下一作业降为 64 环境，使用 user systemd cgroup MemoryHigh=16G、MemoryMax=20G 限制主机 RAM 占用；限制不能保证防止未知硬件 / 内核故障。为使后台服务跨全部 SSH 登出仍运行，已执行 `loginctl --no-ask-password enable-linger yu`，核验 Linger=yes；不设置开机自动训练，也不自动重启失败训练。后续从实时日志确认更新，保留 E001 失败证据。

## E002 运行与恢复

作业 `E002_64env_2000iter`、单元 `yuanqi-a3-20261009-e002.service`，从随机初始化开始，64 环境、2,000 次更新、4 mini-batches、5 epochs、种子 0。官方最终合成配置为每环境采集 24 步（不是基础配置 32 步），每轮 1,536 条 transition。全部 2,000 轮目标约 307 万 transition，不含评测。

```bash
systemd-run --user --unit=yuanqi-a3-20261009-e002 --description="YUANQI A3 SONIC 64env training" --property=WorkingDirectory=/media/yu/FAFF-E9771/YUANQI --property=MemoryHigh=16G --property=MemoryMax=20G --setenv=RUN_ID=E002_64env_2000iter --setenv=NUM_ENVS=64 --setenv=NUM_MINI_BATCHES=4 --setenv=NUM_LEARNING_ITERATIONS=2000 --setenv=SAVE_FREQUENCY=500 --setenv=SAVE_LAST_FREQUENCY=50 /bin/bash /media/yu/FAFF-E9771/YUANQI/script/a3/run_official_training.sh
systemctl --user show yuanqi-a3-20261009-e002 -p ActiveState -p SubState -p MainPID -p MemoryCurrent -p Result
tail -f /media/yu/FAFF-E9771/YUANQI/logs/a3_training_20261009/E002_64env_2000iter/console.log
```

09:40:02 核验：TensorBoard 已记录 122 次更新 / 187,392 transition，14 个主要标量历史均有限；最近 20 轮平均采集 1.285 s、学习 1.072 s。主机内存约 7.30 GB，GPU 查询显存 3,873 MiB、利用率瞬时 65%、温度 41°C。上述是检查时快照，不能推断始终如此。

`last.pt` 首次保存已确认（约 402 MB）；详细训练标量快照 `data/manifests/a3_E002_status_20261009.json`，可用 `script/a3/summarize_training.py` 重新生成。现阶段 training reward 波动上升，但 error_body_pos / error_joint_pos 未改善、timeout 比例接近 0；采样分布在变化，这些不是固定条件评测，不能宣称已学会跟踪。

以上 09:40 为中断前快照。09:44 新核验服务器 uptime 又约 1 分钟、E002 transient unit 为 not-found、GPU 无训练进程，E002 同样已被主机重启中断，没有正常退出码。训练产物在 `data/training/a3_20261009/E002_64env_2000iter/`；当前 E001 / E002 服务均不存在，不可根据旧快照声明正在训练。

## 第二次重启与模型恢复

- E002 在 64 环境、主机内存约 7.3 GB / 显存约 3.9 GB 时已稳定更新到 150 步以上，后来重启。重启前日志再次没有明确 OOM、NVIDIA Xid、panic 或正常 shutdown 证据。主机稳定性仍是未定位问题，不把时间关联写成已确认训练代码 / 电源 / 显存故障。
- 留存本 run 的 `kernel_previous_boot.log` 与 `system_before_restart.log`；训练日志末尾写入被截断，没有 exit_code。
- `early_snapshot.pt` 在重启前能加载、SHA 与 step 150 模型一致，但重启后大小为 0，**无效，不能使用**。这是未持久化成功的副本，不把此前读取成功当成重启后保存有效。
- 原 `last.pt` 重启后重新加载成功：global_step=150、optimizer steps=3,000，Actor / Critic 所有浮点张量有限。大小 402,070,440 bytes，SHA-256 `5310a39aa3c4bbeefa4932f94ee5d0b67c5b13c8d052909fd10806e37f49e988`。
- 从有效 last.pt 另存 `recovery_step000150.pt`，执行 `sync` 刷盘并再次计算 SHA，与原文件一致。该模型在新进程通过官方 MuJoCo 10 步闭环重载；日志 `recovery_reload.log`，指标 `recovery_reload_metrics.json`，仍只是 0.2 秒接口 / 重载验证，不证明行走效果。
- 恢复清单 `data/manifests/a3_E002_recovery_20261009.json`；原 122 更新的 status.json 保留为历史观测，不代表当前进程状态。

本轮停止反复重新启动长训练，待确认两次重启是否人为以及主机供电 / 稳定性。已向用户询问此事实，同时完成记录与模型恢复；没有擅自重启服务器、更改 GPU 功率限制或驱动。用户确认 / 排查后从上述模型继续，官方包装入口默认 warm-start 并重置优化器；若需精确恢复必须单独核验 resume=true 入口与 sampler / trainer 状态，不能把它与普通微调混淆。

## 保存与同步

环境与动作、smoke checkpoints 仅在 hp3090；Mac 只接收小型说明、代码和清单。`data/manifests/a3_smoke_20261009.json` 保存两个 checkpoint 大小、SHA-256 与权重变化检查结果；backup_status=LOCAL_ONLY，尚未上传百度网盘。

源码和数据许可说明已记录，vendor 全目录被忽略。阶段报告、入口、检查脚本与小清单已通过 Git 载荷审计，训练阶段提交 `ebf83e43c7e042a8b3d680250640a9fc1131764d` 推送 main，随后核验本地 HEAD 与远端 refs/heads/main 一致，工作区干净。此前学习主线提交 `13bfe34` 一并发布；小型回执见 `logs/session_records/a3_training_publication_20261009.json`。模型没有进入 Git，Q1 专用工作区 / 分支没有修改，未操作真机。

## 保存与边界

当前仅记录环境检查，无训练模型。运行日志拟为 `logs/a3_training_20261009/`，模型 / 数据拟为 `data/training/a3_20261009/`；工件清单在 `data/manifests/`。大型产物备份未核验时标记 `LOCAL_ONLY`。

阶段完成或失败后补充准确源码 SHA、命令、指标、日志、产物和同步状态，并更新索引及当前状态。前次学习主线阶段仍需核验提交，不据本次服务器连通记为已同步。
