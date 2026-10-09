# A3 中断续训与模型持久化

记录日期：2026-10-09。状态：8 项 CPU 故障检查通过，E003 / E004 两次实际续训完成 150→152→154；E005 已启动，11:13:54 服务 active/running，随后 selector 验证 step 300 模型。尚未完成累计 2,000 更新或固定条件评测。

## 目标与决定

用户接受以中断后续训推进训练，怀疑电源不稳定。现有日志未定位两次重启原因，不以软件未发现记录证明一定是电源。此次改善恢复机制，不操作真机，不主动断电测试，不自动无限重启训练。

沿用 A3 官方 SONIC `fe6868ba37034f89b912f0fb851bce19f120266d` 的 PPO、奖励和物理配置；新增本项目适配入口，不改第三方源码。恢复起点为 E002 的已验证 step 150 模型。

## 已发现的恢复限制

- 官方 launcher 强制 `resume=false`，属于加载权重的 warm-start；需显式使用 `++resume=true`。
- `resume_training` 默认 `resume_in_place=true`，会把输出改回 checkpoint 父目录。首次 E003 两步测试因此把模型 / TB 写入 E002；已补 `++resume_in_place=false`，本次生成文件迁至 E003，E002 配置从该实验原 `.hydra/config.yaml` 恢复并标注重建来源，原 step 150 recovery 文件保留。
- trainer 能加载优化器、scheduler、训练计数和 motion-lib，但会用静态 learning_rate 覆盖优化器当前 LR；适配器重新恢复完整 optimizer state 并核验。
- 官方循环再次执行 `num_total_batches` 次，未减掉已完成更新；适配器以累计目标减去已完成 global_step 得到剩余轮数。
- environment checkpoint 只含 motion-lib，不含完整 PhysX 世界；新的仿真 episode 重新开始，未完成回报累计必须清零。恢复不是逐帧或逐位完全等价的连续训练。
- 官方保存采用临时文件 / rename，没有显式文件和目录 fsync。本轮增加刷盘与发布前重载 / SHA 校验，保留最近两个有效版本及 500 步里程碑；不能保证硬件故障时零损失。

## 实现与验证

- `script/a3/checkpoint_store.py`：文件写完 fsync、重载 / 有限性检查、SHA、目录 fsync、原子发布 sidecar / last.pt；损坏时选择上一份校验通过的版本。
- `script/a3/resumable_sonic.py`：在官方 trainer / callback 上适配恢复 LR、计数、剩余更新，保存额外全局随机状态。原 step 150 没有这些随机状态；即使新版本含 RNG，也不含完整仿真状态。
- `script/a3/run_official_training.sh`：增加显式 RESUME_MODE，保存日志和独立实验目录，不覆盖原 E002。
- `script/a3/select_resume_checkpoint.py`：只选择本项目 writer 生成且大小、哈希、重载通过的模型。
- `script/a3/test_checkpoint_store.py`：CPU 故障注入检查，用于验证序列化 / flush 失败不破坏上一份模型；不是实际断电试验。

验证顺序：CPU 故障测试 → E003 从 150 到 152 → E004 从 152 到 154 → E005 受限后台续训。优化器 step 按参数分别统计，不假定所有参数每轮都更新。

阶段检查：8 项 CPU 测试已通过，包含写入 / fsync 失败、最新模型 / sidecar 损坏回退、NaN 拒绝与保留里程碑。日志 `logs/a3_training_20261009/checkpoint_store_tests.log`，不是物理断电测试。

E003 退出码 0，实际 global_step=152，恢复时 optimizer=3,000，Actor / Critic 与原 step 150 逐张量完全匹配，恢复 LR 两组均 2e-5。首次测试命令原始日志在 E003，但上游 helper 默认写回 E002 的问题导致结果需迁移。新文件为 pid 5143 的 tfevents、model_step_000151/152 的 PT + JSON，以及本次配置；不改变任何 Q1 文件。路径修复后另做 E004 验证独立输出与下一次续训。

E004 退出码 0，恢复 global_step=152、优化器参数 step 集合为 [3000, 3040]，两组 LR 2e-5；Actor / Critic 与源 checkpoint 完全一致。最终 global_step=154。配置、模型、TensorBoard 和日志均在 E004 独立目录，`resume_in_place=false` 生效。153→154 的 Actor 35/45 个 weight/bias 张量、Critic 14/14 改变，网络浮点张量全部有限。清单为 `data/manifests/a3_resume_E004_20261009.json`；selector 成功选择 step 154。

step 154 模型：`data/training/a3_20261009/E004_resume_152to154/model_step_000154.pt`，402081685 bytes，SHA-256 `a4a72086bb77d521620c755ac196e4a7e27738a79f5009b8866af769db11ab31`。

## E005 命令与恢复

单次 user systemd 服务，64 环境、4 minibatches、累计目标 2,000、每 25 次更新保存，内存高水位 16G / 上限 20G，CPUQuota=300%，三个 CPU 线程变量均为 2。不设置开机自动续训或自动重启；这些限制不能修复未知的硬件故障。

```bash
cd /media/yu/FAFF-E9771/YUANQI
systemd-run --user --unit=yuanqi-a3-20261009-e005 --description="YUANQI A3 durable resume" --property=WorkingDirectory=/media/yu/FAFF-E9771/YUANQI --property=MemoryHigh=16G --property=MemoryMax=20G --property=CPUQuota=300% --setenv=RUN_ID=E005_resume_154to2000 --setenv=RESUME_MODE=true --setenv=CHECKPOINT=/media/yu/FAFF-E9771/YUANQI/data/training/a3_20261009/E004_resume_152to154/model_step_000154.pt --setenv=NUM_ENVS=64 --setenv=NUM_MINI_BATCHES=4 --setenv=NUM_LEARNING_ITERATIONS=2000 --setenv=SAVE_FREQUENCY=25 --setenv=SAVE_LAST_FREQUENCY=25 --setenv=OMP_NUM_THREADS=2 --setenv=MKL_NUM_THREADS=2 --setenv=OPENBLAS_NUM_THREADS=2 /bin/bash /media/yu/FAFF-E9771/YUANQI/script/a3/run_official_training.sh
systemctl --user status yuanqi-a3-20261009-e005.service --no-pager
tail -n 30 logs/a3_training_20261009/E005_resume_154to2000/launcher.log
```

模型目录 `data/training/a3_20261009/E005_resume_154to2000/`，终端记录 `logs/a3_training_20261009/E005_resume_154to2000/`。以上启动命令只能执行一次；重启后先核实服务 / GPU 状态，再执行 selector：

```bash
PYTHONPATH=$PWD OMP_NUM_THREADS=2 data/environments/a3-sonic/bin/python script/a3/select_resume_checkpoint.py data/training/a3_20261009/E005_resume_154to2000
```

selector 校验大小、SHA 和重载，最新损坏时回退旧版本；如果 E005 尚无有效模型，使用 E004 step154。下一次启动须使用新 RUN_ID / service 名，并把 CHECKPOINT 改为 selector 的实际输出。每 25 次保存意味着可能丢失尚未保存的更新；文件系统 / 设备故障仍可能损坏多份文件，不能保证只丢 25 次或零损失。

E005 实际验证：北京时间 11:13:54 主机 uptime 1:30，服务 active/running，MemoryCurrent=5447745536 bytes；未发生本轮可确认的重启。中间两次 SSH 建连超时，之后恢复，因此不以超时单独判断主机重启。恢复回执为 `logs/a3_training_20261009/E005_resume_154to2000/resume_loaded.json`：global_step=154，优化器参数 step 集合 [3000, 3080]，两组 LR=2e-5，Actor / Critic 逐张量匹配。旧计数 3000 表示部分参数未继续更新，不把优化器计数集合描述为统一单值。

E005 已产生 step275 / 300 checkpoint sidecar；随后 selector 通过大小 / SHA / 重载检查并选择 `model_step_000300.pt`。step300 是核验时快照，最新两份会随训练被替换，500 步里程碑保留。11:14:59 标量快照到 step329，检查的标量全部有限；训练 reward 0.772→1.311，但 error_body_pos 0.101→0.116、error_joint_pos 0.201→0.216，不能据此认定跟踪改善。训练标量快照记录于 `data/manifests/a3_E005_status_20261009.json`，不等同固定条件评测。下一步为完成固定条件跟踪评测，不以训练 reward 上升宣称可遥操或真机部署。

## 保存与边界

模型留 hp3090 的 data，网盘仍为 LOCAL_ONLY。代码、结论与小清单提交 main；完整版本以本文件的 Git 提交历史为准，发布完成须核验 HEAD 与 refs/heads/main 一致。官方源码固定 fe6868ba37034f89b912f0fb851bce19f120266d；适配改动位于本项目 script/a3。与 Q1 工作区无关，不修改它。
