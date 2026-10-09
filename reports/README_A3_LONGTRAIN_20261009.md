# A3 官方策略续训至累计10000

记录日期：2026-10-09，北京时间。新任务R02，最新20:28：反复整机重启后已blocked/attempt_limit，3/3启动预算用完，2700模型独立校验通过，当前未训练。Isaac原生评测指标依赖仍阻塞。

## 最新故障核验（20:28）

用户要求继续后重新检查，服务器Git起点 `09f60d95a739289fefc024cde2209aa3ba6b7313`、工作区干净。最新服务failed/exit2，账本blocked/attempt_limit、verified_step2700，GPU无计算进程。当前boot `a95a89b9-df91-4c13-b609-86273fbd847d`；20:28:25采样CPU47°C、GPU29°C、可用RAM31769821184 bytes。这是停训后的空闲采样，不能当作负载稳定证据。本轮未重启训练、解除保护、修改预算/阈值或操作真机。

已核验的新启动与恢复链路：

- 第1次 `E007_long_01_s2000`：2000→2675持久模型；18:15:27末次health后整机重启，非正常退出，未留下温度保护退出原因。
- 系统启动历史还有18:16:00和18:16:48两次极短boot，随后18:17:20启动。18:27:21控制器登记第2次 `E007_long_02_s2675`，从2675准备恢复；末次采样18:27:37，之后18:28:11为新boot，未保存更高更新。
- 第3次 `E007_long_03_s2675`：18:38:12登记，从2675成功加载Actor/Critic和优化器（两个网络与源一致、optimizer_steps=[3000,53500]、学习率2e-5），保存2700；末次health18:39:32，18:40:03再次新boot。
- 18:40:13控制器校验到2700，因3次预算已用完返回attempt_limit/exit2，未启动第4次。state中第3attempt仍标running是重启前持久字段；结合已变化boot、总状态blocked与journal确认该进程已中断，不能将此字段当成实时进程。

三个run保护采样：333/4/17条，CPU最高76/67/67°C，GPU最高44/35/37°C，最少可用RAM24575004672/25802792960/25105670144 bytes；未出现CPU90°C连续两次触发。不能据5秒采样排除瞬时异常或硬件问题，但本轮没有R01那样的temperature_stop证据。前一boot完整内核日志搜索未找到NVRM Xid、OOM-kill、kernel panic、明确machine-check/critical thermal线索；没有日志不证明硬件正常，也不能断言电源损坏。电源、CPU散热、主板/内存、驱动仍是待诊断方向。

2700模型独立SHA/CPU重载/网络与优化器有限性/计数通过：`data/training/a3_20261009/E007_long_03_s2675/model_step_002700.pt`，402081749 bytes，SHA `3b78444903cf347c0cee14e9536801745c395a387facf8e54cd669a9945e8e84`，LOCAL_ONLY；未做该模型效果评测。故障诊断与验证日志 `logs/a3_watch_20261009/manual_stop_20261009T1227Z/`，清单 [R02故障记录](../data/manifests/a3_R02_stop_20261009.json)。

监测覆盖缺口：18:14的读取曾观察2650，但尚未完成落盘；18:32至20:17多次heartbeat消息在聊天中存在，本轮未发现对应已完成检查记录，不能补写为逐次巡检成功。此次20:28核验从持久训练记录和journal恢复实际事件，保留准确检查时间。

下一步优先现场核对电源型号/功率与接线、CPU散热风扇/泵和BIOS硬件告警；服务器软件诊断可继续只读收集。若再训练，须显式记录新的受控任务和启动预算、从2700恢复，不能清空R02账本或自动无限重试。当前10分钟巡检继续观察新变化至原有效期，未把任务标为完成。

## 目标与授权

用户明确要求“测试如果不通过，继续按照官方的方法训几个小时，比如训到10000次”。已有MuJoCo E01中20/20触发跌倒，补训练环境Isaac的selected20原生评测时遇缺失指标依赖，尚不能区分策略不足和sim2sim差异。依据已确认的MuJoCo失败，登记新R02，从已验证2000完整checkpoint恢复到累计10000（新增8000次更新），不重新随机初始化。延长训练是待验证假设，不把缺依赖或更新增加当成效果结论。

这轮是延长训练预算的工程尝试，保留官方PPO、reward、64环境、selected20动作和学习率恢复逻辑。训练内表现不代表动作泛化/真机部署。Q1副本和每日研究任务不修改。

## 评测计划与判据

运行官方 `python -m gear_sonic.evaluation run`，20环境、a3_fast、同一20个30Hz motionlib、官方tracking/eval termination，固定已有checkpoint。要求输出完整20动作并通过官方validate；如果成功率未达到全20完成，则原生跟踪未通过。Isaac与MuJoCo失败判据不同，数值不混用。程序异常/温度停止不是模型效果结论。

- E02：`--dry-run` 仍创建prepared_dataset目录，随后包装器拒绝重复输出路径，unit退出1；不是模型失败，保留该目录。
- E03：17:42加载policy成功并加载20动作；17:43在官方回调汇总时缺少 `smpl_sim.smpllib.smpl_eval`，未生成metrics或batch结果。Isaac异常后仍退出0，官方run_manifest写complete；这是执行状态，不能当成有效评测。包装器已补强制官方validate检查，避免以后误报成功。当前原生策略效果为UNKNOWN。
- E01 MuJoCo：step2000在20动作中全部触发跌倒，中位2.40秒，作为本轮续训触发证据。

配置 `script/a3/isaac_eval.yaml`；包装 `script/a3/evaluate_isaac.py`，CPUQuota100%、线程1、最长15分钟、CPU90/GPU85连续两次保护，无自动重试。输出 `data/evaluation/a3_20261009/E03_isaac_step2000/`，日志 `logs/a3_evaluation_20261009/E03_isaac_step2000/`；unit `yuanqi-a3-isaac-eval-20261009-e03.service`。本机已检查路径未找到smpl_eval.py，依赖修复待做，不临时替换指标定义。

## 续训预算与版本

main起点 `1db5d568040d579cf60d05c491bbb2c617bc0f73`，vendor保持 `fe6868ba37034f89b912f0fb851bce19f120266d`。起点模型 `data/training/a3_20261009/E006_auto_03_s1800/model_step_002000.pt`，SHA `c19f3e760245a611918dd91552ef181ba07cb6d00962d22a84dd4557f8ac41fc`，402081495 bytes。

已另建 `data/training/a3_20261009/autoresume_control_R02/`，不修改R01已complete的账本。新预算最多3次启动、2次无进展停止、24小时有效（2026-10-10 17:47:35北京时间到期，以job.json的expires_at为准），CPUQuota100%、原温度阈值保持；每25次持久保存，两份最近模型和500步里程碑保留。真实整机重启可有限恢复；温度/普通异常blocked不自动解锁，不承诺硬件故障已修复。

按最近200次续训约8分24秒的速度，新增8000更新约5–6小时，仅供规划；速度/保存开销/重启可能变化。以累计10000为结束条件，不保证几小时后策略通过。到达目标后重新评测同动作，并保留2000基线。

## 命令与产物

```bash
cd /media/yu/FAFF-E9771/YUANQI
data/environments/a3-sonic/bin/python -m script.a3.evaluate_isaac \
--checkpoint data/training/a3_20261009/E006_auto_03_s1800/model_step_002000.pt \
--output data/evaluation/a3_20261009/E04_isaac_step2000 \
--logs logs/a3_evaluation_20261009/E04_isaac_step2000
```

应通过受限systemd unit执行，实际命令和退出结果写包装execution.json。E04仅预留新ID，未执行。新任务配置入口和隔离账本已实现，35项恢复/保存检查通过，bash语法及systemd unit校验通过。续训运行后补service、ledger、加载回执、新checkpoint和温度证据。

本轮实际初始化（只能对新job执行一次，不用于重置预算）：

```bash
data/environments/a3-sonic/bin/python -m script.a3.autoresume \
data/training/a3_20261009/autoresume_control_R02 --init \
--source-dir data/training/a3_20261009/E006_auto_03_s1800 \
--target-step 10000 --run-prefix E007_long
```

`script/a3/yuanqi-a3-longtrain.service`已安装至用户systemd并enable，HOME boot guard已同步；服务环境选择R02，不影响R01默认配置。启动后恢复读取命令：

```bash
systemctl --user show yuanqi-a3-longtrain.service \
-p ActiveState -p SubState -p Result -p CPUQuotaPerSecUSec
cat data/training/a3_20261009/autoresume_control_R02/state.json
cat logs/a3_training_20261009/E007_long_01_s2000/resume_loaded.json
tail -1 logs/a3_training_20261009/E007_long_01_s2000/health.jsonl
```

17:47:39登记第1次 `E007_long_01_s2000`，加载回执确认global_step=2000，Actor/Critic与源完全一致，optimizer_steps=[3000,40000]、两个组学习率2e-5恢复。日志明确 `target=10000 remaining_updates=8000`。仿真episode重新reset，不保证逐轨迹无缝续接。

17:50独立selector已验证累计2050的模型大小、SHA、网络/优化器有限性与计数：402081749 bytes，SHA `c7ea0e998802d2b3cecaebde34b934ad68932826bba8d4430ba2ed75909af165`。非500倍数模型可能按保留策略清理，本条作为当时启动成功证据。日志 `logs/a3_longtrain_20261009/checkpoint_validation.log`；独立主机快照 `logs/a3_longtrain_20261009/launch_verified/snapshot.jsonl`，CPU69°C、GPU40°C，可用RAM25122996224 bytes，boot未变。service active/running/enabled，CPUQuotaPerSecUSec=1s（100%，并非CPU绑核）。短时正常不能证明数小时稳定或硬件根因已解决。

训练产物 `data/training/a3_20261009/E007_long_01_s2000/`，训练日志 `logs/a3_training_20261009/E007_long_01_s2000/`；账本记录的是起点与运行状态，运行中真实进度查checkpoint sidecar。原生E03官方validate返回1，缺失metrics_eval.json，日志保留；退出0的假成功已经排除。

## 保存与下一步

阶段README、报告索引/current state和小型R02清单已保存，随本次scoped代码提交审计发布main，推送结果须独立核验；提交号由Git历史提供。大模型留服务器，仍LOCAL_ONLY，未验证网盘备份。

原heartbeat `yuanqi-a3`已更新为R02、新服务及新有效期，工具返回ACTIVE，automation.toml回读核验10分钟间隔；每日研究任务保持原配置。巡检不再次启动或解锁，累计10000完成或到期后最终核验并暂停。恢复先读本README与R02 job/state，不复用R01预算。

累计10000后修复原生指标依赖、重评2000与10000，并做同条件MuJoCo对照；运行中不并行抢占GPU评测。不承诺训练更久必然解决sim2sim失败；原生结果缺失仍是需要补齐的诊断项。
