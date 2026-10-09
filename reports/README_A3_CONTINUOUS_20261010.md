# A3 R03 持续续训至累计10000

记录日期：2026-10-10，北京时间。阶段：R03 / E008_cont。最新：用户要求先停核对官方模型，01:09停止、service disabled、巡检PAUSED；最近2950独立校验通过，目标10000未完成。见[官方模型与暂停](README_A3_PRETRAINED_20261010.md)，下方保留启动/巡检历史。

## 目标与授权

用户明确要求“重启训练吧，以后不要3次启动，就一直训练下去吧”。从R02已保存的2700恢复至原目标累计10000，不重置网络或优化器。新任务取消累计启动次数和24小时到期限制，意外重启后自动恢复；达到10000后结束，不无限增加训练目标。

R01/R02历史账本保留。CPUQuota100%、CPU90°C/GPU85°C连续两次的5秒采样保护、300秒恢复冷却、连续两次无保存进度停止、普通训练异常停止、GPU占用检查、20GiB磁盘余量与单任务锁保持。取消启动上限不等于解除温度或错误blocked。

## 完成内容与关键决定

- `script/a3/autoresume.py` 增加仅新任务初始化可用的 `--until-target`；以JSON null表示无启动次数限制与无到期时间，旧任务默认3次/24小时不变。
- 服务 `yuanqi-a3-longtrain.service` 指向新R03。当前官方PPO/reward、64环境、selected20动作和恢复学习率保持；仅调整恢复任务的运行限制。
- 2700检查点大小/SHA/CPU重载/网络与优化器有限性及计数独立验证通过。源为 `data/training/a3_20261009/E007_long_03_s2675/model_step_002700.pt`，402081749 bytes，SHA `3b78444903cf347c0cee14e9536801745c395a387facf8e54cd669a9945e8e84`，LOCAL_ONLY。
- 恢复与持久保存测试39项通过，覆盖无上限多次有进度恢复、目标完成、无进度停止、仅初始化可改变配置、无到期任务仍触发温度保护；systemd配置校验通过。

## 版本、命令与证据

服务器 `/media/yu/FAFF-E9771/YUANQI`，分支main，修改起点 `4810401e4e34a456456a304fd886e50b641d7950`；官方vendor仍固定 `fe6868ba37034f89b912f0fb851bce19f120266d`。

```bash
data/environments/a3-sonic/bin/python -m script.a3.autoresume \
data/training/a3_20261009/autoresume_control_R03 --init \
--source-dir data/training/a3_20261009/E007_long_03_s2675 \
--target-step 10000 --run-prefix E008_cont --until-target
```

初始化已执行；该命令只用于新任务一次，不用于重置已有账本。R03 `job.json` / `state.json` 在 `data/training/a3_20261009/autoresume_control_R03/`。训练产物仍使用既有 `data/training/a3_20261009/E008_cont_*`，执行日志 `logs/a3_training_20261009/E008_cont_*`；日期目录沿用原训练数据集，无需移动大型模型。

校验与测试日志 `logs/a3_continuous_20261010/setup/`。R02 job SHA `c30c72050ca90aef9b5579315c4cad58e3d919a45304796ec2d9cc8eb4cb7581`，state SHA `f37da8ac92279154bcfdeb696189fca6db3ab7c16319daafff6b1b5b4ae26733`，发布和启动后复核未变。

## 实际结果与边界

R03首次定时巡检：2026-10-10 00:59:52.193触发、01:00:52.610602实际采样。服务active/running/enabled、指向R03、CPUQuota100%，boot和run未变；最新2900 sidecar与文件大小402081495匹配，本轮未独立重哈希/重载，最近独立核验为启动时2725。CPU73°C/GPU42°C、可用RAM25223151616 bytes，近期health持续更新、journal无新退出，未见温度保护。尚未到3000通知节点，无新故障/恢复，保留通知去重键。日志 `logs/a3_watch_20261010/scheduled_20261009T165952Z/`；本轮仅巡检，没有启动或恢复训练、配置变更或评测。该条核验更新R03后定时巡检实际触发。

00:52:12服务启动，00:52:15登记首次 `E008_cont_01_s2700`，run和health的boot均为 `a95a89b9-df91-4c13-b609-86273fbd847d`。加载回执确认global_step2700、Actor/Critic与源一致，optimizer_steps=[3000,54000]、两组学习率2e-5；仿真episode重新reset，非逐轨迹无缝恢复。

00:53独立selector已验证新2725模型的大小/SHA/CPU重载/网络与优化器有限性及计数：402081749 bytes，SHA `7024a35ec2355cd1d87864412f9b37c22635a0b87de8ae5425a42388e1b1b82a`，LOCAL_ONLY。日志 `logs/a3_continuous_20261010/setup/new_checkpoint_validation.log`；加载回执 `logs/a3_training_20261009/E008_cont_01_s2700/resume_loaded.json`。非500倍数模型可能按保留策略清理，此项保留当时启动成功证据。

00:53:50独立健康采样CPU66°C、GPU37°C、可用RAM25209962496 bytes，服务active/running/enabled、CPUQuotaPerSecUSec=1s，未见新热停/重启。快照 `logs/a3_continuous_20261010/launch_verified/snapshot.jsonl`。这只证明当时运行和保存成功，不能证明数小时稳定。R02多次整机重启根因仍UNKNOWN，取消启动上限不是修复硬件。

MuJoCo step2000基线20/20跌倒；Isaac原生评测缺smpl_sim指标依赖、validate失败，效果UNKNOWN。R03更新数不证明效果通过，也未操作真机。所有大型模型仍LOCAL_ONLY，未验证网盘备份。

## 保存、同步与下一步

配置与初始README已发布 `287ee882944ef078744845acfd933b6f23af8477`，索引/current state已发布 `889546bf9aea6f1af9f222b3c4a56e02ef5f214b`，各次远端main均核验一致。启动结果、工件/巡检清单和长期偏好随本次scoped审计发布；具体提交由Git历史提供。00:55核验R02 job/state SHA仍与启动前一致。

原 `yuanqi-a3` 已更新R03，工具返回ACTIVE，automation.toml回读确认10分钟间隔、R03范围、无到期限制，完成10000后暂停；每日研究 `yuanqi` 未修改。未来巡检日志 `logs/a3_watch_20261010/`，去重仍使用 `data/manifests/a3_watch_state_20261009.json`。此前00:43触发已保存00:44:55原始快照但被用户中断，未完成去重清单发布，不冒充完整巡检。

恢复先读本文件、R03 job/state和当前service。巡检不主动解除blocked或重复启动；完成10000后独立校验最终模型、保存结果并暂停巡检。原R02到期不适用于新R03。
