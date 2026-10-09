# A3 循环巡检与 CPU 温度保护记录

记录日期：2026-10-09（北京时间）。最新：15:18:29触发的巡检中发现第三次续训正常完成；15:20:33 ledger complete/2000，15:22独立校验最终模型通过，按范围结束本巡检。详见[最终续训](README_A3_CPU_RESUME_20261009.md)。下方保留此前保护停止和恢复历史，不作为最新运行状态。

## 最终巡检（15:22）

第三次run正常退出0、systemd inactive/success，boot未变，最终2000模型402081495 bytes，SHA `c19f3e760245a611918dd91552ef181ba07cb6d00962d22a84dd4557f8ac41fc`，独立重载/有限性核验通过；13项训练标量均有限。保护采样CPU最高71°C，独立巡检曾73°C，无新温度保护。模型LOCAL_ONLY，效果评测未做。日志 `logs/a3_watch_20261009/scheduled_20261009T071829Z/`；最终记录保存后已暂停原heartbeat，工具返回PAUSED、automation.toml回读确认；每日研究yuanqi仍ACTIVE。未启动新训练或评测。

## 15:14 巡检范围更新

原heartbeat已更新为观察 `E006_auto_03_s1800`，CPUQuota100%，保留原温度保护、3次累计预算与有效期。当前服务运行、CPU59°C/GPU39°C，属于短时观察；不宣称长期稳定。巡检不再次解除blocked、不重新启动、不重置预算。完成2000或到期后记录结果并暂停本巡检，保留每日研究任务。

## 目标与范围

用户要求 loop 检查。为当前 R01 任务配置当前聊天的 heartbeat `yuanqi-a3`，名称「YUANQI A3 训练循环巡检」，每10分钟检查一次。每日研究任务 `yuanqi` 保留；本轮巡检不启动新训练、不解除 blocked、不提高温度阈值、不重启机器、不改功率 / 驱动 / BIOS、不操作真机。

配置已通过 app 工具创建，回读 `~/.codex/automations/yuanqi-a3/automation.toml` 核验 ACTIVE、10分钟间隔和当前 thread id。首次创建缺少 destination 被拒绝；补上 destination=thread 后成功，未创建重复任务。这里的loop是持久定时巡检，实际触发已在下节核验。

## 定时触发与静默检查

- 首次heartbeat在2026-10-09 14:04:59.251北京时间到达，14:06:39.022444完成现场只读采样：boot不变、service failed、ledger blocked/temperature_stop、2次启动、最新sidecar1800；CPU49°C、GPU30°C/0%、可用RAM31260962816 bytes。未重载模型、启动训练或解除保护。日志 `logs/a3_watch_20261009/scheduled_20261009T060459Z/` 和同名summary.json。
- 14:14:59.279也收到触发，但前次记录尚未完成；没有单独采样，不能将其写成独立巡检成功。
- 14:24:59.296触发后，14:25:42.695819再次核验：同boot、相同blocked与1800模型、无新journal事件；CPU51°C、GPU31°C/25.52W/0%、RAM31261982720 bytes。日志 `logs/a3_watch_20261009/scheduled_20261009T062459Z/` 与同名summary.json。
- 本轮确认调度可触发，未发现新的故障/恢复/进度，保留已通知事件标记，不重复temperature_stop通知。有效期未到，继续只读巡检。关键调度核验记录按main发布。

## 首轮发现与关键证据

- 前一已核验发布为 main `474a04ae27441d3e2c8aacb7a1138d25f9bdac7a`，本轮开始服务器工作区干净。
- 主机发生新启动，who -b 为 10/9 13:02，boot_id 从 `48038245-8c38-4b31-ba43-9b98b150654f` 变为 `91918243-d1e8-4a3b-a49b-85b19b112134`。重启来源未确认，本代理未触发。
- systemd 开机运行恢复控制器，13:12:07 登记第2次训练 `E006_auto_02_s950`，从第1次的 step950 启动。`resume_loaded.json` 证实 loaded_global_step=950、Actor / Critic 与源一致、optimizer_steps=[3000,19000]，学习率均2e-5。这提供真实重启后的自动恢复证据，不只是模拟测试。
- 13:47:27.335939 与13:47:32.358223，CPU `x86_pkg_temp` 两次均90°C；GPU分别43°C和42°C、功耗128.48W和130.83W。控制器触发 temperature_stop，向训练进程发SIGTERM，returncode=-15，13:47:33写入 blocked。当前 service failed / exit-code 是保护停止结果，不是新的 Python 异常或再次整机断电。
- 已完成 **950→1800** 的续训。第1800模型为402081495 bytes，SHA-256 `c02949f78a221c96e20e2576a74d61cdbd51a2ee695bf2539bfe83eeee71c8e5`；控制器和本轮独立 selector 均通过大小 / SHA / CPU重载 / 有限性核验。
- 13:57:28 最新诊断：同一boot，CPU50°C，GPU31°C / 22.69W / 0% / 96MiB，RAM可用约29GiB。降温后仍保持 blocked，不以空闲温度回落自动认定负载稳定。

结论：自动恢复链路已在一次真实重启后生效，本次明确停训原因是我们设置的 CPU 温度保护。**这不证明之前每次重启都由 CPU 过热导致，也不证明电源 / 散热器已损坏。** 下一次现场检查优先增加 CPU 风扇 / 泵、散热器安装与 BIOS 温度核验，再决定是否做有记录的减负载对照。当前不解除保护。

## 循环检查内容与通知

每轮先核验 SSH / boot_id / uptime，再查 service、job与跨重启的attempt预算、登记run的最新checkpoint sidecar、最近health / 退出原因 / journal以及实时GPU、CPU、RAM。日常读小型记录，避免每10分钟重复加载大模型。新恢复或最终checkpoint再做独立校验。

去重清单 `data/manifests/a3_watch_state_20261009.json` 保存当前观察与已通知事件。只通知新故障 / 恢复 / 每500更新里程碑 / 完成 / 用户需要处理的新问题；同一temperature_stop不反复打扰。SSH失败记UNKNOWN，不视为主机重启。日志放 `logs/a3_watch_20261009/`，重要变化及时写README并按main发布规范同步。

任务目标2000，当前2/3次启动预算，仍差200更新；blocked先核实散热 / 负载，不清零预算。巡检范围截至任务完成或 **2026-10-10 12:47:08** 到期；届时记录结果并暂停本巡检，不改变每日研究简报。延长或新训练按用户后续指令登记。

## 版本、命令与产物

代码未改变，恢复服务仍使用已发布代码。训练位置 `/media/yu/FAFF-E9771/YUANQI`，Q1独立目录不修改。

```bash
ssh hp3090
cd /media/yu/FAFF-E9771/YUANQI
systemctl --user show yuanqi-a3-autoresume.service -p ActiveState -p SubState -p Result
cat data/training/a3_20261009/autoresume_control_R01/state.json
tail -8 logs/a3_training_20261009/E006_auto_02_s950/health.jsonl
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
data/environments/a3-sonic/bin/python -m script.a3.select_resume_checkpoint \
data/training/a3_20261009/E006_auto_02_s950
```

模型 `data/training/a3_20261009/E006_auto_02_s950/model_step_001800.pt`；原始日志 `logs/a3_training_20261009/E006_auto_02_s950/`。本轮快照 `logs/a3_watch_20261009/initial_diagnostics/snapshot.jsonl` 与 `initial_summary.json`；重载核验 `checkpoint_validation.log` 与 `verified_checkpoint.txt`。

## 保存、同步与下一步

本README、恢复README更正、报告索引、当前状态和小型去重清单按main提交；发布须审计并比对远端，实际提交号由Git历史提供。模型仍LOCAL_ONLY，自动恢复不是网盘备份。

下一轮按去重清单检查是否出现新事件；定时触发已验证。下一次训练前优先核实CPU散热与温度读数；未恢复训练，也未把1800更新当作2000完成或策略效果验收。
