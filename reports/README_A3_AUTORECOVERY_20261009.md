# A3 远程故障排查与有限自动续训

记录日期：2026-10-09（北京时间）。实验 / 恢复编号：R01。最新状态（13:57）：真实重启后的自动950→1800续训已验证，13:47因CPU连续90°C保护停止，当前blocked。原始重启原因未定，10分钟巡检已配置，见[巡检与热保护](README_A3_LOOP_WATCH_20261009.md)。下文12:50–13:03为历史快照。

## 目标与当前证据

用户不在现场，现场协助时间有限，希望区分设备 / 系统 / 训练问题，并在无法立即解决时通过 checkpoint 自动续训。本任务继续已授权的 A3 仿真训练，不操作真机，不主动重启主机，不改驱动、BIOS 或供电设置。

最新检查：主机启动时间 2026-10-09 11:26:18，12:32 GPU 空闲，30°C / 22.58W / 96MiB。E005 中断的第 600 次更新模型已校验大小、SHA-256、CPU 重载和网络 / 优化器有限性，仍保留 575 与 500。对应路径及证据见[续训记录](README_A3_RESUME_20261009.md)。

此前 12:19 检查中，上一启动周期日志末尾为 11:23:42；未查到明确 OOM / Killed process、NVIDIA Xid、panic、MCE / Hardware Error、热保护、NVMe / I/O 错误或正常 shutdown。日志无这些记录不能排除对应原因，突发掉电 / 重启可能来不及刷盘。证据支持“整机重启打断训练”，不能确认“电源已坏”。

主机 Kernel 6.8.0-138-generic，NVIDIA 595.84，3090 当前 / 默认功率限值 370W；CPU x86_pkg_temp 空闲约 40°C，31GiB RAM、无 swap。持久 journal 已存在；pstore 读取受权限限制，归档目录为空；sudo 非交互授权不可用。没有据此改驱动、功率、安装诊断工具或读取认证材料。

## 接下来的工作 pipeline

```text
现有 E005 step600
  -> 有限恢复至累计 2000 更新（学习 / 从零对照；不是成熟策略）
  -> 保存、重载、独立仿真评测，确认有效训练进度

官方 A3 035 预训练 PT + 配套配置 / 精确资产核验
  -> B0 不训练：建立动作跟踪基线
  -> B1 普通微调：先适配我们的目标动作
  -> B2 少参数 / LoRA 对照：仅在 B0/B1 成立后考虑
  -> 衣服 / FGP / SMPL 到 A3 参考，处理坐标、关节及实时未来帧
  -> 头环手 / 腕和 SLAM 融合
  -> sim2sim、执行器 / 延迟验证
  -> 满足型号与安全接口条件后安排真机验证
```

训练与主机稳定性并行推进，诊断不以训练回报代替。B0/B1/B2 具体设置见[迁移方法评估](README_A3_TRANSFER_20261009.md)。官方 PT 当前未下载，Hugging Face 查询曾超时；慢传输按用户指定提供文件名、大小、源 / 目标路径，由用户迁移后核验。现有 E005 没有预训练的历史能力。

## 远程排查顺序与判断边界

| 层次 | 最有区分度的证据 / 对照 | 能支持什么判断 |
| --- | --- | --- |
| 网络 | SSH 失联时恢复后比较 boot_id / uptime、训练日志与更新数 | boot 未变且训练继续更偏网络问题；单次 SSH 超时不能证明断电 |
| 训练进程 | 完整 traceback、退出码、GPU / RAM、loss / NaN、模型校验 | Python 错误而主机正常，更偏配置、代码、数据或训练数值；先修错误，不自动反复跑 |
| 系统 / 驱动 | 前一 boot 的 OOM、Xid、panic、MCE、I/O 等事件与时间关联 | 缩小到内存、GPU / driver、内核或存储；Xid 本身不一定能唯一定位 |
| 设备 / 供电 | 突然结束日志、boot 变化、负载 / 温度 / 功耗；现场检查或替换验证 | 电源、GPU、主板、内存等成为候选，日志缺失不是电源故障证明 |

先保存监控数据和持久 journal。后续按一次只改一个因素的顺序做有时长上限的对照：①普通 GPU 计算，②同环境仅仿真 / 固定策略推理不更新，③相同仿真配置训练更新。若第一项也使整机重启，则不只是 PPO 的业务逻辑；若只有第三项失败，也还要排除训练新增的资源压力、CUDA / driver 问题。对照尚未执行，短测通过不证明长期稳定。不能把不同任务负载差异直接解释成唯一原因。

可选受控降功率对照：在有管理员权限和明确设置记录时，将 3090 power limit 从当前 370W 降为约 250–280W，保持其他条件不变，记录同负载重启频率。若变稳定，仅说明与负载 / 功耗有关，不证明电源是唯一原因。本次无非交互 sudo 权限，未调整。

现场一次访问优先提供以下事实：电源品牌、型号、额定功率、使用年限；GPU 供电插头和独立线缆 / 转接线照片；是否经插排 / UPS、其他大功率负载；CPU / GPU / RAM 是否超频；黑屏后是自动重启、关机待人工开机，还是有人按了电源。断电后再由合适人员检查接头，不远程指导带电插拔。需要时由维修人员做电源替换、内存 / GPU / 主板对照。需要恢复供电后自动开机时，现场核对 BIOS 对应功能；软件不能给完全断电的主机开机。

## 恢复机制与限制

- 入口 `script/a3/autoresume.py`；读取 `data/training/a3_20261009/autoresume_control_R01/job.json` 与原子刷盘的 `state.json`。
- 源只允许 E005 和本恢复任务登记的 run 目录；优先最高通过大小 / SHA / 反序列化 / 有限性校验的 checkpoint，损坏则退回有效旧模型。
- num_envs 固定 64，vendor 固定 `fe6868ba37034f89b912f0fb851bce19f120266d`，目标累计 2000；延续 Actor、Critic、optimizer、计数和 RNG。PhysX episode 重置，不是逐位连续轨迹。
- 最多 **3 次训练启动**，计数存在数据盘，跨重启保留；连续 **2 次没有保存进度**停止。有效期 **24 小时**，运行中也检查到期。修改配置会被拒绝；缺失 / 损坏 ledger 不自动清零。后续新预算需要明确登记新任务。
- 每 25 次更新持久保存，保留两份有效模型及 500 步里程碑。断电最多损失最近未保存进度；刷盘校验减少损坏风险，不能保证介质故障零损失。
- 开始前检查 GPU 是否有其他计算任务、磁盘至少 20GiB、可读温度及版本。双 supervisor 与 A3 wrapper 文件锁防本项目重复训练；其他工程不共享此锁，不能保证其他用户不在检查后启动 GPU 作业。
- 开机 / 上次尝试后至少 5 分钟冷却；其他 GPU 任务占用则等待而不消耗训练启动预算。
- 约每 5 秒采样温度、GPU 功耗 / 利用率 / 显存、RAM、load、boot_id，追加 JSONL 并 fsync。GPU ≥85°C 或 CPU ≥90°C 连续两次采样停止；GPU 查询失败、普通 Python / 训练错误停止。温度是监测到的读数，不覆盖所有未暴露的元件温度。
- SIGKILL / exit137 / 文件锁冲突可有限重试；整机重启后通过用户 systemd 服务恢复。监控异常清理子进程，避免监督器退出后残留训练。
- service 为 `yuanqi-a3-autoresume.service`；RestartPreventExitStatus=2 阻止已识别错误循环；总启动预算由 ledger 控制。RAM cgroup 软限16GiB / 硬限20GiB，CPU quota300%，实际支持情况以部署核验为准。
- 挂载等待脚本放在 `/home/yu/.local/lib/yuanqi/`，数据盘未就绪时最多等20分钟。该 ext4 盘已有开机 fstab，用户 `Linger=yes` 已查到；没有人为断电 / 重启来测试开机恢复。

自动续训能减少无人值守期间的进度损失，但不能修复设备、无限开关机或保证论文对照公平。重启次数、损失进度、墙钟时间应进入实验记录。若频繁重启或无进展，任务停止后优先现场排查或换稳定机器。

## 部署与验证记录

服务器 A3 环境中最终 **28 项 CPU 检查通过**（20 项恢复机制 + 8 项 checkpoint），覆盖跨重启预算、连续无进展、到期、缺失 / 损坏 ledger、配置变更、拒绝重复初始化、双 supervisor、损坏 checkpoint 回退、限定实验来源、环境数量、正常错误停止、有限重试及真实子进程清理。最后补充控制器自身错误也把有效 ledger 标为 blocked 并保留预算的检查；此补强已安装 / 测试，当前运行的 supervisor 先于此小修改启动，将在下一次启动使用，不重启当前训练。bash 语法和 systemd unit 检查通过；诊断快照已落盘。首次在小文件复制尚未结束时发起测试，出现测试模块未找到；保存 `initial_transfer_race.log`，完整传输后重跑通过，这不是训练故障。

测试后报告传输 / 状态 SSH 两次连接超时；12:44 重试恢复，uptime 1:17 对应原 11:26 启动，boot_id `48038245-8c38-4b31-ba43-9b98b150654f`。此时未启动新训练，ping 三次响应约 336–1398ms；这次失联没有新的主机启动证据，应与此前整机重启分开记录。

12:47:09 已安装并启用 `yuanqi-a3-autoresume.service`，ActiveState=active、SubState=running、UnitFileState=enabled；Linger=yes，RAM 与 CPU 配额已从 systemctl show 核验。12:47:12 ledger 登记第一次启动 `E006_auto_01_s600`，选择有效 step600，源未被拒绝，GPU 出现新工作负载。

12:50 核验 `resume_loaded.json`：loaded_global_step=600，Actor / Critic 与源完全一致，optimizer_steps=[3000,12000]，两组学习率均 2e-5；仿真 episode 明确重置。selector 在新 run 内通过 step650 的大小 / SHA / CPU 重载 / 有限性检查，说明已产生新学习进度。service 仍 active/running，boot_id 未变。04:50:03 UTC 采样 GPU 39°C / 128.01W / 4116MiB、CPU 67°C、MemAvailable 约23GiB，这是瞬时读数，不是峰值或稳定性结论。

任务有效期到 **2026-10-10 12:47:08（北京时间）**，正常情况下到达2000会提前完成；总预算3次启动，当前已登记1次。模型650会按两份有效模型 / 500里程碑的既有规则被后续有效更新替代，恢复时重新选择最新有效模型。

12:52 训练指标快照到 step712，所选13项标量均有限；reward 0.857→1.334，body_pos error 0.0971→0.1093。两端数据来自不同训练采样，不能当作固定条件对照，reward 上升也不能直接认定跟踪变好。日志 `logs/a3_autoresume_20261009/metrics_snapshot.json`，未据此更改奖励或追加算法。

截至12:52尚未验证真实重启后的服务自动运行；后续首轮巡检已确认一次真实boot后的950→1800自动恢复。物理断电过程、长期稳定性、累计2000完成与固定条件策略效果仍未验收。本代理没有主动断电 / 重启验证，详见[13:57最新记录](README_A3_LOOP_WATCH_20261009.md)。

预定控制命令（安装后生效）：

```bash
ssh hp3090
cd /media/yu/FAFF-E9771/YUANQI
systemctl --user status yuanqi-a3-autoresume.service --no-pager
journalctl --user -u yuanqi-a3-autoresume.service -n 30 --no-pager
cat data/training/a3_20261009/autoresume_control_R01/state.json

# 停止当前训练并取消下次开机启动
systemctl --user disable --now yuanqi-a3-autoresume.service
```

复现测试：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
data/environments/a3-sonic/bin/python -m unittest \
script.a3.test_autoresume script.a3.test_checkpoint_store -v
bash -n script/a3/run_official_training.sh script/a3/boot_autoresume.sh
systemd-analyze --user verify script/a3/yuanqi-a3-autoresume.service
data/environments/a3-sonic/bin/python -m script.a3.host_health logs/a3_diagnostics_20261009
```

代码初始基线 main `17ce20a8f6462cf68e52bbc4eaa7aa0b4b29dff8`，vendor 不修改。新增脚本及测试、wrapper 文件锁是本轮改动，发布版本由 Git 历史查询。

## 产物、日志与同步

- 模型：服务器 `data/training/a3_20261009/E005_resume_154to2000/model_step_000600.pt`，402081559 bytes，SHA-256 `757e96edd2754f298ae4daae95d25f9da52a4c11364a1ea45371a7ae41473aa0`。
- 本次续训目录 `data/training/a3_20261009/E006_auto_01_s600`；后续同任务启动为 `E006_auto_<序号>_s<起点>`，每次独立目录，不覆盖 E005。
- 日志：`logs/a3_autoresume_20261009/`、`logs/a3_diagnostics_20261009/` 和每个新 run 的 `logs/a3_training_20261009/<run_id>/`。系统原始日志留服务器，不整段上传 Git。
- 本阶段代码 / 报告 / 小型清单已作为 `215ed9b140bc91e923dd813335af596a44cada69` 推送 main，2026-10-09 13:02:28 核验本机 HEAD 与 `refs/heads/main` 一致，暂存审计通过（11文件 / 66806字节）。发布回执为 `logs/session_records/a3_autorecovery_publication_20261009.json`，随后补存核验记录的提交号由 Git 历史查阅。Q1 工作副本未修改。模型备份仍 `LOCAL_ONLY`，不能把恢复机制当作云端备份。

13:03:38 再次由独立 selector 核验新 **step950** 的大小 / SHA / CPU 重载 / 有限性，服务仍 active / running；日志 `logs/a3_autoresume_20261009/final_checkpoint_validation.log`。650为此前检查点，继续工作须选择最新有效模型，不固定要求650或950仍存在。

## 下一步

当前先保持temperature_stop的blocked，按[循环巡检](README_A3_LOOP_WATCH_20261009.md)只读观察，现场核实CPU散热与温度读数。累计2000尚未完成，不以更新数直接认定站稳 / 动作跟踪。随后获取官方PT和B0；恢复时先读最新巡检记录、current state及本报告，重新核验service、boot_id与ledger。

原始技术参考：[NVIDIA Xid 文档](https://docs.nvidia.com/deploy/xid-errors/introduction.html)、[Linux ramoops](https://www.kernel.org/doc/html/latest/admin-guide/ramoops.html)、[systemd service 官方文档源码](https://github.com/systemd/systemd/blob/main/man/systemd.service.xml)。Xid 可来自硬件 / 软件 / 应用，ramoops 需要支持的保留内存环境，不保证完全断电后保留。
