# A3 单次训练重启观察 E22/R10

记录日期：2026-10-11。状态：E22/R10已实际启动并确认重启，按用户要求停止；有step1保存点，无step2，不自动恢复。

## 目标与授权

用已验证A3训练入口进行一个独立短试跑，观察初始化、仿真采样、参数更新及保存阶段。目标是记录实际训练负载是否触发重启；本轮不是效果优化验收，不自动长训、评测或操作真机。此前E01负载诊断14通过、1温度保护停止，内存/GPU/小网络继续搁置。

Intel官方处理器温度说明区分降频与自动关机：降频不足以维持安全温度时可自动关机。官方说明不证明本机历史重启由CPU温度导致，也没有证明90摄氏度等于本机硬件关机阈值；前次90摄氏度是项目保护阈值。来源：https://www.intel.com/content/www/us/en/support/articles/000005597/processors.html （2026-10-11实际读取核验）。本轮仍记录boot作为重启证据，SSH失败只记UNKNOWN。

## 固定设置与决定

- 独立E22/R10，从SHA已核验官方035_step200000 PT初始化；训练计数和优化器从0开始。已有E21训练已完成，此次不重复其监督适配器训练、不恢复历史坏模型。
- 复用已验证R09配置：16环境、4 minibatches、epochs1、LR尺度0.1、seed0，目标2次更新，每次保存。R09整体未优于官方；这里只使用其保守配置诊断训练运行。
- 新入口`script/a3/reboot_trial.py`复用`corrected_trial`，独立OUTPUT/LOGS/TRIAL。为兼容既有checkpoint归属校验，新R10保存到`data/training/a3_finetune_20261010/R10_001_s0/`；日期目录为原入口兼容路径，不表示本次在10月10日执行。旧R01-R09/E10/E20/E21账本不改。
- 标准库外层登记保存READY后只允许一次启动：进入RUNNING后，无论异常、重启或正常退出均不可重复；拒绝不同boot的READY。不是项目永久启动预算变更，只约束用户明确授权的这次试跑。
- 临时unit，Restart=no、RuntimeMaxSec15分钟、CPUQuota100%、MemoryMax12G，不enabled、不随开机启动；不恢复yuanqi-a3自动续训服务。用户CPU90/GPU85摄氏度连续两次5秒采样保护保持，增加4GiB可用RAM及缺失遥测/关键内核事件停止。

## 命令与路径

```bash
cd /media/yu/FAFF-E9771/YUANQI_A3
/usr/bin/python3 -m unittest script.a3.test_reboot_trial
/usr/bin/python3 -m script.a3.reboot_trial register
# 注册只能执行一次；当前已注册，不能重复。
systemd-run --user --unit=yuanqi-a3-reboot-trial-E22 \
  --property=Restart=no --property=CPUQuota=100% --property=MemoryMax=12G \
  --property=RuntimeMaxSec=15min \
  --property=WorkingDirectory=/media/yu/FAFF-E9771/YUANQI_A3 \
  --setenv=PYTHONPATH=/media/yu/FAFF-E9771/YUANQI_A3 \
  --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 \
  /media/yu/FAFF-E9771/YUANQI_A3/data/environments/a3-sonic/bin/python \
  -m script.a3.reboot_trial run
# 正常完成且同boot时才独立核验，不能通过此入口重新训练。
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  data/environments/a3-sonic/bin/python -m script.a3.reboot_trial verify
```

登记/外层状态：`data/experiments/a3_reboot_trial_20261011_E22/trial_job.json`、`launch_state.json`；训练详细账本：同目录`job.json`、`state.json`。监测：`logs/a3_reboot_trial_20261011_E22/guard.jsonl`、`R10_001_s0/health.jsonl`与`console.log`。preflight快照也在本日志目录。原始日志/模型LOCAL_ONLY，Mac只保存代码、报告和小型清单。

## 验证与恢复

2026-10-11北京时间14:47:39注册成功，初始boot cb282100-387e-400f-bff0-dab2de7282b8。3项无重复启动/boot隔离/温度保护契约通过，source402050931字节，SHA256 `9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d`。preflight14:43 CPU51摄氏度/GPU30摄氏度、GPU空闲；启动前再次核验温度。登记成功不是训练开始或通过。

确认新boot时只读取state、最后健康记录和保存点sidecar，标记中断并停止本轮，不CPU加载模型、不再启动训练。SSH不可达保存UNKNOWN与最后确认阶段，不能当作已重启；恢复SSH后只读核验boot。训练错误或温度保护同样停止；正常达到2则独立核验模型/优化器、8实际optimizer步和初始权重/计数，不追加200或10000，不做策略效果结论。

## 实际结果与阶段定位

临时服务在北京时间14:51:33启动，Restart=no、RuntimeMax15min、CPUQuotaPerSecUSec1s已实际回读。16环境Isaac场景创建、仿真启动和16动作加载均进入执行；`warm_start_loaded.json`确认官方policy/value与来源一致，global_step0、优化器条目0，初始重置已发生。不是只完成启动命令。

14:51:56.012333至14:51:56.046451，4条实际optimizer返回日志保存（第一轮4 minibatches），实际LR从2e-6至1.333333e-6。step0与step1文件及sidecar均存在，大小/SHA独立读取核对一致，step1 PT402076373字节、SHA `aa2df97688e73c94ac8f1cc2d78ad141be157b4fdd2fdb0b8ef37bd3b3bf0bec`，mtime14:51:56.651651。没有step2或第二轮optimizer日志。只能确认第一更新有保存证据，不能据缺失日志判断第二轮绝未开始。

14:53:09 SSH恢复后首次确认新boot `740972e7-c88e-4588-8fa3-a86e8ed6bc64`，uptime40.42秒；初始boot `cb282100-387e-400f-bff0-dab2de7282b8`。由UTC采样时间减uptime推算新boot约14:52:29，距离启动约55秒；这是开机开始估算，不是精确故障时刻。重启事件至少发生在第一轮保存后至14:53:09新boot确认之间。首次SSH超时先登记UNKNOWN，随后新boot证据才确认重启。

本轮仅5条完整健康采样，14:51:34至14:51:54，CPU最高60摄氏度、GPU最高34摄氏度，没有保存下来的阈值触发；最后GPU利用率3%、功率113.56W、显存3735MiB，可用RAM约23.99GiB。最后采样之后及断连期间遥测缺失，不能排除间隔内温度/功率峰值。此前四workerCPU90摄氏度保护停止与本次重启是两件不同事件。

本轮console日志含1686个零字节且可读正文止于动作加载；它比持久optimizer/sidecar证据更旧。故不能把console最后一行认定为故障调用点，也不能据此认定重启仍停在初始化。前一boot训练unit日志只有Started，所读内核段未见FATAL匹配，不等于不存在驱动/硬件问题。

| 项目 | 状态 | 范围 |
| --- | --- | --- |
| 启动、官方初始化和训练执行 | 通过，限定证据 | 官方载入回执、实际4 optimizer步 |
| step0/step1文件完整性 | 通过，大小/SHA | 未在重启后重载PT或检验数值有限性 |
| 2次更新短试跑 | 失败/中断 | step1后确认新boot，无step2 |
| 模型/优化器独立重载、效果评测 | 搁置 | 重启后按用户要求停止计算 |
| CPU负载E01后续内存/GPU/小网络 | 保持搁置 | 没有普通GPU网络对照，不宣称已排除 |

新外层`launch_state.json`已标记STOPPED_REBOOT，训练详细state标记blocked/CONFIRMED_NEW_BOOT_NO_RESUME；只修改新E22账本。服务在新boot inactive/dead，无训练进程或GPU计算进程；旧autoresume/longtrain仍disabled，两个heartbeat仍PAUSED。重启后只做日志、元数据和文件SHA读取，不重载模型、不再训练/评测。

## 判断与后续假设

用户怀疑“A3训练相关导致重启”，现有观察支持把A3执行链路列为优先关联线索：固定两小时空闲无新boot，单/双CPU整数档完成无新boot，重新启动A3约一分钟后出现新boot。该对照没有隔离GPU矩阵、小CNN、Isaac本体、policy前向和PPO更新，且历史重启也涉及CPU标签准备及GPU推理；不能直接认定A3算法错误或排除驱动、硬件/负载条件。

当前可定位到“第一轮更新及保存后至下一持久保存前”，具体出错操作UNKNOWN。以后若用户重新授权，应分别登记普通PyTorch GPU计算、Isaac场景/步进、仅policy前向及完整更新对照，每次只改变一项、首故障停止，避免反复重启完整训练；本轮没有执行这些分支。CPU散热仍是另一已记录问题，不因此取消温度保护。

最终证据入口：`data/manifests/a3_reboot_trial_result_20261011_E22.json`，包含boot转换、时间、4optimizer行、采样、文件SHA及检查范围；原始健康/console/训练日志与两个PT留服务器LOCAL_ONLY。该清单明确`cpu_reload=false`，不将SHA完整性当作模型有限性/效果验收。

## 同步

每个重要阶段保存本README、索引/current state及小型清单，审计后scoped提交推送main并核验远端。保留并行修改。原hp3090与yuanqi-a3 heartbeat仍PAUSED，每日研究yuanqi保留。
