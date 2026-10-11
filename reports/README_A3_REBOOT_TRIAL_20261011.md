# A3 单次训练重启观察 E22/R10

记录日期：2026-10-11。状态：已注册，尚未启动。用户在CPU负载保护后明确重新授权A3训练，并要求确认重启后先停止。

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

## 同步

每个重要阶段保存本README、索引/current state及小型清单，审计后scoped提交推送main并核验远端。保留并行修改。原hp3090与yuanqi-a3 heartbeat仍PAUSED，每日研究yuanqi保留。
