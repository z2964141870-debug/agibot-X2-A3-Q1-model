# A3 官方模型全链路执行

日期：2026-10-10，北京时间。独立工作根为 hp3090 的 `/media/yu/FAFF-E9771/YUANQI_A3`，main 分支。本阶段继续用户批准的官方 PT、基线、2→200 短微调、对照及导出计划。旧全链路记录和 R01/R02/R03 保留暂停。

## 已完成与范围

官方035发布包5文件已验证，模型 SHA `9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d`，CPU重载及网络/优化器有限性通过。固定HF revision和配置身份见 `data/manifests/a3_official_import_20261010.json`。本轮没有重新下载模型。

官方10策略步冒烟实际退出0，完整10步、数值有限、未跌倒；全29关节RMSE 0.050375 rad。只覆盖0.2秒，不作为动作稳定性验收。证据 `data/experiments/a3_fullchain_20261010/official_smoke/explicit_summary.json`，日志 `logs/a3_fullchain_20261010/official_smoke/`。CPUQuota100%执行。

新台账 `data/experiments/a3_fullchain_20261010/tasks.json` 独立初始化。按旧Git工件清单逐文件核验大小/SHA后复用人体参考、A3参考、运动学和MotionLib，见 `reused_history.json`。未复制旧训练账本。seed42整动作16训练/4留出与原固定分配一致。24个PKL格式/数值与26个CSV时长合同重新验证通过；120Hz官方CSV stride4，30Hz新参考stride1，转换后30Hz。动捕TEST_ONLY标识保持，IK/脚穿地质量仍失败，微调使用官方样例。

## 独立卡点

Isaac指标导入再次失败：`ModuleNotFoundError: No module named 'smpl_sim'`。评测需兼容 `smpl_sim.smpllib.smpl_eval` 和独立validate，暂时搁置，退出码不能代替指标。RKNN模块仍未安装，转换/包和板端推理搁置。此二项不阻止MuJoCo和短微调。动捕根轨迹、接触、个人标定与姿态质量待用户处理，结构测试不能证明高保真。

## 版本与复现

vendor固定 `fe6868ba37034f89b912f0fb851bce19f120266d`，Python为新目录的 `data/environments/a3-sonic/bin/python`，读取复用原环境。本轮boot `5723f6c9-951e-4866-b4dd-cf4663b8084d`；启动前旧服务inactive、GPU无计算作业。重启仅登记和恢复。

```bash
cd /media/yu/FAFF-E9771/YUANQI_A3
data/environments/a3-sonic/bin/python -m script.a3.fullchain_support download
data/environments/a3-sonic/bin/python -m script.a3.fullchain_support split
data/environments/a3-sonic/bin/python -m script.a3.prepare_official_stage
data/environments/a3-sonic/bin/python -m script.a3.reference_contract motionlib
data/environments/a3-sonic/bin/python -m script.a3.reference_contract validate
```

以上download入口实际只核验5个已有文件。新工件留服务器LOCAL_ONLY，未进行网盘往返校验。

## 下一步与保护

10:16服务器连接中断后恢复，boot变为 `3f33490f-5dea-483f-8f83-a08e53a04b01`。磁盘持久完整动作16个，剩余4个按manifest恢复，不依赖中断前终端打印。GPU无计算进程；按300秒冷却等待后继续GPU任务，不排查重启根因。微调尚未启动。固定16动作输入preflight已通过。

新增因果策略回放钩子，保留官方vendor不修改。编码器速度采用前向差分，最后0–180ms姿态窗口还需要200ms处的位置，因此名义180ms缓冲实际先预填充200ms，并按30Hz到达量化记录实际年龄。逐策略步核验插值及速度所需的最后原始样本已到达；指标针对延迟后的目标帧，启动等待与模拟到达另记。新增3项未来数据扰动/速度边界/指标目标检查及已有17项检查通过，共20项。

官方CPU ONNX已实际导出，100个MuJoCo真实策略输入对比通过：输入[1,1570]、输出[1,29]，最大动作误差2.8610e-6，满足atol=1e-4/rtol=1e-3。模型57673235字节，SHA `5fbb77310cbe63f1468f9f1cbbec8c056256b4806cf4c3a1b15ccba239f66122`。证据 `onnx_official/parity.json`、真实输入 `official_baseline/001_walk_front_slow.inputs.npz`。使用固定MuJoCo策略适配器及opset13 atanh等价表达，未声称原Isaac导出入口或板端通过。

开始官方完整selected20，逐动作可恢复且完整覆盖核验；然后独立R04从官方权重初始化，计数与优化器重置，16环境，2次冒烟后完整恢复至200，每10保存。无累计启动次数/到期限制，CPU90°C/GPU85°C连续2次、5秒采样、300秒恢复冷却、异常与连续2次无保存进度保护保留。评测和训练串行，不操作真机。

每阶段更新本README、索引/current state，审计后scoped提交推送main并核验远端。发布版本以Git历史为准；后续更新实际结果后再报告完成。
