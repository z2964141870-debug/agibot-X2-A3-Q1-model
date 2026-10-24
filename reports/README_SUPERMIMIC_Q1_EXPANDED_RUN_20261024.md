# Q1扩库训练准备与低内存仿真

记录日期：2026-10-24，北京时间。阶段：SM-Q1-P05。状态：组库入口及本地四项组成边界检查通过；模型槽后端、数值/梯度验证入口及64环境有限训练入口已实现、语法检查通过，尚未实测。真实组库和新训练未执行。

## 目标与范围

继续SuperMimic for Q1可微世界模型、想象BPTT与critic路线。P04扩量已完成前829个不同候选，196个训练参考/708.38秒；本阶段将完整固定P04候选的已检查结果组装为可训练库，再核查更大在线交互规模的资源可行性。AMASS缺男女SMPL-X模型、完整BONES-SEED包与metadata尚缺，不以这些缺项阻止现有PHUMA准备。

## 已完成实现

新`script/supermimic_q1_expanded/assemble_library.py`复用P04跨批时轴汇总，要求完整固定名单按原顺序覆盖一次，核对来源组/原字节/clip唯一、Q1模型/控制/协议相同、全部19门齐全。随后独立重算完整源划分与实际ZIP原字节，只为最终合格参考创建符号链接。原始NPZ、IK轨迹及各批检查不搬动或重写。

组装manifest保留全部候选转换记录，参考收据保留已有独立检查记录；显式标注`COMPOSITION_OF_PRIOR_INDEPENDENT_19_GATE_RECORDS`及`geometric_checks_rerun=false`，不把组成记录说成重跑几何验证。新完整源重算、实际时间轴核对与旧trainer的`admitted_clips`分别执行。输出目录排他创建，失败保留，不覆盖重跑。

本地四项控制通过：完整批次可乱序输入但输出按原offset恢复且不修改输入；缺失/重复批次拒绝；混用模型与缺少一门拒绝；相同原字节跨划分拒绝。尚无真实组库通过结论，须等P04固定名单转换完成。

## 版本、命令与证据

Mac工作副本`/tmp/yuanqi-supermimic-q1-20261023`、分支Q1，阶段前HEAD`c54e067e9c89fc469bbd92f8d3e20a4a1e9a7c0d`。服务器huanyu项目根`/opt/data/private/YuanQi_Q1`，Python`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。

```bash
python3 -m unittest discover -s script/supermimic_q1_expanded -p 'test_assembly.py' -v
```

P04的B010于北京时间17:56:28新核主机/boot、无训练/转换进程、GPU38摄氏度/17788MiB空闲、内存27,764,060,160字节/OOM14后启动。固定offset829/count256，原会话句柄81765，日志`logs/PHUMA_P04_B010_20261024.log`、输出`data/supermimic_for_q1/PHUMA_P04_B010_20261024/`。原句柄已退出0，65训练参考/233.00秒，18:05:34完整completion及OOM14核验完成，详见[P04](README_SUPERMIMIC_Q1_PHUMA_EXPANSION_20261024.md)。下一新offset1085/B011，未重做已完成批次。

## 下一步与边界

继续单路P04剩余批次；完整输入到位后运行组库与原trainer准入。再独立实现少量模型槽的原生MuJoCo rollout，继承既有reset/step/reward/采样，不改世界模型算法或H32；须与原后端在相同随机种子、连续DR、噪声和反复reset下逐状态比较，并实测更大并行数资源。尚未改旧后端、未扩大环境数、未启动新训练或产生新策略效果。

每个实际完成阶段及时更新本文件、报告索引和当前状态；代码与小清单按Q1审核、提交、显式推送并核对远端。大工件仍LOCAL_ONLY，无新增网盘备份。

## 模型槽实现与拟定训练条件

新`slotted_sim.py`继承原`Q1Sim`的reset、step、奖励、回放、难度采样和终止逻辑。只替换初始化的物理模型存储及native rollout入口：少量真实模型分波处理全部独立环境，每波按各自的连续摩擦/躯干COM context重设参数并调用`mj_setConst`，使用现有MuJoCo3.3.7 `Rollout`，不量化DR或修改冻结旧后端。每环境的状态、历史、轨迹、context和帧号仍独立。

`verify_slotted_backend.py`拟在7环境/3槽及8环境/4槽分别执行96轮原后端/新后端对照，开启原连续DR、观测/动作噪声、7步episode和11步间隔的显式部分reset；比较逐步状态、奖励各项、帧号、历史、context、难度记忆及全部replay张量。浮点差上限1e-6，离散量精确一致；必须覆盖反复自动reset。随后从新后端实际replay重新执行既有冻结世界模型动作梯度有限差分与H32 actor反传，再做64环境/4槽32轮CPU资源采样。验证数据仅取训练参考，开发/测试不用于选后端。此为待执行协议，不是已通过结论。

`train_expanded.py`只在上述收据及精确source身份有效时绑定新收集器，再调用原PHUMA随机初始化训练入口。`expanded_pipeline.py`复用原登记中的资产、mesh与输入冻结检查，新建独立spec；实际任务显式64环境/4线程/4模型槽、Actor B128、WM B64、每次10批、H32，所有学习率/损失/优化器逻辑来自原trainer。

拟定新R02预算12000采集轮，即768000在线控制转移；原R01为48000。原R01与旧STOPPED实验不恢复。该新配置先固定数据、并行数和预算，以同一随机初始化的step0/final完整动力学评测衡量学习进展；不将与原R01的差异归因于单独的数据量或环境数。主要考察量为该固定配置内的训练进度，开发/测试隔离。源码只通过语法检查，预算尚未登记、无新训练进程；只有完整组库、后端数值/梯度及实际资源通过后才登记。

B011已正常结束，B012固定offset1341/count256、原句柄8737正在处理。数据最新终态及算术累计见[P04](README_SUPERMIMIC_Q1_PHUMA_EXPANSION_20261024.md)，截至B011训练311片/1121.28秒，未作完整新汇总。源名单的30分钟/1小时扩库工程目标仍未达到，当前不宣称数据量已复现论文规模。
