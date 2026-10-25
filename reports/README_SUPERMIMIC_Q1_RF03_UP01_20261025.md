# RF03-UP01 双臂单动作可学习性实验

记录日期：2026-10-25，北京时间。状态：22:35:23已启动，已观察到世界模型和策略真实更新；训练与终后评测仍进行中，尚无本轮效果结论。

## 目标与范围

用户确认重新定目标并继续推进。先回答一个具体问题：保持现有SuperMimic世界模型、H32想象BPTT、奖励和控制参数，Q1能否从零学会一条包含中途双臂抬起和移动的参考。主要变量只改训练参考范围，从RF02的1024条收缩到1条。不把单动作学习当作未见动作泛化或真机成功。

依据[RF02全身诊断D02](README_SUPERMIMIC_Q1_RF02_FULLBODY_20261025.md)：平均肩肘误差改善，但主动抬手幅度只有目标8.45–13.16%，移动也主要停在站立附近；源集本身上肢覆盖少。先做单动作对照，避免同时改变数据、奖励和控制后无法判断原因。此目标不恢复代理已取消的数据几何硬门，也不把诊断指标当新准入条件。

## 固定输入与预算

- 参考：`train_3419f09ba93c91d75fc1`，365帧/50Hz，原1024条train中的双臂抬起动作，参考根XY净位移约1.6015m；短于当前500步截断条件。
- 既有合库：`data/supermimic_for_q1/BONES_PF02_1024_20261025/library/`；直接传`--clips`，不复制、不重新转换、不扩库。
- NPZ SHA-256：`f9bce4c3452720cda88a5adc4111fe2354440dd04e8c8ecb50cbf94d7181b8c3`。实体来自`BONES_PF02_0768_20261025/converted/`，合库入口沿用既有链接。
- 64环境、4线程/4模型槽、12000采集轮/768000真实转移；batch128、WM batch64×10、warmup128、model warmup50、每1000轮保存、seed2026102513、CUDA且不compile。与RF02相同，从零初始化，不加载RF02或A3权重。
- 后端收据：`data/supermimic_for_q1/PHUMA_P05_slots_V01_20261024/receipt.json`；既有PD、力矩、接触与网络配置全部保持。
- 单实验12GiB，余盘至少100GiB，共享容器内存30GiB/OOM基线14、无进展900秒及既有86摄氏度保护。共享盘变化按现场记录，不归因成本项目写入。

## 实现与评测协议

入口`script/project_tools/run_q1_single_motion.py`建立独立、有限预算、无自动重试的流程，复用现有guard、训练和评测工具，不修改训练核心。所有阶段落单独日志与guard收据，停止信号回收其拥有的worker进程组。

1. 经`train_q1_with_dashboard.py`训练并被动记录已完成回合；终后自动生成用户约定2×3训练曲线PNG/PDF/CSV。
2. 核查最终保存；只有实际完成12000轮、768000转移且实际训练名单只有上述1条，才进入最终对照。
3. 本轮自己的`step_000000.pt`与`final.pt`在同一条参考上分别做帧0开始的真实MuJoCo评测，不使用RF02初始轨迹替代。保留提前失败、整参考分母和完整执行步数。
4. 核对原数组与身份，复用`diagnose_fullbody.py analyze`报告肩肘、腰腿、腕相对躯干轨迹、左右抬手幅度/峰值时序/目标高于肩时的跟踪，以及根移动与完成度。平均误差与主动抬手事件分别报告。
5. 初始和最终都以保存的真实动力学轨迹渲染短MP4，不另建网站；与六宫格图一起复制到Mac的对应`data/`目录并核大小/SHA。图像工作保持简短，优先报告结果。

入口同时冻结训练包装器、绘图、评测、分析和渲染源码身份，训练自身仍保存核心源码/配置/输入身份。3项聚焦控制覆盖停止/部分预算不得冒充完成、多动作实际准入不得冒充单动作、配对身份变化不得比较；Mac及服务器`python3 -m unittest discover -s script/project_tools -p test_q1_single_motion.py -v`均3/3通过，`git diff --check`通过。

## 版本、命令与产物

权威代码副本`/tmp/yuanqi-supermimic-q1-20261023`，Q1分支，训练冻结提交`adb25ba3a63851eccb30b305128671ac5a835f20`，已推送并独立核对`refs/heads/Q1`；五个部署文件逐SHA与服务器一致。唯一实体服务器根`huanyu:/opt/data/private/ZHY`。服务器没有rsync，首次同步退出127，已改scp完成，没有安装软件。

```sh
data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python -u script/project_tools/run_q1_single_motion.py --root /opt/data/private/ZHY --code-commit adb25ba3a63851eccb30b305128671ac5a835f20 --expected-oom-kill 14 --detach
```

产物根`data/supermimic_for_q1/BONES_RF03_UP01_12000_20261025/`：`launch.json`、`plan.json`、`status.json`、各阶段`*_guard.json`、`run/`权重与逐轮指标、`run_dashboard/`六宫格、`evaluation_{initial,final}/`、`fullbody_analysis.json`、`comparison.json`、`render_{initial,final}/comparison.mp4`。日志`logs/BONES_RF03_UP01_12000_20261025_*.log`。大型模型及完整轨迹只留服务器，状态`LOCAL_ONLY`。

## 启动观察与证据

22:35:23实际启动独立controller883459，start_ticks3623092904；训练worker883460，start_ticks3623092915，boot仍`d994e71f-bae0-46ba-a5f7-6c2163771c6c`。实际准入只有选中1条/365帧，从零初始化。逐项比较RF02/RF03实际admission：资产、仿真、WM、actor、critic及两个trainer配置全相同，所有共同训练源码SHA相同、无新增训练源码；参数只有`clips`、输出目录和发布提交号不同。[范围与配置核对](../data/manifests/supermimic_q1_RF03_UP01_scope_check_20261025.json)。

已原样取回第375轮/24000转移状态，WM99组/actor49次、数值有限、GPU40摄氏度/容器24.697GB/OOM14。同期guard RUNNING、盘余1111389138944字节，温度/内存/存储仍在预算内；这是启动观察，不是跟踪改善证据。训练已脱离前台SSH独立运行；此处PID不能代替后续实时核验。

小收据：[启动](../data/manifests/supermimic_q1_RF03_UP01_launch_20261025.json)、[完整计划](../data/manifests/supermimic_q1_RF03_UP01_plan_20261025.json)、[375轮原样状态](../data/manifests/supermimic_q1_RF03_UP01_start_status_20261025.json)、[同期guard](../data/manifests/supermimic_q1_RF03_UP01_start_guard_20261025.json)。完整admission438926字节/SHA`540bb3c4ac5cb40af9744f662cacb8173a273574575d5f9c32cbf81bc673a95f`保留服务器；Mac仅有小副本，没有下载权重。后续完成固定评测、图与本地MP4，再依据主动抬手和移动的结果决定下一步。旧RF02以及FAILED/STOPPED终态保持，本轮无自动重试或延长预算。

恢复先读本文件、`README_CURRENT_STATE.md`和服务器本轮`status.json`/`train_guard.json`/`run/status.json`，重新核实进程身份与资源，不根据历史RUNNING快照重启或追加预算。
