# RF03-UP01 双臂单动作可学习性实验

记录日期：2026-10-25，北京时间。状态：目标与自动流程已实现，尚未启动训练；实际状态以本文件后续记录和服务器收据为准。

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

入口同时冻结训练包装器、绘图、评测、分析和渲染源码身份，训练自身仍保存核心源码/配置/输入身份。3项聚焦控制覆盖停止/部分预算不得冒充完成、多动作实际准入不得冒充单动作、配对身份变化不得比较；本机`python3 -m unittest discover -s script/project_tools -p test_q1_single_motion.py -v`已3/3通过，`git diff --check`通过。服务器验证待。

## 版本、命令与产物

权威代码副本`/tmp/yuanqi-supermimic-q1-20261023`，Q1分支，准备基于已发布`109e4616a04bfb5e5e5e6366995703fe506fe9a9`。执行前先提交并核对远端，再把其40位SHA传入。唯一实体服务器根`huanyu:/opt/data/private/ZHY`。

```sh
data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python -u script/project_tools/run_q1_single_motion.py --root /opt/data/private/ZHY --code-commit PUBLISHED_40_CHARACTER_SHA --expected-oom-kill 14 --detach
```

产物根`data/supermimic_for_q1/BONES_RF03_UP01_12000_20261025/`：`launch.json`、`plan.json`、`status.json`、各阶段`*_guard.json`、`run/`权重与逐轮指标、`run_dashboard/`六宫格、`evaluation_{initial,final}/`、`fullbody_analysis.json`、`comparison.json`、`render_{initial,final}/comparison.mp4`。日志`logs/BONES_RF03_UP01_12000_20261025_*.log`。大型模型及完整轨迹只留服务器，状态`LOCAL_ONLY`。

## 当前验证与下一步

新入口尚未在服务器执行，训练与效果均未完成。先跑聚焦控制、发布/同步代码、重新核GPU/盘余/共享内存及OOM，然后实际启动并观察真实更新；后续完成固定评测、图与本地MP4，再依据主动抬手和移动的结果决定下一步。旧RF02以及FAILED/STOPPED终态保持。

恢复先读本文件、`README_CURRENT_STATE.md`和服务器本轮`status.json`/`train_guard.json`/`run/status.json`，重新核实进程身份与资源，不根据历史RUNNING快照重启或追加预算。
