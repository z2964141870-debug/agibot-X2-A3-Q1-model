# Q1改进方案筛选与验证

记录日期：2026-10-23，北京时间。阶段：SM-Q1-S01/P07/J01。状态：原文核对完成；代码准备，尚未执行新增动力学检查。

## 目标与范围

用户要求寻找提升和解决方案。保持SuperMimic世界模型、k8/H32/BPTT/critic方法复现，优先修复已有配置问题，基于Q1实际证据决定补模型精度、训练预算或控制适配。不恢复A3，不把额外观测、PPO、教师策略或新奖励混入主线；原R01数据、权重、评测和视频通道保留。

## 原始来源核对

web检索/原站入口未返回可用内容，改用直接HTTP获取作者原文。原文`https://arxiv.org/pdf/2610.28480v1`保存于服务器`data/references/supermimic_2610.28480v1_S01.pdf`，3524904字节/SHA`fe213017ab062f4acc3965da89abd3c29969163e144a2628a662b8ce46691009`，与先前原文身份一致。已读第13–16页完整相关正文，第16页已渲染PNG并实际目视核对表格、公式和脚注。第三方固定源码`CuboiLeo/SuperMimic-Reproduced` d9be385的README/PAPER_SPEC只用于交叉核查，不替代原文或视为作者官方成绩。

| 项目 | 原文Table A.5 | Q1 R01实际 |
| --- | --- | --- |
| 每卡仿真环境 | 4096 | 16 |
| 每次actor想象轨迹 | 4096 | 32 |
| WM每次更新 | 10批×2048窗口 | 10批×64窗口 |
| 训练长度 | 320000 iterations | 3000采集轮/1099 actor更新 |
| 学习率 / H / k | 2e-4 / 32 / 8 | 相同初始值 / 32 / 8 |

论文iteration与采集/更新的精确计数口径有未指明项，不能把上述长度直接换成匹配交互倍数或训练耗时。16/4096、32/4096、64/2048分别为本轮环境、actor batch、WM batch相对论文数值，不声称同等数据难度/吞吐/学习效率。原文第13页强调低PD增益下设定点是力矩代理；因此“直接播放参考角度跟不上”不证明参考物理不可行。默认关节角/动作尺度/力矩范围仍需Q1核验。

固定独立复现README报告，论文观测下e2根跟踪较差，加入9维锚点相对位姿的e3效果更好；其结果不与原文同配置，不能直接作为我们加观测的复现成功依据。主线先保留论文观测，相关观测增强只列备选诊断，不作为当前方案实现。

## 关键决定与预登记

P07：新增`contact_protocol.ContactCorrectedAssets`，仅提高足碰撞priority到floor+1，沿用原PD/动作/资产。旧`supermimic/q1_assets.py`等冻结模块不改；R01仍由原适配器重现。新`verify_contact_protocol.py`在实际Q1Sim三次随机重置（两环境）核对context与contact，确定性重置核对真实摩擦1，随后82控制步/1640子步和FK/积分/冻结H32图验证，0optimizer。诊断用64宽网络只验证图和后端，不是完整策略成绩。新训练入口尚未接入该适配器。

J01：新增`probe_action_response.py`，只接纳R01最终两段训练轨迹，固定每段有效H32区间25%/65%各一状态，PCG64 seed20261023，每状态4个单位动作方向、首动作±0.05/±0.10，其余31动作固定。先四个无扰动基线全部复现，再做扰动；上限4×17×32=2176控制步/43520子步，0策略推理/optimizer。保持R01实际摩擦0.6，模型context仍为旧请求参数1，不混入P07物理改变。读取H1/H8/H32的根/关节响应，与模型有限差分/JVP比较；float64模型1e-4中心差分验证JVP，float32/float64再比较。失败样本保留、无替换或自动重试；弱响应不报cosine。所有结果为描述性局部响应，不是全闭环actor梯度正确性或策略成功门。

## 命令与证据

服务器`huanyu:/opt/data/private/YuanQi_Q1`。Python`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。入场资源快照3090 24576MiB、已用6348MiB、GPU0%/39°C；容器usage27942223872/34359738368字节、历史OOM14、boot d994e71f-bae0-46ba-a5f7-6c2163771c6c。其他历史进程保留，不能以GPU空闲视为资源独占。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' $PY -u script/supermimic_for_q1/verify_contact_protocol.py --root "$PWD" --output data/supermimic_for_q1/P07_backend_20261023
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' $PY -u script/supermimic_for_q1/probe_action_response.py --root "$PWD" --evaluation data/supermimic_for_q1/R01_eval_final_20261023/receipt.json --output data/supermimic_for_q1/J01_action_response_20261023
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 $PY -u script/supermimic_for_q1/probe_action_response.py --root "$PWD" --output data/supermimic_for_q1/J01_action_response_20261023 --execute --device cuda:0
```

此处为待执行命令，输出拒绝覆盖。源码以Git和准备plan中的SHA绑定。大文件仍LOCAL_ONLY，原始PDF不进Git或Mac。验证结果、原生退出码、实际计数和下一选择完成后更新本文件，不把准备写成通过。

准备验证：三个新增脚本AST通过；服务器计分控制原生退出0，反向模型必须给负cosine、零物理响应抑制cosine/ratio、完全匹配响应误差为零三项通过。上述不是实际物理/学习结果；日志`logs/supermimic_q1_S01_metric_controls_20261023.log`。

## 下一步与恢复

检查语法与科学计分控制→保存代码/协议→P07实际验证→J01准备/执行→按结果确定最优先一个改进方向，必要时建立独立训练对照。当前R02未登记或启动。恢复读取本文件、D02和Q1主阶段，并重新核对实际作业/资源；V01播放服务保留。报告索引和current state随结果更新，提交推送Q1并独立核验远端。
