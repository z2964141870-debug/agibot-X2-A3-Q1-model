# Q1参考接触与逆动力学诊断

记录日期：2026-10-24，北京时间。阶段：SM-Q1-D04。状态：独立实现及本地4项控制通过；3项MuJoCo控制和真实参考分析待服务器执行，尚无真实参考动力学结果。

## 目标与范围

R03继续原固定追加12000轮，00:35:49实核原controller762428/start3606081278、worker762431/start3606081380同boot非僵尸，2400轮/38400新交互、累计5400轮，冻结条件通过，GPU39摄氏度/OOM14。等待训练期间，补查现有参考的足底接触、近地滑移和逆动力学残差，帮助后续区分训练预算与控制/参考质量问题。

不修改R03冻结的两个训练源码目录、参考、奖励、PD/力矩、预算或停止保护，不启动并行训练、评测推理或GPU渲染。

## 方法与决定

独立脚本`script/supermimic_q1_analysis/inspect_reference_dynamics.py`只使用NumPy和MuJoCo，不导入Torch、加载checkpoint或调用积分。先绑定真实R03 admission中的源码/模型/参考manifest及三个合格参考身份，逐文件记录24个mesh输入，结束再核对全部输入。

按确定性评测协议重建同一Q1模型：相同关节/执行器映射、25%模型力矩上限、PD50/2、零关节被动阻尼、摩擦1.0及足碰撞优先级2。每片排除首尾各2帧，以保存的50Hz qvel分别做三点/五点中心差分；两种加速度都执行`mj_inverse`，用`mj_rne + dof_armature*qacc - qfrc_passive - qfrc_constraint`独立核对所需广义力，容差1e-7、数据time恒0。明确排除未覆盖的tendon和离散inverse模式，并以非零armature解析例子验证该项。

记录实际活跃脚地接触、左右足最低碰撞点离地及其切向速度；1/10/30mm邻地阈值和0.05m/s仅作描述，不新增训练通过门。逆动力学分别报告根所需外力/力矩和主动关节所需力矩对当前cap的比值，保存逐帧NPZ以便独立重算。

这些残差只描述精确保存参考和所选加速度在软接触模型下的力，未优化接触力分配；不能证明近似动作跟随不可行、不能把非零根力当作已证明根因，也不把模型额定力矩当厂家真机能力。三点/五点差异用于显示差分敏感性。

## 版本与复现

代码副本`/tmp/yuanqi-supermimic-q1-20261023`，分支`Q1`。服务器`huanyu:/opt/data/private/YuanQi_Q1`，Python沿用`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。服务器只导入NumPy/MuJoCo的探测RSS55284KiB，版本1.26.0/3.3.7；模型编译和完整诊断峰值仍待实测。脚本入场要求384MiB内存余量，运行每128帧保留128MiB余量并核对OOM计数，诊断自身失败不向训练发信号。

```bash
python3 -m unittest discover -s script/supermimic_q1_analysis -p test_inspect_reference_dynamics.py -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $PY -u script/supermimic_q1_analysis/inspect_reference_dynamics.py \
  --root "$PWD" \
  --admission data/supermimic_for_q1/R03_B128_extend12000_20261023/run/admission.json \
  --output data/supermimic_for_q1/D04_reference_dynamics_20261024
```

输出必须使用新目录；上述真实诊断命令尚未执行。日志拟保存到`logs/supermimic_q1_D04_reference_dynamics_20261024.log`，NPZ仅服务器LOCAL_ONLY，Git只保存源码/配置说明和小JSON/日志。

## 验证、未解决项与下一步

本地unittest原生退出0：7项中4项通过、3项因本地未安装MuJoCo明确跳过，不将其记为7项已通过。已通过中心差分解析例子、非法速度拒绝、近地/力矩统计分母及空集、输入篡改/越界拒绝。服务器继续实际执行静态重力/自由落体、软接触分解和非零armature三项阳性对照，要求0跳过后才运行真实参考分析。

先发布并核对部署SHA，再执行服务器控制和一次真实三片分析、独立从保存数组重算并记录实际内存与R03进程身份。尚无真实结果，不按诊断准备声称已解决行走；R03完整配对评测和V02渲染仍等待原训练结束。

MuJoCo官方方法依据：`https://mujoco.readthedocs.io/en/stable/computation/index.html`及`https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html`，2026-10-24核对。论文项目入口`https://supermimic.github.io/`本轮复核仍列策略通过可微世界模型内回报反传的路线；本诊断不替代该学习方法。
