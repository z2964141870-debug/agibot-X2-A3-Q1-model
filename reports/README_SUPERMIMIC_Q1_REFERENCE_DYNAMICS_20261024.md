# Q1参考接触与逆动力学诊断

记录日期：2026-10-24，北京时间。阶段：SM-Q1-D04。状态：独立实现、本地4项/服务器7项控制、真实三片分析及276项保存数组重算完成；三个峰值帧碰撞定位待执行。发现精确参考接触缺口，未证明训练失败根因或近似跟随不可行。

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

输出必须使用新目录；上述真实诊断命令已实际执行一次，原生退出0。日志`logs/supermimic_q1_D04_reference_dynamics_20261024.log`12037字节/SHA`df4bd478b1834ee1fe2377b34bfe6f6656c0efc1b753b02f7dbdaa1f653eeab6`，收据26307字节/SHA`d8d7efd272be4e0f8d985fbe1c50a5408eedebfb6d4f1f7fc54eef085f030e06`。逐帧NPZ仅服务器LOCAL_ONLY，Git只保存源码/配置说明和小JSON/日志。

## 验证、未解决项与下一步

本地unittest原生退出0：7项中4项通过、3项因本地未安装MuJoCo明确跳过，不将其记为7项已通过。已通过中心差分解析例子、非法速度拒绝、近地/力矩统计分母及空集、输入篡改/越界拒绝。服务器继续实际执行静态重力/自由落体、软接触分解和非零armature三项阳性对照，要求0跳过后才运行真实参考分析。

源码提交`503ffe8350c766c7984402e0b4bc6eaee03133a7`已推送并独立核对远端Q1。服务器部署大小/SHA逐项一致：诊断15962字节/SHA`ee36f6f9c89663b61c2655d06d58db7be44c356658cad440f1cbbe80f2c9e6ba`，测试5300字节/SHA`bbe9834780a5b51f9b52f049b59e47e8b2a7480cad28f4751c46aa4a3358eec8`。

服务器控制实际原生退出0，7项全部通过、0跳过，包括静态重力/自由落体、软接触分解和非零armature。日志`logs/supermimic_q1_D04_controls_20261024.log`1252字节/SHA`a6555bbf7197af1779a9bed67989bec2cb1ba259a4a816dd226cb32779554714`。

实际三片输入共3487帧，每片首尾各排除2帧，共分析3475帧，6950次inverse调用；91项源码/模型/参考/mesh身份前后一致，0积分/策略推理/optimizer、未导入Torch。实际峰值RSS709900KiB；进/出时容器31048278016/31726620672字节，OOM均14，未触发诊断或训练保护。该峰值包含模型编译，先前55284KiB只导入测量不能冒充完整峰值。

| 片段/用途 | 分析帧数 | 实际活跃脚地接触帧 | 较低脚底离地中位/P95 mm | 距地面30mm内帧比例 |
| --- | --- | --- | --- | --- |
| 走路/训练 | 1648 | 1/1648，0.06068% | 21.5335 / 26.5487 | 100% |
| 浅蹲/训练 | 1153 | 0/1153 | 3.8753 / 7.2152 | 100% |
| 出拳/复用开发 | 674 | 0/674 | 3.7184 / 4.6992 | 100% |

走路较低脚底平均离地19.6819mm。此前运动学门只要求距地面3cm以内，因而其通过不能证明实际支撑接触。小于等于10mm的走路足点样本共189个，切向速度P95为0.307234m/s，其中90.476%超过0.05m/s；这是近地候选点速度，不等同实际接触滑移率，更不能套用到全部走路帧。

两种差分下，三段精确参考所需根外力的中位数均约1倍体重，主要反映当前原始几何几乎无支撑接触。中央三点/五点主动关节所需力矩超过当前cap的帧比例，走路均为0.06068%，浅蹲均0，出拳为12.3145%/12.9080%；出拳峰值42.4879/43.2093倍cap。未优化接触力，也未分离软关节限位等约束贡献；这些数字不证明增加力矩就能跟随。出拳仍为复用开发片段。全部inverse与RNE+armature分解误差通过1e-7检查，数据time始终0。

00:54:11只读核验R03原controller/worker同boot且非僵尸、冻结源码/输入/运行时通过，2950/12000追加轮、47200新交互、1129组WM/1079次actor，累计5950轮；GPU39摄氏度/空闲16671MiB，诊断进程退出后容器31037546496字节、OOM14，无终态收据。没有停止/重启训练或读取checkpoint张量。

新增独立重算脚本`audit_reference_dynamics.py`从不可变NPZ/参考重新计算两种速度差分、全部统计分母/分位数/力矩比例，另列峰值对应关节/时刻供定位；不导入原统计函数，也不再次调用MuJoCo。命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $PY script/supermimic_q1_analysis/audit_reference_dynamics.py \
  --root "$PWD" \
  --receipt data/supermimic_for_q1/D04_reference_dynamics_20261024/receipt.json \
  --output data/supermimic_for_q1/D04_reference_dynamics_20261024/summary_audit.json
```

重算代码`de09192c56dc0f4323bdd70eb04efdf5ec368b08`已推送/独立核对Q1，部署SHA`e07c0a597500cb64e8b78279e4d6e39bd21272bad34b4d641e53d803641bedf6`一致。上述命令实际原生退出0，276项统计/加速度差分检查通过，最大绝对差1.77636e-15；0新MuJoCo/inverse/积分/推理/optimizer。审计13611字节/SHA`a1290fd078437ad9a3047e04a4c1d65613ef338212a8a6124416bf40443c32b0`，日志344字节/SHA`ce69ad81f45fedc42be9a1702b4fc4b72cd858d8d3cb9b3cb110d44ab2c9fac2`。

峰值定位：走路25.08秒右髋pitch为-14.1887Nm/cap9Nm；浅蹲0.10秒左肩pitch为-1.9554Nm/cap5.5Nm；出拳7.64秒左肩pitch为233.6832Nm/cap5.5Nm。出拳该关节q=0.156747rad、dq=1.999998rad/s、差分ddq=-20.2531rad/s2，原始限位[-3.1416,1.5708]，不能仅凭该大力矩认定需要提高cap。还需检查该帧其他接触/约束贡献。

新增`inspect_reference_contacts.py`对三片各自中央三点峰值帧做一次同协议inverse重算，要求与保存广义力最大差<1e-7，再列接触几何/身体、穿透距离、局部力和峰值关节constraint/passive/惯性偏置分解。只覆盖3帧、0积分/策略/optimizer，源码部署核验后执行一次：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $PY script/supermimic_q1_analysis/inspect_reference_contacts.py \
  --root "$PWD" \
  --receipt data/supermimic_for_q1/D04_reference_dynamics_20261024/receipt.json \
  --audit data/supermimic_for_q1/D04_reference_dynamics_20261024/summary_audit.json \
  --output data/supermimic_for_q1/D04_reference_dynamics_20261024/peak_contacts.json
```

该三帧补充尚未执行。01:02:22原R03仍3175/12000追加轮、50800新交互、累计6175轮，原进程/冻结条件通过，GPU39摄氏度/OOM14。等待其完整配对评测与V02渲染；如继续未过门，优先把接触一致性作为独立参考候选的待验证因素，保持匹配基线，不在正在运行的R03内替换输入。参考动力学缺口已确认，是否为学习失败主因仍未确认。

MuJoCo官方方法依据：`https://mujoco.readthedocs.io/en/stable/computation/index.html`及`https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html`，2026-10-24核对。论文项目入口`https://supermimic.github.io/`本轮复核仍列策略通过可微世界模型内回报反传的路线；本诊断不替代该学习方法。
