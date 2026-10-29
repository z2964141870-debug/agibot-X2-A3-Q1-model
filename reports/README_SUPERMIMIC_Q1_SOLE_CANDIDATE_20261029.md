# Q1脚底奖励候选与连接恢复

记录日期：2026-10-29，北京时间。阶段：SOLE01。状态：连接已恢复；真实环境10项测试及原Q1三固定片CPU几何/奖励契约通过，底座候选、原生16/4096与H32方向验证继续准备。

## 目标与范围

按用户“SuperMimic底座与PPO两手抓”和“可以，继续推进”的决定，继续把[脚底时序诊断](README_SUPERMIMIC_Q1_DUALTRACK_DIAGNOSIS_20261029.md)变成可验证的独立候选。本次在权威Git的Q1分支实现脚底几何、摆脚奖励替换和CPU契约入口，没有沿用已经结束的夜间预算。

底座线下一项仍是原生H32闭环actor/critic方向核查及精确MDP表达；它尚未完成。适配线已实现单变量奖励候选，但不是可直接开训的完整配方。候选采纳仍为null。

本次0新物理积分、0策略推理、0optimizer更新；没有恢复旧RF15/RF16、启动agent、修改goal或自动任务。原监督保留ACTIVE/每10分钟，整体目标未结算。

## 恢复后的真实环境验证

2026-10-29 13:48后，本次复用已由独立工作恢复的25386 master47384，没有再次输入凭证或替换Windows桥。hostname、boot与GPU仍为原interactive40173/4d859008/GPU-26028785，现场GPU15MiB/0%且无匹配计算任务。场景与330库SHA均与既有记录一致；此前UNKNOWN已经解除，下面认证拒绝记录作为历史保留。

在原HP Python与activate.py下，10项测试全部通过，0跳过。原三固定片804帧中抽51个姿态，碰撞脚底FK与MuJoCo几何最大差1.356607071e-7m；参考相位/特征及其他奖励分项逐项保持，自参考奖励一致，无摆脚时奖励完全一致。根Z整体偏移下两种Z误差基本相同；固定右踝roll偏移.12rad时新旧奖励最大差3.502517939e-5。这是计算与隔离契约，不是物理收益。

15项已提交候选/双线记录、5项既有抬脚分析产物及3项本次CPU收据/日志，合计23项239048字节，Mac/25386逐大小与SHA一致。共享AGENTS/两README在本次编辑前也与当前提交相同。新CPU原收据在`data/supermimic_for_q1/Q1_SOLE01_CPU_CONTRACT_20261029/receipt.json`；[恢复与实测清单](../data/manifests/supermimic_q1_sole_cpu_validation_20261029.json)记录全部来源和镜像范围，覆盖早期“服务器镜像待认证”的当前状态。

已读取实际MDP与learner：底座身体位置误差为14个跟踪body三维误差的均值，脚部采用ankle-roll body原点；H32 actor目标包含闭环想象、soft termination、折扣及critic末端值。下一项按此真实表达实现独立脚底Z候选与方向核查，不更改历史runtime。当前仍0新积分/策略推理/optimizer，未运行原生16/4096或新训练，用户采纳null。

## 首次连接事实（历史）

用户提供Windows登录方式后，本次发现已有Windows桥：`/tmp/yuanqi-windows-25386.sock`，master为47236，`127.0.0.1:18873`正在监听并转向当前25386；本次保留此进程，没有另建或重启桥。

本机通过原HostKeyAlias连接25386时，BatchMode认证返回`Permission denied (publickey,password)`。Windows端的BatchMode SSH在主机密钥核查处停止，未进入容器认证；未为此放宽或写入Windows主机密钥文件。

用户随后明确提供容器认证信息，本次通过非回显交互提示输入一次，25386仍返回`Permission denied (publickey,password)`；未重试或猜测其他账号、密码及入口。凭证未写入命令参数、文件或Git。已请求核对该容器的当前登录方式，Windows桥保持。

因此容器的当前hostname、boot、GPU占用和runtime状态仍为UNKNOWN，不能以旧身份或认证失败推断远端停机。服务器镜像、实际Torch检查和原生验证均待登录恢复。

## 已实现内容

新增目录`script/supermimic_q1_sole/`，只读取既有Q1/HP接口与RF16逻辑，不修改历史冻结代码。

| 文件 | 作用与当前边界 |
| --- | --- |
| `sole_geometry.py` | 从实际MJCF获取活跃碰撞网格和球体，转换到脚部body局部坐标；用世界Z方向求最低网格顶点或球面高度。复用已有`foot_geometry`选择逻辑，限定真实z0平面 |
| `tensor_geometry.py` | 通过Q1现有可微FK的body位置/旋转计算相同脚底高度；另提供摆脚Z评分修正量。真实Torch与梯度检查未运行 |
| `sole_reference.py` | 继承RF16的ContactReference，预计算原参考脚底Z；只修正摆脚world-Z评分，其他奖励分项、特征、相位与系数保持 |
| `sole_adapter.py` | 在独立Adapter中替换奖励对象，复用RF16动作与PPO接口；尚未用原生环境实例化 |
| `check_contract.py` | 零积分CPU入口，拟比较原三固定片FK脚底与MuJoCo几何、自参考等价、特征/相位/其他奖励一致性；CLI通过，真实执行未运行 |
| `test_geometry.py` | 几何解析反例与真实Torch张量/梯度/评分检查；本地实际7通过、3因Torch缺失跳过 |

摆脚原式为`0.5*exp(-xy_error) + 0.5*exp(-ankle_z_error)`，外乘`.02`并按参考摆脚数归一。候选仅将第二项改为`0.5*exp(-sole_z_error)`，Z误差尺度仍为`.04m`。实现通过向旧reward和旧swing_bonus加入同一修正量，保留XY分支和其余分解。没有摆脚时修正为零。

脚底几何会随足部旋转变化；球体半径沿世界Z扣除，不能将局部球底点旋转后当作球面最低点。网格最小值在最低顶点切换处是分段可微的，不因此声称完整训练梯度已经有效。

教师、静态上肢alpha.50、14维腿腰残差、原`.15*tanh`权限、std.1、原PPO设置及控制/数据均沿既有接口保留。这只是代码上的受控候选，尚未通过实际Adapter与16/4096预检，不能称已证明全部契约不变或步态改善。

## 实际验证与未完成项

Mac本地执行：

```sh
python3 -m unittest discover -s script/supermimic_q1_sole -p 'test_*.py' -v
python3 script/supermimic_q1_sole/check_contract.py --help
```

结果为7项解析测试通过、3项真实Torch检查跳过；6个Python文件AST解析通过，CPU检查CLI help通过。本地Python有numpy，没有torch或mujoco，未安装额外包。不能把这组结果称为10项通过、真实Q1几何核验、原生预检或训练通过。

7项已运行检查覆盖平移/批量维度、足部倾斜、球体世界Z半径、网格与球体取最小、非有限/负半径/空几何/脚序拒绝、姿态维度拒绝以及几何副本/指纹不随外部数组改变。3项待Torch检查为张量与numpy一致、姿态梯度有限差分、摆脚修正的相位与系数。

恢复后，先核服务器身份、唯一实体根、现有GPU任务和原runtime/scene，再发布/镜像新来源并运行以下CPU检查；此命令未执行：

```sh
Q1_ROOT=/opt/data/private/ZHY
Q1_PY="$Q1_ROOT/data/environments/supermimic_q1_hp_20261026/bin/python"
Q1_RUNTIME="$Q1_ROOT/data/references/supermimichp/TRACKING_C02_sampler_runtime_20261027"
Q1_TRIAL="$Q1_ROOT/data/supermimic_for_q1/BONES_RF16_CONTACT_PHASE_SOMA300_E4096_MAX20M_20261029"
"$Q1_PY" "$Q1_ROOT/script/supermimic_q1_hp/activate.py" \
  "$Q1_ROOT/script/supermimic_q1_sole/check_contract.py" \
  --runtime "$Q1_RUNTIME" --scene "$Q1_RUNTIME/q1_assets/scene.xml" \
  --library "$Q1_TRIAL/evaluation/evaluation_library.json" \
  --output "$Q1_ROOT/data/supermimic_for_q1/Q1_SOLE01_CPU_CONTRACT_20261029"
```

输出目录已存在会拒绝覆盖。CPU真实检查会从原库取三固定片，按保存参考姿态核FK、碰撞脚底、特征/接触相位和reward分解，不积分或加载策略；拟采样自参考及合成root-Z/踝roll偏移，不能当实际物理收益。

底座H32方向必须读服务器实际learner/MDP，在克隆参数和状态/RNG/归一化保护下验证闭环想象方向与真实物理方向。旧固定未来动作探针不替代它。当前尚未补齐该工作，也没有从外围脚本猜写底座reward。

## 保存与下一步

阶段清单见[SOLE01](../data/manifests/supermimic_q1_sole_candidate_20261029.json)。源码和记录在持久权威Git，Git发布/独立40位远端核验及Mac镜像以实际发布收据为准，服务器镜像明确待认证；不声称三端一致。权重与大NPZ保持SERVER_LOCAL_ONLY。

下一步是恢复25386认证，执行真实CPU与张量检查，读取原生H32接口，再分别完成底座与适配候选的16/4096契约和发布冻结。没有自动批量四轮训练、恢复旧训练或补夜间预算；后续单轮上限仍为3600实训秒，不自动故障重试或扩时。原版MorphMimic复现属于另一独立工作，相关未提交记录保留，不据此替代SuperMimic/PPO两条线。
