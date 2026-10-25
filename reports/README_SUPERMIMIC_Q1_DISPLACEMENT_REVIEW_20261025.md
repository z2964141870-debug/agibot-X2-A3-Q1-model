# Q1站立但不移动：只读位移诊断

记录日期：2026-10-25，北京时间。阶段：RF02-D03。状态：只读诊断完成，原因尚未通过独立干预实验确定。

## 目标与范围

解释RF02为什么能提高回报、保持不倒，却没有学会跟随参考移动；为主代理的新单动作训练提供最少的高价值检查。继续优先复刻SuperMimic，不增加数据筛选硬门槛，也不自动增加actor的anchor观测。

本阶段由唯一子agent完成。实际读取本地源码、小型收据，以及huanyu上固定8段参考、已有最终轨迹和D02数组；补充计算只导入JSON、Path、NumPy，单线程、少于100MiB，约0.74秒。未加载Torch/checkpoint，未用GPU，未启动训练或新动力仿真，未改核心代码、输入、索引或当前状态。D02此前的动力重放与方向探针是已有证据，不是本阶段新执行。

## 最重要的结论

目前最支持的解释是**站立附近的局部最优**：策略降低关节越限、改善相对姿态和部分速度指标，从而提高回报，但腿部动作幅度较小、几乎一直双脚支撑，未形成持续移动。全局root位置项在偏离后迅速衰减，身体姿态项又会随机器人XY/yaw重新对齐，稳定站立仍可得到较高回报。

这是一条由行为和奖励结构支持的解释，尚不能确定是数据覆盖、接触转换处的世界模型/actor梯度、Q1控制参数还是重定向时序中的哪一项阻止了迈步。延长训练、增加环境数或增加数据都不是已证明的修复。

已核验参考移动信号存在且一致；没有发现root坐标被置零、目标速度未输入actor或动作clip20已成为主要瓶颈的证据。实际root线速度与全局漂移没有作为actor反馈输入，但这与论文的本体观测设计一致，不能直接判为实现错误，也不能据此承诺增加观测就能解决Q1迈步。

## 完成内容与实测证据

### 固定8段的初始/最终动力学结果

这8段均属于训练集，不能充当未见动作泛化评测。所有均值先按段计算，再对8段等权平均。

| 指标 | 初始 | RF02最终 | 能说明什么 |
| --- | ---: | ---: | --- |
| 全参考归一化回报 | 0.379982 | 0.654285 | 回报确实提高 |
| 完成比例 | 0.824645 | 1.000000 | 全长执行改善，不等同移动成功 |
| 提前失败段数 | 2/8 | 0/8 | 稳定性改善 |
| 共同执行前缀全关节RMSE/rad | 0.439520 | 0.274180 | 同时段姿态跟随改善 |
| 最终净root XY位移/m | 未作等长对照 | 0.020347 | 参考均值1.815288m，移动未学会 |
| 最终root XY路径长度/m | 未作等长对照 | 0.097389 | 不只是走了又返回原处 |
| root XY速度均值/(m/s) | 未复算 | 0.018562 | 参考均值0.326671m/s |

共同前缀RMSE与最终全执行RMSE不是同一个统计量；最终全执行RMSE为0.278758rad，不应混写。

归一化回报分解如下。表中为各项加权、按全参考长度和最大单步奖励归一化后的贡献，可直接相加；不是未加权的指数分数。

| 奖励项 | 初始 | 最终 | 变化 |
| --- | ---: | ---: | ---: |
| 全局root位置 | 0.031155 | 0.031841 | +0.000687 |
| 全局root朝向 | 0.046644 | 0.060070 | +0.013426 |
| 身体相对位置 | 0.154197 | 0.190804 | +0.036607 |
| 身体相对朝向 | 0.007776 | 0.074723 | +0.066947 |
| 身体线速度 | 0.135121 | 0.164040 | +0.028919 |
| 身体角速度 | 0.114578 | 0.137593 | +0.023015 |
| 关节越限 | -0.109489 | -0.000656 | +0.108833 |
| 动作变化 | 约0 | -0.004131 | -0.004131 |

最终约49.77896%的已执行帧，全局root位置的指数分数低于该项最大值的1%。回报增长中，root位置改善很小；越限惩罚减少贡献明显。这支持“回报提高但移动没有改善”，不能据总回报声称运控已有效。

来源：`data/manifests/supermimic_q1_BONES_RF02_comparison_20261025.json`、`data/manifests/supermimic_q1_RF02_D02_eight_20261025.json`。

### 新补充的纯数组核验

| 检查 | 实际结果 | 解释边界 |
| --- | ---: | --- |
| 参考qpos root与body pelvis位置最大差 | 0 | 导出的root与骨盆位置一致 |
| 参考qvel root与body pelvis线速度最大差 | 0 | 导出的两个速度入口一致 |
| 以0.02秒积分参考XY速度与净位移的最大差/m | 9.4761e-8 | 移动速度没有丢失或被清零 |
| 最大参考静态目标动作绝对值 | 14.2800 | `(joint_pos-q0)/scale`，低于clip20；不是动力学可执行性证明 |
| 实际策略动作最大绝对值 | 9.78395 | 现有最终轨迹未触及clip20 |
| 实际动作平均绝对值 | 2.01099 | 单位为归一化动作 |
| 腿部时域峰峰值向量范数：实际/参考 | 0.264911 | 逐段比值的均值；含reset暂态，仅描述幅度，不是步态评分 |
| 各脚最低离地高度20%帧中的脚踝原点速度/(m/s) | 0.459728 | 低分位帧不等于真实支撑；脚部转动/支点变化可移动踝原点，不能认定已证实滑脚 |

最后一项只提示应检查同源G1→Q1映射→导出后的脚部刚体点与时序，不作为数据拒收依据。没有把稀疏手臂参考动作及reset暂态放大的幅度比解释成手臂学习效果。

| 固定段后缀（均为train_） | 帧数 | 参考root速度/(m/s) | 实际root速度/(m/s) | 腿部幅度比 |
| --- | ---: | ---: | ---: | ---: |
| 0052d4522747ef665036 | 244 | 0.409276 | 0.013500 | 0.277263 |
| 00660a3755173f532abb | 474 | 0.203865 | 0.008155 | 0.224738 |
| 0081d048da3e3d090c08 | 192 | 0.336067 | 0.014190 | 0.172586 |
| 008ad0df4aced3b85225 | 228 | 0.447660 | 0.018115 | 0.167554 |
| 00aeab45fa5f744d3138 | 566 | 0.325922 | 0.005954 | 0.178973 |
| 00e98fc0d7a57764c762 | 253 | 0.266431 | 0.017485 | 0.198818 |
| 0122791b8661dc86bc7b | 266 | 0.381159 | 0.047856 | 0.630405 |
| 0124097cf15003106ad9 | 186 | 0.242986 | 0.023246 | 0.268952 |

### 已有接触和梯度证据能排除到哪里

D02对已存动作精确重放，8段平均双脚法向力均高于1N的帧比例为97.86147%；腿部力矩饱和比例为0.50786%。这与“主要保持双支撑”一致，不支持“腿部普遍撞力矩上限”作为现阶段主因，但不能证明PD增益或25%额定力矩对行走已足够。

已有6个站立附近状态、H1/H8/H32、两种扰动幅度的局部探针，每个horizon的12/12方向与真实中心差分方向一致；去除动作变化项后仍成立。不过H32真实root XY改变只有约0.0000705m。探针扰动首个动作后开环执行，没有覆盖完整actor反馈、存活权重、末端critic，尤其没有覆盖真正接触切换状态。因此既不能宣布“世界模型梯度已全面正确”，也不能说“梯度完全错误导致不走”。

来源：`data/manifests/supermimic_q1_RF02_D02_contacts_20261025.json`、`data/manifests/supermimic_q1_RF02_D02_directions_20261025.json`、`data/manifests/supermimic_q1_RF02_D02_direction_audit_20261025.json`。

## 源码机制、假设与不确定性

以下行号对应本次审阅提交；是代码证据，不是额外物理实验。

| 环节 | 代码入口 | 核对结果及限制 |
| --- | --- | --- |
| 目标输入 | `script/supermimic_for_q1/supermimic/envs/mdp/motion_library.py:53`、`:222` | 10帧goal含参考heading系root线速度；没有被漏掉。 |
| actor自身输入 | `script/supermimic_for_q1/supermimic/world_model/heads.py:28` | 关节、角速度、重力、前次动作历史；不含实际root线速度/XY/yaw漂移。历史可提供部分间接信息，不能据此证明必然无法起步。 |
| critic特权量 | 同文件`:148`、`script/supermimic_for_q1/supermimic/policy/nets.py:85` | 全局漂移特征仅供critic。 |
| 坐标/积分 | `script/supermimic_for_q1/supermimic/world_model/state_spec.py:71`、`:90` | 世界系root线速度和root积分保留；未见固定root的代码。 |
| 奖励 | `script/supermimic_for_q1/supermimic/envs/mdp/rewards.py:46`、`:99`、`:114` | 姿态参考重新对齐机器人XY/yaw，全局root项仍独立；root项为`0.02*0.5*exp(-||error||²/0.3²)`，偏离后信号变弱。身体速度仍对全局参考比较。 |
| 终止 | `script/supermimic_for_q1/supermimic/world_model/heads.py:143` | 只按高度/倾斜硬终止，不按XY漂移终止；完成比例不能替代位移质量。 |
| actor优化 | `script/supermimic_for_q1/supermimic/imagination/policy_trainer.py:65`、`imagination/objective.py:10` | 含存活权重与末端critic；已有开环探针未测试完整目标。 |
| Q1控制 | `script/supermimic_for_q1/supermimic/q1_assets.py:62`、`q1_sim.py:128` | PD50/2、XML力矩25%、归一化噪声0.1；髋膝目标噪声约0.018rad、踝约0.011rad，探索幅度可能有限，尚无因果证据。 |
| 重定向 | `script/supermimic_q1_sources/bones_retarget.py:41`、`:83`；`script/q1/q1_retarget.py:154`；`script/supermimic_for_q1/prepare_motion.py:112` | 原时间重采样、root轨迹缩放和独立IK后，按全段最低脚高作一次竖直偏移；没有逐帧接触约束，不能保证支撑时序被保留。未据此判定参考不可行。 |

RF02使用1024段、326883帧、6517.18秒动作，共7048个一秒bin；768000条真实转移除以参考总帧数仅为约2.349。这只是平均暴露比例，非实际每帧覆盖统计，也不是任何最低训练门槛。64环境的真实采样多样性和探索不足是合理嫌疑，仍需干预验证。

已知500步人工截断的采样回报偏差见`reports/README_SUPERMIMIC_Q1_RF02_METHOD_REVIEW_20261025.md`，1024段中86段长度超过500帧。它可能影响困难采样；当前新训练的365帧单动作不触发该条件。不能把此项当作所有站立行为的已证实解释。

## 原论文与第三方边界

原论文的actor本体观测设计及Table A.5配置已重新核对。Table A.5为每GPU4096环境，WM每轮10批×2048个8步窗口，actor4096条H32轨迹，更新比例0.4、动作噪声0.1、每环境replay1000。表A.6是velocity-tracking reward，不能用作reset噪声的表格依据。

RF02的64环境、WM64×10、actor128与论文的相对计算比例不一致：按每条真实转移，WM预测步数约32、actor想象步数约25.6；论文对应16、12.8。主代理拟用2048环境、WM1024×10、actor2048、H32，可恢复16和12.8的比例。它同时改变真实采样量与批量，不能用已停止的64环境667轮试验作为等预算因果对照，也不能提前宣称扩容成功。

当前核心来源是第三方`CuboiLeo/SuperMimic-Reproduced@d9be385b3eeb13a638fb33fb200d3bfd9be9839f`，不是作者官方训练代码。该固定README报告paper-only本体观测试验约0.56且root跟踪差；增加anchor-relative观测的约0.904是其明确登记的方法偏离。固定实验日志也有扩大批量/DDP后仍约0.56的平台期。这些仅说明第三方曾遇到类似困难，不能证明Q1原因相同；其长LAFAN1动作协议与当前短BONES段也不同。

原始来源：

- 论文：`https://arxiv.org/pdf/2610.28480`，Table A.1、A.5。
- 官方项目：`https://supermimic.github.io/`。
- 固定第三方说明：`https://raw.githubusercontent.com/CuboiLeo/SuperMimic-Reproduced/d9be385b3eeb13a638fb33fb200d3bfd9be9839f/README.md`。
- 固定第三方实验：`https://raw.githubusercontent.com/CuboiLeo/SuperMimic-Reproduced/d9be385b3eeb13a638fb33fb200d3bfd9be9839f/docs/EXPERIMENTS.md`。

服务器该第三方目录为`/opt/data/private/ZHY/data/yuanqi_ops/references/supermimic_reproduced_20261022`，Git HEAD已独立核对为上述固定提交。没有把第三方高分当作官方复现结果。

## 最多三项后续因果检查

1. **先完成按论文比例配置的单动作训练和成对评测。** 主代理已于23:21:31启动2048环境、至少7200秒实际训练、8模型槽/8线程，真实现场见[RF04](README_SUPERMIMIC_Q1_E2048_20261025.md)；本报告的诊断不代表该训练完成。冻结同一365帧参考、初始/最终评测与控制配置，联合看净位移/路径、跟随误差、回合完成/硬终止、接触切换、H8模型误差与H32存活率。若仍只站立，就不再把“多训一会”当默认解释。训练规模、批量、真实转移一起变化，结果只能先证明整套设置是否更有效，不能分离单项因果。
2. **在少数冻结的支撑/迈步起始状态核对完整闭环actor目标。** 资源空闲后，以极小预算比较actor参数或动作偏置方向的正负有限差分，分别记录原始奖励、root奖励、存活项、末端critic；不更新optimizer。建议总计不超过约640个控制步。先回答完整目标是否鼓励真实移动，再判断需不需要修正某一条链路；不重复仅首动作开环探针。
3. **核对同源G1→映射Q1→最终导出的脚部时序。** 比较脚部刚体上的地面点、姿态/支点、root速度和相位，不能只用脚踝原点速度认定滑脚。若出现明确时序或控制疑点，再做相同起点/参考的极小PD或力矩上限对照。单纯关节PD跟踪失败也不能证明参考物理不可行。所有结果作为诊断，不恢复此前额外制定的数据入选门槛。

## 版本、命令与工件

- 审阅checkout：`/tmp/yuanqi-supermimic-q1-20261023`，分支`Q1`，审阅起点`f03db7566fcd590912b4cc9b24621d32b19319a2`。
- 服务器唯一项目根：`/opt/data/private/ZHY`。没有新增平级目录或大文件。
- RF02最终checkpoint既有身份：`/opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF02_12000_20261025/run/final.pt`，205912083字节，SHA-256 `fa5bdc4542986df522fd3aeb96a59f2c121eba3688f6527805f3fb98b254c8fb`；本阶段仅读收据身份，没有加载模型。
- 最终8段收据：`/opt/data/private/ZHY/data/supermimic_for_q1/BONES_BASELINE_G02_20261025/evaluation_final/receipt.json`，455027字节，SHA-256 `3cf73f19adbce9427f9a7cd109d2c555e6d5be94a32b9b73bd416949c7d0342a`。逐段reference/trace路径、大小与SHA在该收据中。
- D02数组：`/opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF02_D02_fullbody_20261025/physics/<clip>_contacts.npz`；身份见已跟踪contacts收据。
- RF03单动作试验已由主代理核验在2026-10-25 22:51:46停止：STOPPED_SIGNAL、667采集轮、42688真实转移、WM216、actor166，controller和guard已回收。此为主代理核验结果；本子agent没有重新探测进程，也不恢复该试验。

下面命令可在服务器用已有NumPy环境重算移动信号核验，不会加载Torch、checkpoint或执行物理步。只读receipt指定的8段，`state`的前29维为qpos、随后28维为qvel。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/opt/data/private/ZHY/data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np
root = Path('/opt/data/private/ZHY/data/supermimic_for_q1/BONES_BASELINE_G02_20261025/evaluation_final')
rows = json.loads((root / 'receipt.json').read_text())['rows']
ref_speeds, real_speeds, integral_errors = [], [], []
for row in rows:
    with np.load(row['reference']['path'], allow_pickle=False) as ref:
        pos = ref['qpos'][:, :2].astype(np.float64)
        vel = ref['qvel'][:, :2].astype(np.float64)
        ref_speeds.append(np.linalg.norm(vel, axis=1).mean())
        integral_errors.append(np.max(np.abs(.02 * vel[:-1].sum(axis=0) - (pos[-1] - pos[0]))))
    with np.load(row['trace']['path'], allow_pickle=False) as trace:
        real_speeds.append(np.linalg.norm(trace['state'][:, 29:31], axis=1).mean())
print({'clips': len(rows), 'reference_speed_mps': float(np.mean(ref_speeds)),
       'actual_speed_mps': float(np.mean(real_speeds)),
       'max_integrated_displacement_error_m': float(max(integral_errors))})
PY
```

## 保存、同步与恢复

本阶段只新增本报告。子agent未提交或推送，主代理已审阅并将报告链接集成到索引、当前状态及RF04入口；GitHub和三端同步以父阶段实际提交/核验为准。既有服务器大型工件继续按原登记的LOCAL_ONLY处理，本阶段没有新增备份或删除。

没有本阶段遗留的训练、仿真或计算进程。主代理的新实验状态应以其最新日志/收据为准，本报告不覆盖正在更新的训练终态。

继续前读取`reports/README_CURRENT_STATE.md`、`reports/README_SUPERMIMIC_Q1_RF02_FULLBODY_20261025.md`、`reports/README_SUPERMIMIC_Q1_RF03_UP01_20261025.md`及主代理新单动作阶段入口。最优先事项是完成已授权的论文比例单动作试验并对照实际位移，再决定三项检查中后两项的执行顺序。
