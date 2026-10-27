# MG05：原生接触、闭环方向与候选验证

记录日期：2026-10-27，北京时间。BD03/C01进行中；MG05工具已核为active，保留用户原四项目标。25386为唯一工作入口，原25245停服/paused条目仅历史。未启动本轮正式训练，不再开子agent。

## 范围与版本

权威Git为`/tmp/yuanqi-supermimic-q1-20261023`的Q1分支，服务器根`/opt/data/private/ZHY`。原RF09 runtime、权重和330参考保持；新候选独立放在`data/references/supermimichp/TRACKING_C01_runtime_20261027`。起始发布提交`eee73affab7ba7eede2da3fccded2dc4d5af71be`，本阶段改动及提交见Git历史。

先完成上肢/接触/模型方向诊断，再选择观测/tracking改动，验证后只运行一次4096环境、实际训练不超过3600秒。不追加数据门、第二训练或续训。唯一上游子agent核查已完成，见[上游报告](README_SUPERMIMICHP_Q1_UPDATE_20261027.md)。

## 原生接触与重复性

`script/project_tools/diagnose_q1_hp_native.py`使用原生330环境/固定RF09 final策略，在`backend.step_tensor`完成后、manager刷新与autoreset前读取已求解接触力；没有额外forward或积分。每20ms控制周期只读最后5ms子步快照，不是20ms积分冲量。脚地垂直力大于5N判承重，以该力加权接触点切向速度。

V01因速度自检错误失败：`mjOBJ_BODY`报告质心速度，原公式却代入连杆原点。V02起同时核BODY/`xipos`和XBODY/`xpos`，最大误差为0；失败目录保持。实际接触点公式仍以`cvel`的子树质心为原点，没有为过检查修改力或滑速。

V02/V03的普通走受力快照摘要：

| 固定参考 | 单支撑比例范围 | 支撑侧切换 | 承重滑速均值范围 | 力矩饱和比例范围 |
| --- | ---: | ---: | ---: | ---: |
| walk_ff_stop_180_R_002__A095 | 60.0%-71.4% | 23-24 | 0.169-0.288m/s | 0.142%-0.290% |
| walk_ff_start_180_R_002__A071 | 46.1%-49.7% | 9-10 | 0.156-0.170m/s | 0.161%-0.282% |

两条手势双支撑约98.9%-99.8%，承重滑速约0.005-0.007m/s。以上是所选真实执行前缀，不是全库结论；滑速也受接触点转动/滑动及5ms快照定义影响，不能把所有非零值都解释为足底整块滑移。

V03在496个控制步逐次核所选4环境的`qpos/qvel/qacc_warmstart`，读取接触前后完全不变。两次不带接触读取的rollout本身也分歧：stop步数320/304，start均210；加入读取后的另一次为266/141。前10步状态误差可以很小，随后放大；手势也有轻微首步差异。故接触读取没有直接写这三类状态，但不能宣称全流程轨迹逐位确定，也不能用单次是否倒地证明观察器或候选改善。具体非确定性/重置残余来源未锁定，旧GPU RF09的291/142步不要求精确重现。

本结果排除“这些原生走路完全没有单支撑”的描述；未证明滑移为唯一倒地原因，也不证明全身PD已调好。模型统一推导PD仍非厂家逐轴实测参数。

## 模型动作方向

V02复用同一原生模型，6个预选状态、22轴中心差分，首动作扰动±0.08，后续冻结策略按分支状态重新给动作；所有分支冷启动，原生逐world参数一致性检查通过。模型端继承RF09 BF16。比较累计tracking/关节负平方误差对首动作的导数，不包含完整actor的critic、折扣与存活权重。

| 6状态平均余弦 | H1 | H4 | H8 | H32 |
| --- | ---: | ---: | ---: | ---: |
| 原tracking | 0.878 | 0.731 | 0.648 | 0.191 |
| 上肢关节误差 | 0.996 | 0.988 | 0.936 | 0.292 |
| 腿关节误差 | 0.961 | 0.924 | 0.804 | 0.315 |

H32综合方向符号一致率约53.8%，明显弱于短步长。支持把长程模型方向作为主要风险；不证明H32完整actor目标必然有害，不直接据此缩短horizon或追加试验。短步长上肢方向较好，结合[离线持续bias](README_SUPERMIMIC_Q1_MG05_SIGNAL_ANALYSIS_20261027.md)，优先验证显式误差反馈和直接关节tracking信号。

同一已保存方向JSON再按类型分组，0新增物理/策略/optimizer：普通走的4个状态，tracking余弦H1/H8/H32为0.938/0.591/-0.105，H32符号一致47.7%；两条手势状态相应0.758/0.763/0.782，H32符号一致65.9%。各组只有4/2状态，不能统计外推整个库；长程有限差分还受接触/冷启动/BF16及扰动尺度影响。该分组说明原6状态平均掩盖类型差别，不能称所有动作的H32方向都坏。RF10保持H32，先检验反馈/关节tracking组合，不声称同时解决长程腿部问题。

## C01实际构建与CPU契约

候选已在25386实际生成，来源清单SHA`f46399f7980e4861742c11ae6ccf64b868f2359778a996b7abfe7fd4800b0d8b`。CPU实际5组检查/4片12状态通过：参考和合法bins、观测/goal、批量历史噪声及导数、reward分解/方向/最高值、旧checkpoint输入schema拒绝。0物理步/0optimizer，不能替代GPU闭环和正式效果。

候选组合：

- 修复已证尾bin问题，不改变动作/帧率或删数据。
- 当前`q_ref[f]-q_actual`22维追加到历史观测，80到102；保持未来goal590维不变。传感器关节噪声在误差字段使用相反数，真实采集/replay/imagine共用同一函数。
- 身体朝向std由0.4到0.8；位置/朝向权重各1，身体线/角速度各0.5，根位置/朝向各0.5，上/下肢关节项各0.5，tracking最高值仍5。
- 上肢/下肢关节std分别0.8/0.5rad。保留原动作平滑、PD/full effort、reset、参考及future padding。

多项组合改变，只评价整体是否改善，不作单因素归因或称已证最优。H32、WM1024×10、actor1024、critic5×4、UTD0.4/seed0保持；4096是环境数。

## 命令与产物

新源码`diagnose_q1_hp_native.py`的`--contacts-only`模式只补重复性/只读接触验证，不重做已完成方向。实际产物均在服务器`data/supermimic_for_q1/TRACKING_BD03_NATIVE_V0{1,2,3}_20261027`，日志同编号放`logs/`。trace留服务器，只有小JSON进入Git/Mac。

小证据：

- [V01失败原件](../data/manifests/supermimic_q1_MG05_BD03_V01_failure_20261027.json)
- [V02接触](../data/manifests/supermimic_q1_MG05_BD03_V02_contact_20261027.json)、[闭环方向](../data/manifests/supermimic_q1_MG05_BD03_directions_20261027.json)
- [V03只读/重复性](../data/manifests/supermimic_q1_MG05_BD03_V03_contact_20261027.json)、[终态/资源](../data/manifests/supermimic_q1_MG05_BD03_V03_status_20261027.json)
- [候选来源](../data/manifests/supermimic_q1_MG05_C01_candidate_source_20261027.json)、[CPU契约](../data/manifests/supermimic_q1_MG05_C01_cpu_contract_20261027.json)

复现命令的共同前缀为服务器根下`data/environments/supermimic_q1_hp_20261026/bin/python script/supermimic_q1_hp/activate.py`，完整参数由各status、source及日志登记；所有旧输出拒绝覆盖。

## 进行中与下一步

4096完整profile P01于18:29:55开始、18:32:14全流程PROFILE_COMPLETED，worker3916退出0并reap、GPU回15MiB。独立根`data/supermimic_for_q1/TRACKING_C01_PROFILE_P01_20261027`，102维真实转移/replay/H32世界模型与actor/critic更新全部有限：64步/262144转移、WM10次与actor1次，实际WM1024/actor1024/H32。CUDA分配/保留峰值6352994304/6924795904字节。它有诊断optimizer更新，不算正式训练，不复用其模型作正式初始化。[原生契约](../data/manifests/supermimic_q1_MG05_C01_profile_contract_20261027.json)、[receipt](../data/manifests/supermimic_q1_MG05_C01_profile_receipt_20261027.json)、[guard](../data/manifests/supermimic_q1_MG05_C01_profile_guard_20261027.json)及[终态](../data/manifests/supermimic_q1_MG05_C01_profile_status_20261027.json)已取回。

RF09 `run/ckpt_003500.pt`已核内部iteration3500、sim_steps36102144、wall_time3424.2676889896393秒，schema1/obs80。通过activate加载CUDA库后读取成功，直接调用Python缺libcudnn的失败不代表环境损坏。独立同库330片物理评测已完成，0optimizer，输出`data/supermimic_for_q1/MG05_RF09_CKPT3500_BASELINE_20261027`：[receipt](../data/manifests/supermimic_q1_MG05_RF09_ckpt3500_eval_receipt_20261027.json)/[结果](../data/manifests/supermimic_q1_MG05_RF09_ckpt3500_eval_20261027.json)。训练54/300提前终止、验证2/30；平均完成93.48%/95.70%，上肢RMSE0.9490/0.9376rad、根XY0.13993/0.07904m。新旧最终比较将使用共同执行前缀，不直接把这里全前缀误差当公平差值。

`evaluate_trial.py`增加可选单checkpoint模式已用该基线实际运行；runner增加通用baseline标签/文件名、可直接复用主评测作对比，避免同库重复积分。默认初末评测语义保持。

profile与基线完成后，发布并冻结候选/入口，再独立RF10运行一次≤3600秒；结果与RF09近一小时及五小时final分别比较。继续100GiB盘余/12GiB实验/30GiB共享内存/OOM0/86度/900秒无进度保护。终后交付六格PNG/PDF/CSV和至少30秒多类MP4，采用相同原始跟踪指标及共同前缀，分别披露新旧reward。重要阶段提交push Q1并独立核远端，三端小证据核大小/SHA；当前本阶段尚未提交，未称网盘备份。
