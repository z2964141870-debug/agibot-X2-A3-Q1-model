# RF02全身跟踪与奖励接触诊断

记录日期：2026-10-25，北京时间。阶段：RF02-D02。状态：全身诊断完成，下一轮训练建议已记录；0optimizer更新。完成诊断不代表机器人已学会动作。

## 目标与范围

用户要求继续细分奖励、重定向、接触和模型动作方向问题，并明确上肢/抬手跟踪同样重要。目标是在本轮RF02模型上形成可复现的全身诊断，依据实测决定下一轮单变量训练方向。保持SuperMimic复现路线，不把诊断指标重新变成数据准入门。

## 固定任务与预算

1. 在现有固定8条train初始/最终轨迹上分解真实奖励，分别报告全参考分母和共同执行前缀；量化腿、腰、肩肘以及腕位置的跟踪。
2. 只读取1024条已训练Q1参考，用左右腕相对肩的高度、幅度选择最多4条上肢参考，选择时不读取策略表现。固定名单后以原初始/最终模型各做一次真实MuJoCo评测；这些仍是训练内动作，不声称泛化。Q1只有肩肘8个上肢主动关节，腕为固定链上的位置点。
3. 重放已有最终动作以记录真实脚地接触和力矩，对照同参考的脚底几何高度/世界系速度。必须先复现原轨迹，参考的几何近地状态不冒称真实支撑力。
4. 在最多6个预先登记的已访问状态上比较模型奖励梯度与真实动作扰动的H1/H8/H32响应：3个模型奖励梯度方向及1个固定随机方向、幅度0.02/0.10、正负扰动，最多3264控制步/65280物理子步。首步动作扰动、后续动作固定，是局部开环奖励方向检查，不冒充完整闭环actor梯度。

只加载现有模型，0optimizer更新，不恢复旧训练。图像不作为诊断前置步骤；有必要的代表视频仅保存本地MP4，不建网站。

## 已确认输入与资源

当前代码基于Q1提交`3a8e7544364d2bea342d31ceca0d16a423591d2a`，RF02冻结训练版本`d66ac1cd21de1682ca6f069227c070ebf98da02f`。输入为`BONES_BASELINE_G02_20261025/evaluation_{initial,final}/receipt.json`、`BONES_PF02_1024_20261025/library/references/`及`BONES_RF02_12000_20261025/run/`。

统一服务器根`/opt/data/private/ZHY`，Python为`data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python`。本次开始实读盘余1479159111680字节，容器内存22725197824字节，under_oom0/OOM14。沿用100GiB余盘、30GiB内存与OOM保护，不新增大数据副本。

计划新增脚本位于`script/supermimic_q1_analysis/`，产物根`data/supermimic_for_q1/BONES_RF02_D02_fullbody_20261025/`，日志在`logs/`。未运行的命令和结果不记为完成。

## 已完成：原八片与上肢覆盖

`diagnose_fullbody.py`新增独立诊断入口，不改训练源码。三个聚焦控制通过：躯干坐标系对整体平移/旋转不变、早停分母及负奖励保留、只按参考确定性的去重选择。真实读取原八片16份轨迹，奖励逐步复算与原存值最大误差受2e-5控制；0新积分/策略前向/更新。

原八片共同前缀的片均RMSE（rad）：全关节0.439520→0.274180；腿0.316152→0.261113；腰0.054814→0.066016；肩肘8关节0.613089→0.300695。腕相对躯干3D位置RMSE：左0.129098→0.075521m，右0.123330→0.060843m。平均上肢姿态有改善，但不能据此声称完整抬手能力。

按整参考分母计算的奖励贡献：根位置0.031155→0.031841；根朝向0.046644→0.060070；身体位置0.154197→0.190804；身体朝向0.007776→0.074723；线速度0.135121→0.164040；角速度0.114578→0.137593；关节限位负项-0.109489→-0.000656；动作变化项约0→-0.004131；总计0.379982→0.654285。增益中约0.108833来自限位负项减少，根位置仅增加0.000687。最终根位置项低于自身上限1%的执行帧比例片均49.78%。这些是贡献分解，不是奖励权重的因果实验。

完整扫描1024份训练参考，腕在躯干坐标系高于同侧肩的动作：左4段、右3段；高度变化幅度中位数左0.03155m/右0.03102m，最大0.26033m/0.25163m。这里只描述当前Q1转换后的覆盖，尚未区分原G1源不足还是重定向丢失。

在不读取策略表现的条件下，冻结4条上肢评测：`train_0499c71242ed71cdf537`、`train_785a58b7421497bf9d30`、`train_c1fc0de44204d0196ac2`、`train_3419f09ba93c91d75fc1`。按左幅度/右幅度/左最高/右最高依次选取未选的最大值、同值按clip排序；472/670/898/365帧。第1片最大左抬手在frame0，必须区别起始姿态与后来主动抬手。下一用原初始/最终模型各评一次，不用结果重选样本。

小证据：[八片分解](../data/manifests/supermimic_q1_RF02_D02_eight_20261025.json)、[参考预选](../data/manifests/supermimic_q1_RF02_D02_arm_selection_20261025.json)。完整扫描仅留服务器`arm_reference_scan.json`；诊断产物均在本阶段产物根，无参考或权重副本。日志`logs/supermimic_q1_RF02_D02_{scan,eight}_20261025.log`。

## 已完成：四条上肢配对与来源覆盖

原初始/最终模型按冻结名单各真实评测一次，合计4810个控制步/96200物理子步；4/4均完整、无提前终止。肩肘RMSE0.636309→0.287691rad，腕相对躯干误差左0.130459→0.087599m/右0.117613→0.057980m，整片回报0.486932→0.718837。**平均误差改善，但没有学会中途主动抬手。**

3条中途抬手片`3419...`（双手）、`785a...`（右手）、`c1fc...`（左手）的4个抬手事件，最终在参考腕高于肩的帧中实际高于肩比例均0。实际高度变化幅度仅参考的10.46%/13.16%/11.07%/8.45%；参考峰值时实际腕仍比肩低约17.96–18.51cm。`0499...`的左腕峰值在frame0，由参考初始化提供，不能当作主动抬手；其61个目标高于肩帧只保持2帧。峰值时序是描述量，不是完整动作识别器。

只读追溯全部1024份G1源与转换诊断：**1024份均来自`locomotion_candidate`桶**；源G1出现腕高于肩的片数左5/右4，Q1左4/右3。因此上肢大幅动作稀少首先是本轮已选来源的覆盖问题，不能归咎为重定向凭空丢掉大量抬手。Q1与G1肢体尺寸不同，绝对幅度约减半不能直接解释为损失；每片腕映射位置残差p95的片间中位数左2.533cm/右2.537cm，仍需有代表动作才能评价转向/幅度的保真。

证据：[四片上肢分解](../data/manifests/supermimic_q1_RF02_D02_arms_20261025.json)、[初始收据](../data/manifests/supermimic_q1_RF02_D02_arm_initial_20261025.json)、[最终收据](../data/manifests/supermimic_q1_RF02_D02_arm_final_20261025.json)、[来源覆盖摘要](../data/manifests/supermimic_q1_RF02_D02_retarget_20261025.json)。服务器本阶段`arm_{initial,final}/receipt.json`及原始NPZ、`arm_analysis.json`、`retarget_arm_coverage.json`；日志`logs/supermimic_q1_RF02_D02_{arm_initial,arm_final,arms,retarget}_20261025.log`。下一步数据建议是保留原来源身份规则，同时加入上肢/操练动作覆盖；这是任务覆盖，不能成为新增几何硬门。

## 已完成：真实接触与力矩

重放原8条最终策略的2409控制步，保存状态逐值最大差0；native rollout的Data最终状态与对应float32状态误差低于2e-5。读取最后求解子步的真实`mj_contactForce`和`actuator_force`，未用单独forward构造“真实支撑”。几何来自同一求解阶段，较最后积分早一个物理子步；原参考则在其帧上做运动学forward。

片均双脚法向力均超过1N的比例97.86%，双脚均无此力0.87%；全关节饱和1.69%，腿0.51%、臂3.80%。因此已观察到的行为主要是双脚支撑站立，**没有支持“腿部普遍打满力矩导致走不动”的证据**；这些统计也不能证明控制增益/力矩设置最优。

参考足底低于15mm的几何标记与真实1N力标记不一致片均77.29%。两者定义不同，不能称接触标签错误率或原参考真实支撑力；只表明策略的双支撑关系和目标脚几何/速度明显不同。重定向足地时序与原G1同源比较仍是后续可诊断项，尚不能从这项统计断言唯一根因。15mm/1N均为本次描述标记，不恢复被用户取消的数据准入要求。

证据：[接触重放收据](../data/manifests/supermimic_q1_RF02_D02_contacts_20261025.json)，服务器`physics/*_contacts.npz`，日志`logs/supermimic_q1_RF02_D02_contacts_20261025.log`。

## 已完成：模型奖励方向与真实响应

先按原8片最终参考净位移排序固定前3片，分别取可完整重放32步窗口内的25%/65%位置，合计6个访问过的状态；随机方向seed20261025。计划在运行前保存，未按结果重选。每状态先逐值重现原32步，再仅扰动首步动作，后续动作固定；H1/H8/H32各自总奖励梯度方向及1个随机方向，幅度0.02/0.10、正负各一次，实际3264控制步/65280物理子步，恰为计划上限。

每个H上，6状态×2幅度的12个同H梯度方向均与真实中心差分方向一致，正向扰动均提高真实总折扣回报。中位真实增益H1/H8/H32为0.000140/0.000148/0.000132，模型预测增益0.000141/0.000158/0.000176。两个幅度共用状态，不是12个独立状态。

额外只读审计原数组，解析剥离精确动作变化惩罚（包括首步扰动对下一步惩罚的影响），控制其解析导数与中心差分一致。剥离后“6项跟踪+限位”的方向仍各12/12一致，正向扰动也均提高这部分回报；H32状态相关回报中位增益0.0001525。因此当前局部方向一致性不只是动作平滑项的代数效果。

**这不代表世界模型已经会指导行走。** H1/H8/H32的根位置预测误差均值为0.000268/0.003268/0.020016m；沿各自梯度正向扰动后，真实根XY变化中位数仅0.0000187/0.0000476/0.0000705m。局部结果只证明这些已访问站立附近状态的小幅开环收益方向可用，没有核完整策略反馈、critic bootstrap、存活加权或脚支撑切换附近的方向，也没有证明大幅移动/抬手的方向有效。

证据：[冻结计划](../data/manifests/supermimic_q1_RF02_D02_probe_plan_20261025.json)、[方向收据](../data/manifests/supermimic_q1_RF02_D02_directions_20261025.json)、[原数组和动作率剥离审计](../data/manifests/supermimic_q1_RF02_D02_direction_audit_20261025.json)。服务器`physics/sample_00.npz`至`sample_05.npz`；日志`logs/supermimic_q1_RF02_D02_{prepare,directions,audit}_20261025.log`。含动作率跨步依赖的聚焦自动微分/有限差分控制通过，本阶段共4个聚焦单测通过。

## 归因与下一轮单变量建议

1. 已确认：本轮1024份全为移动候选，明显上肢动作稀少；专项抬手没学会，不能以平均关节误差改善代替全身遥操能力。
2. 已确认：策略的稳定站立/平均姿态收益远大于全局位置收益；双脚长期承重、腿力矩饱和少。支持“站立附近的局部解”描述，但尚无改变奖励权重的因果对照。
3. 本次未发现：这6个已访问状态的局部开环总奖励方向失效；不能把过去R03的长程状态方向问题直接当成本轮结论。
4. 尚未排除：原速重定向的足地时序/可实现性、控制尺度、训练覆盖及全闭环长程梯度等问题共同作用。

建议下一新实验先保持SuperMimic算法、奖励、控制参数、网络与时间不变，**只将训练参考范围收缩到已有`train_3419f09ba93c91d75fc1`这一条包含中途双臂抬起的365帧参考**，做单动作可学习性对照。它来自原1024train且短于500步，不新增数据或硬门；评测固定原同片、初始/最终，并报告肩肘、腕躯干相对轨迹、双臂抬起幅度与峰值时序、全局移动/接触和完成度。若单动作仍只学站立，优先查奖励/控制/训练链的条件依赖；若能主动抬手，再单独扩为含上肢/操练的混合参考库。此处是已具名可复现的下一轮建议，**本阶段尚未启动新训练，也不将单动作过拟合当作泛化**。

## 可复现命令

以下在统一根执行，`PY=data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python`，`D=data/supermimic_for_q1/BONES_RF02_D02_fullbody_20261025`。输出为排他创建，复跑请使用新的阶段产物路径，不覆盖本次证据。

```sh
$PY script/supermimic_q1_analysis/diagnose_fullbody.py scan --root /opt/data/private/ZHY --motions data/supermimic_for_q1/BONES_PF02_1024_20261025/library/references --output "$D"
$PY script/supermimic_q1_analysis/diagnose_fullbody.py analyze --root /opt/data/private/ZHY --evaluation initial data/supermimic_for_q1/BONES_BASELINE_G02_20261025/evaluation_initial/receipt.json --evaluation final data/supermimic_for_q1/BONES_BASELINE_G02_20261025/evaluation_final/receipt.json --output "$D/existing_eight_analysis.json"
$PY script/supermimic_q1_analysis/diagnose_fullbody.py retarget --root /opt/data/private/ZHY --motions data/supermimic_for_q1/BONES_PF02_1024_20261025/library/references --output "$D/retarget_arm_coverage.json"
$PY script/supermimic_q1_reproduction/evaluate_reproduction.py --root /opt/data/private/ZHY --motions data/supermimic_for_q1/BONES_PF02_1024_20261025/library/references --checkpoint data/supermimic_for_q1/BONES_RF02_12000_20261025/run/final.pt --count 4 --clips train_0499c71242ed71cdf537 train_785a58b7421497bf9d30 train_c1fc0de44204d0196ac2 train_3419f09ba93c91d75fc1 --output "$D/arm_final" --device cpu
$PY script/supermimic_q1_analysis/probe_reward_directions.py prepare --root /opt/data/private/ZHY --evaluation data/supermimic_for_q1/BONES_BASELINE_G02_20261025/evaluation_final/receipt.json --output "$D/physics"
$PY script/supermimic_q1_analysis/probe_reward_directions.py contacts --root /opt/data/private/ZHY --output "$D/physics"
$PY script/supermimic_q1_analysis/probe_reward_directions.py probe --root /opt/data/private/ZHY --output "$D/physics" --device cuda:0
$PY script/supermimic_q1_analysis/audit_fullbody_directions.py --plan "$D/physics/plan.json" --probe "$D/physics/reward_directions.json" --output "$D/physics/direction_audit.json"
```

上肢初始评测使用完全相同名单，将checkpoint改为`step_000000.pt`、output为`arm_initial`；配对分解沿同一analyze入口，evaluation指向这两份receipt，output为`arm_analysis.json`。单测使用`python -m unittest discover -s script/supermimic_q1_analysis -p test_diagnose_fullbody.py -v`及`test_probe_reward_directions.py`。

## 资源与保存

本阶段总新控制步10483（上肢4810+接触2409+方向3264），物理子步209660，0optimizer更新。服务器本阶段产物实际分配6725632字节（约6.73MB），没有权重或大数据副本。最后观察GPU27摄氏度/14MiB/0%利用率、容器内存22466924544字节、under_oom0/OOM14，盘余1312765497344字节；相关诊断/评测进程已退出。共享盘余量变动不是本次新增工件的大小。大型/原始证据仍为LOCAL_ONLY。

首步代码/结果已发布Q1提交`a87ac965df792a5b1934af1a382e8968ad2ccea4`并核远端；补充上肢/来源/接触/方向代码与收据随本阶段结论提交。小收据与README同步Mac/服务器，完整1024源覆盖数组只留服务器，Git保留摘要、4条选中来源及完整报告身份。

## 保存与下一步

本轮诊断全部完成；下一是具名双臂动作的独立单动作训练对照，不恢复RF02或历史FAILED/STOPPED，不追加RF02预算。现有MP4与训练图保持，诊断不额外建网站或为图像延后报告。
