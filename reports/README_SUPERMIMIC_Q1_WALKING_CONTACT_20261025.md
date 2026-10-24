# PHUMA P12训练步行接触一致性对照

记录日期：2026-10-25，北京时间。状态：P12/P12B两轮共48组IK/inverse及两次独立几何/inverse核验完成；两种接触修正均无准入，原件保留。

## 目标与范围

用户要求继续已有PHUMA、AMASS和SMPL-X资源；BONES由师兄处理。本阶段以[P11](README_SUPERMIMIC_Q1_WALKING_RETARGET_20261025.md)相同四个训练walking来源，比较支撑脚目标修正及Mink原生避碰。仍复现SuperMimic世界模型、想象BPTT、critic；这里是离线参考适配诊断，不增加强化学习方法。P10的15个目录继续预留test，不用于参数选择。唯一R02D训练保持原340输入、7000新增轮和516冻结。

## 方案与边界

新脚本`script/supermimic_q1_sources/{walking_contact_targets,probe_walking_contact,test_walking_contact_targets}.py`均在原冻结集合之外。重新连接原选择/源索引/预留目录并要求与P11逐项相同；原片、时轴、根位置/方向和关节位置限位不变。2与8rad/s分别比较baseline、stance、stance_collision，共预定24组；baseline重新解算且须与P11保存误差小于1e-8。8rad/s仅诊断，仍不满足原2rad/s训练准入。

接触是假设：目标足底相对较低脚15mm内视为支撑，闭合孤立一帧缺口，去除短于4帧的内部片段。只适用于固定walking训练样本，不能当接触真值。支撑期固定踝部位置、以校准脚姿态保留航向并展平；两个支撑段之间用SciPy五阶Hermite位置修正和旋转插值连接。各脚锚点独立，根轨迹保持；所有变体使用由对应baseline整片最低足底确定的同一个常数地面原点。没有逐帧投影根高。

Mink沿用原全身任务权重、位置/速度限位与硬根约束；stance改变脚任务目标，stance_collision另加腕部到躯干/腿部原生距离约束（2mm余隙、50mm激活）。它不是全身碰撞认证，最终仍查全部模型自穿透。安装版本Mink1.2.0的距离上界含多余`/dt`；已读现场函数并核对[上游问题752](https://github.com/kevinzakka/mink/issues/752)与[修正658](https://github.com/kevinzakka/mink/pull/658)。仅本诊断实例使用`gain=.85*dt`，使位移QP上界为`.85*(distance-.002)`；不改环境或冻结代码。两个时间步的真实原生不等式控制通过前不能执行真实对照。

保存全部qpos/qvel、原/新任务目标、误差、支撑标签、全帧脚底及穿透。复用既有MuJoCo脚底点Jacobian、三/五点差分及inverse/RNE分解，对比接触、近地滑动、根残余外力和关节超cap。只做FK/IK/逆动力学，0积分/策略推理/optimizer；完整十九门与真实动力学跟随尚待，所有输出无训练资格。

## 资源、版本与命令

Git权威副本`/tmp/yuanqi-supermimic-q1-20261023`，分支Q1；服务器`huanyu:/opt/data/private/YuanQi_Q1`。代码/报告审计发布并独立核远端后逐SHA部署，不编辑原516冻结文件。单CPU/nice10/无CUDA，28.75GiB/OOM14、100GiB余盘、128MiB输出、900秒无进展停止及自身worker回收；无自动重启。

```bash
ROOT=/opt/data/private/YuanQi_Q1
PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -u script/supermimic_q1_sources/probe_walking_contact.py \
  --root "$ROOT" \
  --output "$ROOT/data/supermimic_for_q1/PHUMA_P12_walking_contact_probe_20261025" \
  --expected-oom-kill 14
```

日志为同名`logs/`，guard.json在实验根，selection/receipt及NPZ在probe/，inverse数组在probe/dynamics/。完成后独立重算保存数组统计，不把相同helper重复调用当独立证据。原始模型/数组仍服务器LOCAL_ONLY；Mac只取小收据，代码/重要结果按Q1规范发布。

## 结果与下一步

本地三项来源/标签控制通过、三份新代码语法通过；Mac没有Mink/MuJoCo/SciPy，未执行本地真实几何控制。服务器完整五项控制及实际24组对照待。任何求解失败保留该变体FAILED及原因，诊断完成不等于全部变体成功。先查支撑/滑动是否改善且未增大原任务误差、穿透或动力学残差；不因近地比例改善自动合库。R02D原预算继续，目标未见步行动力学跟随尚未达成。

## 06:21真实依赖控制通过

准备代码发布`3259124bf9f5b3232565172021834fdfd511c10b`已推送并独立核Q1远端。三份新文件部署前均不存在，且不属于516冻结；部署后SHA分别为helper`448b5042c02d12851ae6d2221bc09da7d9ac77620a280cee3bc7fa1a941af142`、producer`8e4f57014fbae5b593680dfc6cd176efac781699c5680c75e25935929aecd5b3`、test`ec0eb30580ca5ce908d91e5b78a6a62c71189d29c5dbfc809074a6d950b4a3cd`，与发布一致。

服务器`python -m unittest test_walking_contact_targets -v`原生退出0，五项全过/0跳过：来源角色/目录排除、相对高度不变性、非法输入拒绝、支撑锚点/摆动连接/原数组不变、原生Mink两个dt的位移不等式实际数值。日志`logs/PHUMA_P12_contact_controls_20261025.log`，SHA`ca24fa860eba785f96def040c45f2c98739f6554e640e309ed719d8f24d6476b`。这不是四片真实对照或准入结果。

06:19训练实读2475新增轮/158400转移，全部516冻结重新核SHA保持，OOM14/内存约14.15GB、盘余约655.86GB。共享盘继续下降，当前独立诊断维持128MiB上限及100GiB余量，不删除未知文件。下一实际新目录P12，不改R02D原预算/输入。

## 06:22首轮完成但方案拒绝

06:21:32至06:22:00原生退出0，24组均完成（四来源、两速度、三变体）；guard842934/worker842935已回收，0CUDA/积分/策略/optimizer，新参考0。八个baseline逐qpos复现P11，原输入前后SHA保持。实验分配17,805,312字节，结束共享内存13,785,378,816/OOM14、盘余647,097,020,416字节。原样[receipt](../data/manifests/supermimic_q1_P12_contact_receipt_20261025.json)304495字节/SHA`ad5cc6e367f507899d96d1e05f1e713747469e6967a24ee422aaa30b77d90fa8`和[guard](../data/manifests/supermimic_q1_P12_contact_guard_20261025.json)保留。

仅按两脚相对高度推断支撑过宽：每脚81.41%至100%被标记站立，fast02甚至双脚整片同时站立。固定世界锚点与向前根轨迹冲突，脚任务最大改动0.38至1.04m，明显失真；8rad/s下三片原任务P95由17.9至21.1mm升为165至357mm。所有修正变体的支撑/穿地/自碰撞子集均未通过，不能导出为训练参考。

8rad/s较低脚底15mm内比例：Circle06由64.32%降31.66%、fast02由43.02%降11.73%、Circle02由61.97%降36.62%、slow02由18.59%降2.51%。slow02在2rad/s下约11mm穿地。此失败属于本次接触假设/目标构造，不是正在运行的R02D失败。

原生避碰启用后，全部八组stance_collision的模型自穿透最大值为0；无避碰stance六组有6.68至18.20mm穿透。这个小样本结果支持保留避碰方向，但不能证明全部动作可避碰或动力学可行，且它未修复落地。根残余外力中位数仍约一倍体重。下一先独立FK/差分/数组重算，再在同四个train中加入固定脚水平速度条件，避免把低高度摆动当站立；保留首轮原件/生产者，不以P10测试调参。

新增独立`audit_walking_contact.py`不调用原IK/producer统计；重读原数组并使用独立脚底查询、新MuJoCo状态逐帧FK和inverse，核全部速度/三五点加速度、任务误差、足底、穿透及关键结果。原始生产者/helper保持不变；审计发布部署后实际结果待。

## 06:27独立几何与inverse核验

审计发布`a7f85730514bb5bba370f8ad58c9e35ac273f075`已独立核Q1远端；auditor部署SHA`c7f1e14d00350630e4db40321141fcd16c786ab448d2b6040ea7f06c9298fb91`一致。独立CPU进程原生退出0，24组共8940项数值核验通过，重新运行8436次MuJoCo inverse，所有保存数组与原输入前后SHA保持；原根轨迹、非足任务、时轴、速度/加速度、逐帧足底/穿透及任务残差已核。未重新运行IK/策略或积分，不把统计/逆动力学一致提升为跟随效果。

[原样独立收据](../data/manifests/supermimic_q1_P12_contact_audit_20261025.json)6626字节/SHA`590c0f01038c34b782e6b014ef2b56fb21c011593ec84cbf11f648ef38c2ed6a`，服务器原件为实验根`independent_audit.json`，stdout`logs/PHUMA_P12_contact_audit_20261025.log`。高度单判方案的失败结论保持，所有输出无训练资格。

## P12B速度条件准备

新增薄适配器`probe_walking_contact_velocity.py`及三项标签控制，保留原P12生产者/helper/原件。复用同一发动程序、四个train、两种限速、三个变体及全部任务/物理设置，唯一新增标签条件为映射踝部世界XY速度不大于0.15m/s，再去短于4帧的内部支撑段；不强行补足无接触帧。速度来自原50Hz目标的二阶边界/中央差分，阈值预定、不按测试选型。

适配器显式替换本worker的接触目标回调，只修改本地推断标签，不改原任务/源数组；选择与最终receipt记录实际hook、适配器/控制/基生产者SHA和有效标签规则，输入前后绑定。若某脚没有支撑或求解失败，保留真实失败，不补虚假接触。原输出继续无资格，8rad/s仍不符合原速度门。

本地三项速度标签控制及两份新代码语法通过。代码发布/服务器标签控制后，以上命令只将入口换为`probe_walking_contact_velocity.py`、新目录换为`PHUMA_P12B_walking_contact_velocity_probe_20261025`，其余预算、无CUDA和停止保护不变。真实P12B与独立auditor执行待；不改R02D或预留P10。

P12B准备发布`58fc9f8c13157e4a9f0c0798580f8e07aaf2ee82`已推送并独立核Q1。两新文件部署前不存在，部署SHA适配器`9e9136a72bf7ef3a683f63e4de228ff63a1fb8bde7066ff4211fac8256c50970`、test`884f1a79c9968cb4495521e76c710b9f7281179a976ac49925aa17606839b7dd`与发布一致。服务器原五项加新三项共八控制原生退出0、0跳过，日志`logs/PHUMA_P12B_velocity_contact_controls_20261025.log`/SHA`a598801b141f0ad63c6b83da4f24c704ad9727ea7afcf7d8c3425fd47e10e5e8`；真实P12B尚待。

P12及P12B准备阶段13份已提交文件共423250字节机械同步用户恢复镜像，复制前逐份比对旧基线/新内容，复制后逐字节一致。未复制BONES脚本或大型资产。06:32实读R02D2800新增轮、179200转移，OOM14/数值有限，盘余约591.72GB；仍按原停止线运行。

## P12B实际完成与独立核验

06:34:50至06:35:28 guard843514/worker843515原生退出0并回收，四来源/24组全部完成；实际速度适配hook已记入选择与receipt并核True，基producer、适配器/test及全部输入前后SHA保持。实验分配18,071,552字节，结束内存13,805,686,784/OOM14、盘余578,629,177,344字节。原样[receipt](../data/manifests/supermimic_q1_P12B_contact_receipt_20261025.json)、[guard](../data/manifests/supermimic_q1_P12B_contact_guard_20261025.json)已取回核对；不改原P12。

每脚推断支撑比例降至18.59%至69.01%，脚任务最大改动缩小到72至156mm；原任务P95保持在约20至40mm，未出现首轮165至357mm的严重失真。但全部16个修正变体仍未通过支撑/穿地/自碰撞子集，0新准入。8rad/s加避碰结果如下（比例单侧统计会把穿地也计入，必须连同最小高度看）：

| 固定train事件 | baseline15mm内 | P12B15mm内 | P12B最低脚底 | P12B近地滑动比例 | P12B超cap帧比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Circle06 | 64.32% | 83.42% | -22.13mm | 91.96% | 25.13% |
| walking_fast02 | 43.02% | 43.02% | -8.09mm | 100% | 7.43% |
| Circle02 | 61.97% | 63.38% | -17.64mm | 96.36% | 16.67% |
| walking_slow02 | 18.59% | 21.11% | -4.61mm | 100% | 2.05% |

近地滑动按脚底10mm内的点速度超过0.05m/s；超cap为中央三点inverse的任一关节超原cap帧比例，不是真机测试。全部八个加避碰变体自穿透0，但浮脚/穿地和高滑动仍在，根残余外力中位数约1.000至1.051倍体重。这些结果不支持把“近地比例增加”描述成有效动作参考或策略改善。

原独立auditor未改，针对新P12B另起CPU进程原生退出0，再次8940项数值核验和8436次fresh inverse全过；不是重复IK或动力学rollout。[原样审计](../data/manifests/supermimic_q1_P12B_contact_audit_20261025.json)6636字节/SHA`c97df012fc64e0d60c59aa47e5e5d1f9bb8c5635651a94daf5cd24515809af47`，服务器实验根`independent_audit.json`，stdout`logs/PHUMA_P12B_contact_audit_20261025.log`。两轮诊断所有大型原件LOCAL_ONLY。

决定：不采用上述两种锁脚修正，不放宽原十九门。后续若继续接触适配，必须在同train小样本把足地不穿透和可达性放进原生求解约束，并验证摆动/支撑转换与实际动力学代价；不能继续只靠软目标贴地。R02D原340/7000预算与终态规则保持，最终353片配对尚待。

## P12C原生足地约束准备

新增`probe_walking_contact_floor.py`和原生几何控制，复用P12B速度标签/目标及相同四个train、2/8rad/s和三个变体。baseline继续复现原P11；两个修正变体均加原生足地距离下界0、激活50mm，并将踝部任务位置/方向代价20/1提高至200/10。加避碰变体另保留原腕部2mm约束。这是足地约束与脚任务权重的联合候选，不能当纯单变量因果消融。

每个solver的平面只按对应baseline地面原点做同一个常数坐标平移；根位置/方向和原速不变，不做逐帧根高投影。初始静态拟合也执行足地约束；时序帧保留硬根、关节位置/速度限位。Mink采用已控制的dt缩放，仅约束每次QP的一阶位移，最终几何仍可能失败，必须查全帧实际足底/自碰撞和fresh inverse，不能因为使用约束就认定不穿透。

新薄适配器在选择和receipt显式登记源/test/有效足代价/实际solve hook，与P12B的实际标签hook同时绑定，保留全部原生产者、旧输出和516训练冻结。两份新代码本地语法通过；Mac无Mink/MuJoCo，真实控制和24组结果待。服务器先运行原生双脚到平面控制，覆盖0/-40mm两种平面原点和每种30次向地内目标求解；通过后入口换为`probe_walking_contact_floor.py`，新目录`PHUMA_P12C_walking_floor_probe_20261025`。预算、无CUDA/无积分/0训练资格等保持。

## P12C控制A测试层级错误

准备发布`b24284fd0cc692a4188915bf87fef285fb812ed4`已独立核Q1远端，新两文件部署SHA一致且不在516冻结。首次服务器控制原生退出1，原八项通过，新足地控制ERROR：测试将两只脚直接作为world子刚体，Mink的父子刚体过滤删除了这两组；现场读取`_are_geom_bodies_parent_child`确认该规则。它没有执行真实P12C转换，未创建其输出目录，不能称真实Q1足地失败。

原生[失败日志A](../logs/session_records/supermimic_q1_P12C_floor_controls_A_20261025.log)保留，服务器原件`logs/PHUMA_P12C_floor_controls_20261025.log`/SHA`90b435470789da569c450629fda533be5a17ab3880e1639d7ff3d8374188c177`。修正只把测试骨架改成与Q1相符的浮动骨盆/子足关节并增加硬根约束，另查真实Q1的十份足部几何均获得原生平面配对；不改Mink、P12C生产者或原训练。修正后新B日志/真实对照待。
