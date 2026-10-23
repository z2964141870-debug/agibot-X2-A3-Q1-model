# SuperMimic for Q1

记录日期：2026-10-23，北京时间。阶段 SM-Q1-P01–P04/R01。状态：模型/物理/梯度通过，三段参考独立质检通过、过头动作拒绝；R01训练正在运行，尚无完整效果结论。

## 目标与范围

用户要求停止 A3 路线的当前推进，将同一项 SuperMimic 复现工作改为 Q1，训练 `supermimic_for_q1`。继续遵循世界模型路线、只做方法复现的约束。之前未发布的 A3 目标草案不再作为当前执行目标；已停止的 A3 R05 及其他历史实验保持原终态。

目标：在 Q1 浮动基座、无人工骨盆支撑的物理仿真中，从随机初始化训练一个参考动作条件策略，执行物理采样 → 可微世界模型拟合 → 模型内策略反向传播 → critic 更新的闭环，交付固定版本的源码、参考数据身份、checkpoint、完整动作评测和可复跑命令。先完成少量基础动作的真实训练闭环，再扩展动作覆盖。

Q1 动作集和指标以 Q1 数据及物理模型的实际核验为依据重新登记，不能把 A3 的20段关节文件或旧分数直接改名使用。候选工程门为完整片段无提前失败、完整片段归一化回报至少0.5、主动关节RMSE不超过0.15rad；这些是拟定的项目验收目标，不是论文原始阈值或已取得的结果。训练动作覆盖、未见动作泛化、真机验证分别报告。

## 方法与实现约束

保留 SuperMimic 的当前策略物理采样、可微确定性世界模型、k=8自回归拟合、H=32模型内回报反传、survival weighting、critic bootstrap与GAE。策略更新冻结世界模型参数但保留状态/动作梯度。主结果不加载 A3/G1/035 权重，不以PPO、行为克隆或教师蒸馏替代世界模型策略学习。

固定论文/已审查独立MIT复现与本工程既有实现进行逐项比对；必要的Q1尺寸、观测、关节/身体映射、动作幅度、PD、碰撞与终止尺度适配登记为机器人适配。本项目只主张Q1方法复现，不主张论文G1全量数据、H100耗时或全部成绩的数值复现。

## 资源与已知问题

- 本次用户附件在 `/Users/yu/Downloads/`。Q1模型与两工具链ZIP的SHA与10/8记录一致；SDK SHA由旧857eb034…变为ebee4463…，ZIP来源提交注释由14a85d1…变为87ba32c…，需按新版本核验。
- Q1提供22DOF URDF/MJCF和网格；SDK与两个Docker工具链提供控制/构建资源，不等同SuperMimic训练器或动作数据。
- 旧Q1分支已有MuJoCo关节控制、Mink/GMR/UMR重定向和测试。旧动作存在校准、穿地、速度或保真问题，均不能直接提升为合格训练参考。旧人工支撑PD不是平衡策略。
- 旧hp3090 Q1目录仅作历史来源，本次SSH连接超时，不在原机恢复训练。沿用本聊天现有训练平台huanyu，在独立 `/opt/data/private/YuanQi_Q1` 下组织 `data / logs / script / reports`；复用既有Python环境按只读依赖处理，不修改A3冻结源码/数据。
- 实核时huanyu RTX3090 24576MiB，显存6348MiB/利用率0%/38摄氏度，磁盘约1.9TiB可用；均为当时快照。`pgrep -af '[s]upermimic_for_'`无匹配。新Q1加载/梯度/训练尚未通过；不沿用旧机器或A3的通过收据。
- 原A3转换曾把源120Hz当30Hz，产生四倍慢放；Q1转换必须明确源时间、单位、四元数、重力轴及身体语义，不能继承该错误。

## 执行顺序

1. 核对四附件内容与身份，按需迁移服务器，独立大小/SHA验证；模型/第三方原件不入Git或Mac新增解压副本。
2. 建立Q1全状态/22关节映射、实际碰撞与PD协议；验证可微FK对MuJoCo、真实控制间隔和梯度合同。
3. 建立原速Q1参考与独立质检，先固定少量基础动作及评测分母。人体/其他机器人动作先重定向，在线Q1物理转移单独用于世界模型。
4. 固定源码/配置/输入、预算、资源和停止保护后，从随机初始化启动独立Q1训练；保存step0和最终模型，新进程完整片段对照评测。

## 版本、证据与保存

P01实际结果：4份ZIP共34,995,419字节已迁移至新Q1根；服务器独立大小/SHA均匹配。24个STL和3个MJCF完整，全部原件保持不变。三个模型均为nq29/nv28，展示版与full_range无执行器，只有mc版nu22，控制固定选mc版。首次检查器误将3份XML均要求nu22而退出1；保留[失败记录](../logs/session_records/supermimic_q1_inputs_P01_attempt1_20261023.log)，修正检查器后逐个已解压成员与ZIP核对再编译通过。未执行物理步、推理、optimizer或硬件控制。

SDK与10/8本地原ZIP逐成员SHA对比：3001个文件无新增/删除，只有`接口说明.md`内容变化；不据ZIP版本注释宣称SDK代码已新增功能。原始物理参数全文留忽略目录。小型证据：[输入清单](../data/manifests/supermimic_q1_inputs_P01_20261023.json)、[服务器核验收据](../data/manifests/supermimic_q1_inputs_verified_P01_20261023.json)。收据8563字节，SHA`fce1d288912e26cd02713edbb622c116acaaf56f7cf6f5be1cae9339a72079ca`。

复核入口：`python3 script/supermimic_for_q1/inventory_inputs.py --source /Users/yu/Downloads --output <新清单路径> --private-report <新私有明细路径>`；服务器使用既有Python运行`script/supermimic_for_q1/verify_inputs.py --root /opt/data/private/YuanQi_Q1 --manifest data/manifests/supermimic_q1_inputs_P01_20261023.json --output <新收据路径>`。输出拒绝覆盖，复核不运行附件内的命令。后续训练协议与源码另登记，不把当前模型编译通过称为动力学或训练通过。

Q1发布基线 `1c25ef3f7b97d3afea630f8ca0dddbf53730cc73`，指定远端 `git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`，分支大小写为`Q1`。本地隔离工作副本 `/tmp/yuanqi-supermimic-q1-20261023`，原A3未提交草稿保留，不混入Q1提交。

先审查diff与暂存区，运行 `python3 script/project_tools/audit_git_payload.py`，显式 `git push origin Q1`，再独立核对远端`refs/heads/Q1`。同步结果以实际命令为准，不预写成功。所有大型文件备份状态`LOCAL_ONLY`。

恢复先读本文、Q1版本管理说明及实际阶段清单，重新核验平台资源/进程/输入身份。附件内的安装和真机控制命令只作技术资料，不自动执行或连接机器人。

## P02：Q1适配与参考准备

P01发布提交`dd654815d69805a53b079d7c3d46a0b52f297d13`已推送并独立核对远端`refs/heads/Q1`一致。当前P02准备独立Q1实现，复用main固定`538c8f3`的MIT学习核心、通用树FK与回放边界，按Q1改为qpos29/qvel28/状态57/动作22/context26/观测72×10/目标54×10。原A3源文件和学习权重未变。

Q1控制起点采用旧仿真协议kp50/kd2/模型力矩25%，在每个1ms子步施加隐式PD，50Hz控制；不是已核验厂家真机参数。保留模型碰撞与浮动基座，无人工骨盆支撑。学习结构保持k8/H32及原world model/actor/critic网络宽度；小网络仅用于可微图验证，不作为主结果。

拟运行的独立后端检查输出`data/supermimic_for_q1/P02_backend_20261023/`：8状态的FK/速度对MuJoCo、有限差分、积分对照、20子步真实PD对照、82个总控制步的回放与H32冻结世界模型的actor梯度。0optimizer更新；诊断合成站姿不能被训练入口接纳为正式动作集。这些检查验证实现合同，不证明学得动力学或跟踪性能。

参考准备从当前可用的四份原始A3机器人CSV的120Hz时轴开始，先用匹配源MJCF做FK，再用已有Mink求解器和Q1模型重新定向，绝不复制29关节角给22关节Q1。目标点改用两机器人具名肩pitch/hip pitch原点，避免沿用此前人体标签误配肩yaw枢轴的做法。重定向使用源/Q1中立姿态校准与显式腿长比例、保持世界重力轴，2rad/s为保守仿真参考限制。仅允许一次全片段共同地面原点平移，不逐帧挪根或减速。先记录完整四段的时长、关节/脚底/笛卡尔残差与失败，不按分数删动作。输出NPZ和IK诊断保存在服务器，质量门通过后才登记训练。

候选参考的运动学预检门：原120Hz时间身份正确、50Hz重采样时长误差小于一个控制间隔、有限状态/四元数、关节原始限位、配置2rad/s、IK位置残差P95≤5cm、共同地面原点调整≤10cm、足碰撞不穿地且基础动作至少半数帧有足碰撞面接近地面3cm内、根高度>15cm。门定义在代码与清单固定，不把其通过称为物理可执行或人体同姿精度通过。旧20段A3结果和旧Q1 TEST_ONLY轨迹不作为本次通过证据。

P02首轮实际后端检查保留为`P02_backend_20261023/receipt.json`：已执行82控制步/1640物理子步、0optimizer；FK位置/速度/朝向最大误差约1.5e-15，方向梯度差1.52e-10，恒速积分差2.22e-16，原生rollout对20子步显式控制状态差1.39e-8。随后H32测试fixture把double状态与float参考混用而失败，未证明H32通过；修复仅使图验证参考同为float64，新独立输出使用`P02_backend_B_20261023/`，原失败及源身份保留。服务器没有rsync，第一次同步退出127，随后的入口因文件尚未到位退出2；已改用SCP且完整复制退出0，无依赖安装或附件脚本执行。

### P02B实际后端结果与P02A参考失败检查点

`P02_backend_B_20261023/receipt.json`已完成，状态`BACKEND_AND_GRADIENT_CONTRACT_PASSED`。FK、积分与原生/显式PD差异与上轮同量级；H32动作方向梯度误差2.684e-12，actor末端状态梯度范数2.039e-4且世界模型参数梯度为0。两轮合计164控制步/3280物理子步、0optimizer。图检查使用64宽fixture，不能据此宣称默认宽度训练或学得世界模型准确。收据见[首轮](../data/manifests/supermimic_q1_backend_P02A_20261023.json)及[修复轮](../data/manifests/supermimic_q1_backend_P02B_20261023.json)。

原速参考首轮`motions_P02A_20261023`实际退出2，状态`REFERENCE_QUALITY_FAILED`，0/4合格，不能用于启动训练。结果如下：

| 原始120Hz片段 | IK位置残差P95 | 全片共同地面平移 | 失败项 |
| --- | --- | --- | --- |
| `001_walk_front_slow` | 2.097cm | 12.138cm | 地面原点平移超过拟定10cm |
| `042_squat_shallow_repeated` | 2.113cm | 10.234cm | 同上 |
| `058_punch_front_alternating` | 4.174cm | 10.069cm | 同上 |
| `077_reach_overhead_both` | 5.496cm | 10.461cm | 地面原点与IK残差 |

四段其余已配置数值/限位/2rad/s/足碰撞地面/根高度门均通过；这不证明动力学可跟踪。服务器原始manifest为18285字节、SHA`ac303012ef83c000334bf2e110eefde0dcb69d2fb1cc606602d047835619f0f1`，路径`data/supermimic_for_q1/motions_P02A_20261023/manifest.json`，日志`logs/supermimic_q1_motions_P02A_20261023.log`，原始IK诊断在同产物目录`diagnostics/`。保留全部失败输出。

下一条工作是独立核对源/Q1中立根高和足底几何、逐身体残差，定位共同约10cm地面偏移和过头动作误差。根因尚未证实；上述门是本项目拟定标准，不能静默放宽或把失败输出改成通过。修正转换须使用新协议/新输出目录，之后独立检查时间、FK、速度与几何再冻结训练输入。当前训练/evaluate入口只完成语法检查，未实际运行；没有Q1 checkpoint。

### P02D只读诊断与P03修正协议

P02代码、backend两轮及参考失败已发布至`Q1`提交`dca6fd01362ae0367a4e02ec314fe7ebe8834461`，独立`ls-remote`一致。诊断`diagnose_reference.py`实际退出0，0物理步/optimizer；原始IK误差逐项独立FK重算差0，根目标公式最大差约1e-17。源默认根高1.3m而足碰撞最低点0.2250673763m，Q1默认根高0.419m而足底-0.0016113788m。源动作实际根高约1.06m；旧标定把源悬空的默认姿态当落地姿态，将约24cm的源原点差乘0.38357584后压入Q1根高，是约10cm共同地面偏移的主要原因。诊断收据`data/supermimic_for_q1/P02_reference_diagnostic_20261023.json`，33591字节、SHA`ba693fb8dea21fced87258ac5a8c386a6e01f1c5f66f590e3ee09202bca6b20a`，小副本见[诊断](../data/manifests/supermimic_q1_reference_diagnostic_P02_20261023.json)。

过头动作误差独立存在：右肘/右腕P95约6.58/8.73cm，非地面平移所致；当前2rad/s受限IK也有初始化/追踪延迟，不能把根高修复描述成全部保真问题解决。各片段、各身体和旧门结果全部保留。

P03协议在新目录中仅纠正源标定姿态的固定地面原点与Q1中立足底原点，不更改原120Hz时间、50Hz输出、2rad/s限制、IK目标公式、5cm残差或10cm末端地面调整门。不重跑旧输出。生成完整四段及逐段通过/失败后，新增独立质检。首轮训练可明确只接纳经验证的合格参考，任何未通过片段继续列入拒绝清单；这属于训练原料筛选，不报告四段全合格或未见泛化。预选走路/浅蹲用于训练、出拳用于开发评测；过头动作在修复前不进入训练或可用性分母，单独报告其输入不合格。上述划分在P03新结果前固定，不能按策略效果挑选。

### P03结果与R01训练登记

P03四段转换退出2，状态`KINEMATIC_REFERENCE_PARTIAL`；前三段通过，过头动作仍因IK残差5.496cm被拒绝。共同末端地面调整分别3.344/1.440/1.275/1.667cm。此修正改变中立标定原点，最终落地轨迹与P02A几乎相同，不能宣称动作保真提升。manifest SHA`ce72f44b2f653043784044b9c1c6d3911735eb87014d9ef380f588a95714eb9c`。独立`verify_reference.py`完成，前三段逐项通过，最后一段同样拒绝；时长损失均小于0.02s，FK导出最大差3.40e-7，qvel独立重算差小于1.17e-5。收据7427字节/SHA`832b09da7997558216f5bb4adeb10ddc5dbc7201ef22b0f4a8683dc68271c497`。两份小清单保存在`data/manifests/supermimic_q1_{motions,reference_verified}_P03_20261023.json`。

训练前动作覆盖检查发现原Q1动作缩放错误地以已乘25%试验力矩的effort再次计算0.25*effort/kp：手臂目标范围仅±0.55rad，走路/浅蹲约10.7%/10.5%关节帧超范围，最大需归一化动作43.84/42.62，超过clip20。P04将归一化比例定义为0.25*模型完整力矩/kp，腿0.18/手臂0.11rad；实际力矩仍限制模型25%，kp50/kd2不变。这样三合格动作所需最大动作分别10.96/10.66/11.69，均覆盖。此为仿真接口适配，非厂家参数或真机验证，原P02B收据不覆盖新scale，须新后端检查。

R01预算预登记：固定走路/浅蹲训练、出拳开发、过头输入拒绝；随机初始化，16环境/4线程，3000采集轮=48000新控制步，k8世界模型每组10次batch64，H32策略batch32。128采集轮后以0.4更新组/采集轮执行，先50组世界模型预热，再联合actor/critic；预计1149组WM/1099次actor、11490次WM Adam和21980次critic Adam，实际计数以checkpoint优化器状态审计为准。保存step0及500/1000/1500/2000/2500/3000与final，主网络宽度保持；不compile、不加载A3权重、不用教师/行为克隆替代模型内学习。

启动前重新核对GPU/内存/OOM/作业和输入/源码身份，冻结发布提交作为`--code-commit`。GPU>=86°C、容器内存>=30GiB、新OOM计数、非有限物理/损失或异常触发停止；不自动恢复失败轮。训练输出拟为`data/supermimic_for_q1/R01_20261023/`，日志`logs/supermimic_q1_R01_20261023.log`。结束后新进程分别对step0/final执行三合格片段从首帧开始的完整评测，提前失败按整片分母计入；独立核对权重变化、优化器计数及评测轨迹。尚未启动R01，不把预算写成实际完成。

P04新后端实际退出0，`BACKEND_AND_GRADIENT_CONTRACT_PASSED`；新动作scale下原生/显式PD最大差7.668e-9，H32动作方向梯度误差4.075e-13、actor梯度非零且WM参数无梯度。新增82控制步/1640子步，Q1累计246控制步/4920子步，仍0optimizer。训练将绑定此新收据`data/supermimic_for_q1/P04_backend_20261023/receipt.json`，不是P02B旧版本。现有阶段产物全部`LOCAL_ONLY`。

可复现P03/P04命令（在服务器Q1根，`PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`，所有输出须为新路径）：

```bash
$PY script/supermimic_for_q1/prepare_motion.py --root "$PWD" --source-vendor /opt/data/private/hechengzhu/ZHY/script/vendor/sonic_for_a3 --source-hz 120 --clips 001_walk_front_slow 042_squat_shallow_repeated 058_punch_front_alternating 077_reach_overhead_both --output data/supermimic_for_q1/motions_P03_20261023
$PY script/supermimic_for_q1/verify_reference.py --root "$PWD" --motions data/supermimic_for_q1/motions_P03_20261023 --output data/supermimic_for_q1/P03_independent_reference_20261023.json
$PY script/supermimic_for_q1/verify_backend.py --root "$PWD" --output data/supermimic_for_q1/P04_backend_20261023
```

### R01运行检查点

代码/输入及P03/P04记录已发布为`cc3ecb61f8ab00cab112ed52a6c40147d30f5eb8`，独立远端`refs/heads/Q1`一致。Git archive部署后，服务器train入口及资产适配SHA与本地相同，backend入口再次在训练前核对全部冻结模块。R01实际启动，native PID735033；250采集轮/4000控制步时49组WM/0actor，GPU40°C，容器OOM仍14（历史计数），当时内存约28.4GiB。此为运行快照，不代表完成或有效策略。进程、资源与结果继续以服务器`R01_20261023/status.json`及日志为准。

实际启动命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 $PY -u script/supermimic_for_q1/train.py --root "$PWD" --motions data/supermimic_for_q1/motions_P03_20261023 --output data/supermimic_for_q1/R01_20261023 --backend-receipt data/supermimic_for_q1/P04_backend_20261023/receipt.json --reference-receipt data/supermimic_for_q1/P03_independent_reference_20261023.json --steps 3000 --num-envs 16 --threads 4 --batch 32 --wm-batch 64 --wm-batches 10 --warmup-steps 128 --model-warmup-updates 50 --checkpoint-every 500 --seed 20261023 --device cuda:0 --clips 001_walk_front_slow 042_squat_shallow_repeated --code-commit cc3ecb61f8ab00cab112ed52a6c40147d30f5eb8
```

### R01完成核验与诊断规格

训练仍在原作业进行，未改预算/参数/学习模块。独立保存核验、评测重算和渲染工具已在`19b3b73fc7be99ac96d52a481c1d334f510a5bd8`发布，远端独立核验一致；这些工具尚未完成实际R01运行，不能记为结果。后续新增的只读脚本不改训练已冻结文件。

完成后依序运行：`audit_training.py`读取step0/final及原receipt，独立核对预算、初始空优化器、各网络有限且变化、WM/actor/critic实际Adam计数和源码/输入SHA；`evaluate.py`分别评测step0与final的三合格整片，失败后的未执行帧保留在分母中；`audit_evaluation.py`从保存state/action/参考重新计算奖励、关节/身体误差、终止和三个可用性门。核验通过与可用性通过分别报告。

世界模型诊断使用final评测已保存轨迹，固定H=1/8/32、stride8，给定相同真实初态、执行动作及无随机化context，对比已学模型与零残差恒速积分。报告根位置逐坐标RMSE、根旋转角、关节角、分项速度和归一化状态误差；不跨已终止片段、不生成新物理数据。窗口重叠且复用开发评测，不称独立最终测试或统计泛化证据。此规格在看到R01完整效果前固定，预计输出`data/supermimic_for_q1/R01_world_model_20261023.json`。

`plot_training.py`绘制采样回报/关节误差和在线batch模型误差，标明分布随策略变化，不能当整片成绩；输出拟为`R01_curves_20261023/`。`render_trace.py`只渲染保存轨迹与对应参考，共用相机、显示提前终止与整片分母；0新推理/物理/optimizer，实际图像还需目视核验。所有结果仍待R01训练结束后实际执行。
