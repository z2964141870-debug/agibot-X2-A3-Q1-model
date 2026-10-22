# SuperMimic for A3：论文、实现与训练

记录日期：2026-10-23（北京时间）。阶段：SM-A3-P01/P02/R01/R02/C01/R03。状态：R02八段全提前失败；C01完成且独立验收通过，长视野根误差降低50.60%。R03已实际启动，尚待预算完成与完整物理评测，尚非可用跟踪策略。

## 目标与范围

用户明确要求阅读SuperMimic论文、理解思想并训练`supermimic_for_a3`；代码、配置、阶段结论和小清单维护在`z2964141870-debug/agibot-X2-A3-Q1-model`的main。该新任务授权独立训练，不恢复历史STOPPED试验。当前计算主机是用户10/22指定的huanyu（59.77.15.15:25893），工作根`/opt/data/private/yuanqi_ops`；历史ZHY目录仅复用只读A3输入。

## 论文思想与实现含义

实际读取arXiv:2610.28480v1全文19页，并渲染核对第5页公式1–3及第16页超参数表。PDF大小3,524,904字节，SHA-256 `fe213017ab062f4acc3965da89abd3c29969163e144a2628a662b8ce46691009`。原始来源为https://arxiv.org/pdf/2610.28480v1 与https://supermimic.github.io/；web工具未返回可用正文，实际采用直接HTTP下载及PDF文本/页面核对。

1. **共享动力学而非逐动作学习后果。** 一个确定性的世界模型输入完整机器人状态、执行动作及摩擦/编码器偏差/CoM上下文；每轮用当前策略采集的仿真转移拟合。动作库扩大时，机器人仍共享同一套物理动力学。这是计算复用的依据，不能直接假定A3获得同样倍数收益。
2. **保留真实物理数据闭环。** 每轮采集→8步自回归模型拟合→冻结模型权重进行32步想象更新。回放存执行过的加噪动作；想象中重新采样观测噪声。冻结模型参数仍须保留对状态/动作的梯度，不能用no_grad断掉策略梯度。
3. **直接沿模型反传回报。** 策略为未来10帧目标编码器、32维tanh潜变量及共享动作decoder，输入10帧本体观测。式2使用软生存概率和视野末critic，式3以固定GAE目标拟合critic；未使用PPO替代该梯度链。
4. **短视野加critic的意义。** 模型精度只需支撑短期可用梯度；critic近似更远收益。论文无critic消融在H≤16失败，在匹配跟踪回报时也更难恢复推力。这支持保留critic，但不是A3已经验证的结论。
5. **后续可接衣服目标接口。** 共享模型/decoder可以连接SMPL或速度等encoder；论文多接口研究仍初步，且下游任务只更新对应encoder。先完成A3动作跟踪闭环，再评估衣服SMPL接口，避免同时改变感知、目标和控制。

论文的H100三小时、129514训练动作覆盖及零失败均为作者在G1上的结果；Appendix F明确不是未见动作泛化。H100/3090、G1/A3、数据和仿真后端不同，不能直接作为本任务的耗时、成功率或真机证明。

## 第三方来源与选择

服务器已克隆两份实现，读取两份MIT许可和README，并审查所选基础的核心训练源码：

- `CuboiLeo/SuperMimic-Reproduced`固定`d9be385b3eeb13a638fb33fb200d3bfd9be9839f`，与早晨记录的旧提交不同。选用其可分离的可微FK、世界模型、回放、BPTT/GAE实现作为A3端口的基础；保留许可、来源及适配差异，数字不当作我们复现。
- `zita-ch/supermimic_vibed`固定`a1508d42c9d1699a52e8233ab16d08c488441ab9`，交叉核对方法与偏离；其实现依赖kitless IsaacLab3，非当前环境的直接训练入口。
- `mujocolab/mjlab`固定`8bb2bc9e187797689ec82c78c35019afd2fd0376`，只读参考。数学工具文件有IsaacLab BSD-3-Clause标头，单独保留对应声明。

官方浏览器Demo的专有参数/资产不用于本任务。第三方MIT许可不自动覆盖其引用的动作数据；现有A3输入只在服务器使用并登记，不重新分发模型/动作资产。

## A3结构和运行决定

实际MuJoCo加载现有优化MJCF，`nq=42, nv=41, nu=31`；29个策略关节、2个锁定头关节、4个无执行器被动足关节。完整状态是83维，包含全部35个铰链的位置/速度；策略动作仍29维。不能直接沿用G1的71维切片、关节顺序或足底几何。既有A3 CSV有20段；根位置厘米、欧拉/关节角度，源30Hz，转换必须按名称映射和重采样，并核对地板与限位。

当前驱动570.153.02，已用环境为Python3.11、torch2.8.0+cu128、MuJoCo3.3.7；CUDA可用。第三方默认环境要求torch2.14/cu130及MuJoCo3.16，未经本机验证。首个A3端口使用已验证MuJoCo物理采集和GPU学习，独立环境复用只读基础包；它保留论文算法，吞吐不是论文的MuJoCo-Warp口径。后续若迁移Warp，单独建立匹配验证。

主机boot `d994e71f-bae0-46ba-a5f7-6c2163771c6c`，RTX3090 24576MiB，UUID `GPU-2225f49c-1e19-9891-6cae-d084e5f828cf`；核查时已有6348MiB占用、0%利用率，不能把共享显存视为可释放资源。容器内存上限32GiB。保留未知进程、官方035和历史账本。

## 版本、命令与证据

- 来源克隆：服务器`data/references/supermimic_reproduced_20261022`、`supermimic_vibed_20261022`、`mjlab_supermimic_20261023`。
- 结构核验：`script/supermimic_for_a3/inspect_assets.py`；收据`data/supermimic_for_a3/source_audit_20261023/a3_assets.json`。
- 输入MJCF SHA-256：`3915053acd573f0cc093a080d64197574fdc146ad4cf91ec1b0ceff71983b75f`；实际新增训练入口、数据清单、命令和权重在完成后登记。
- 前一T13已按实际FAILED_PRE_IK保存并推送`6e43590167848669763b479de42df1703fe00bdb`，独立ls-remote一致。T13没有遗留训练，也不是SuperMimic的通过证据。

## 结果与边界

P01已完成全文方法/超参数核对、两份实现的来源核验、所选基础核心源码审查和A3结构核验。后续P02已通过新后端验证（见下文V04）；尚无完成训练、独立评测或真机结果，不能把数值/梯度检查称为策略训练成功。

## 保存与下一步

原始PDF/来源/资产核查为`LOCAL_ONLY`。本阶段README和入口在指定仓库保存，推送状态以实际Git核验为准。

接续：实现完整83维A3状态和29维动作合同→独立FK/观测/奖励/梯度及物理验证→固定训练/留出动作、step0基线→实际模型/策略联合训练→确定性真实仿真评测与工件核验。保持H=32、k=8和critic，按3090额度调整批量并记录偏离。

## P02适配中的验证记录（00:43）

已实现完整状态/主动关节选择、命名CSV转换、原生MuJoCo并行rollout、明确PD控制和BPTT训练/评测入口，尚未训练。

- M01转换20段，但参考link速度误用了`mjOBJ_BODY`（质心）。V01/V02独立FK检查发现速度差0.10494，位置/朝向约1e-15；修正为`mjOBJ_XBODY`（link原点），不改写M01，重新生成M02。M02的20段主动关节限位超量均为0；真实足碰撞网格最低高度约-0.8至-3.1cm，未经地板修正。
- V03的12组随机状态完整FK位置/速度最大差均1.55e-15、朝向1.11e-15。保存参考的一致性、观测选择和回放终止标签检查继续通过；5ms物理步长出现QACC数值不稳定，保护检查正确退出1，无训练和通过收据。
- 采用原A3 MJCF的1ms步长（20个子步/50Hz控制），明确作为G1论文4×5ms的偏离；下一独立V04验证，不能将V03称为物理通过。
- 日志：服务器`logs/supermimic_a3_V01.log`、`V02.log`、`V03.log`（均使用完整`supermimic_a3_`前缀）；数据M01/M02与源码保存于服务器工作根对应目录。原始工件`LOCAL_ONLY`，后续阶段收据绑定精确源码和环境。

## P02通过：V04后端与梯度合同

V04实际退出0，收据`data/supermimic_for_a3/source_audit_20261023/backend_V04.json`，小副本进入Git：`data/manifests/supermimic_a3_backend_V04_20261023.json`。12组全身随机状态FK对拍位置/速度最大差1.55e-15；保存参考/93维观测/主动关节隔离/终止标签对齐通过；1ms物理384个env-step完成，5.681秒。小网络诊断完成2次8步模型拟合和一次H=32 actor/critic更新，策略梯度非零、策略参数改变、世界模型在actor更新中逐参数不变，指标有限。这只证明数值与梯度合同，不是学习效果。

M02的20段参考、小型输入清单`data/manifests/supermimic_a3_motions_M02_20261023.json`已保存，权重/NPZ不进入Git。独立环境`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023`通过`.pth`只读复用原ZHY包，未更新原环境；新代码只写本工作根。当前包源码SHA绑定在V04，训练前逐文件复核。

## R01首轮训练合同

目标是获得并实际评测首个A3 SuperMimic策略，初次动作覆盖范围为四段，不把初次验证扩称论文规模通用控制器。

- 训练整段：001_walk_front_slow、042_squat_shallow_repeated、058_punch_front_alternating、077_reach_overhead_both。
- 预留整段：004_walk_frontleft_slow、048_balance_right_singleleg、056_lunge_side_alternating、080_reach_front_both。不用于本轮训练；来自同一个既有A3来源，不宣称独立受试者/跨域泛化。
- 完整论文网络宽度：WM 2048×3；encoder/decoder 2048/1024/512/256，z32；critic256/128，H32、k8、γ0.99、GAE0.95、10帧本体和未来10帧目标。
- 资源适配：64个真实环境、WM10批×128窗口、想象批64，UTD0.4；128个采集步后开始模型更新，先200次模型更新，再联合策略。目标3000个采集步、192000真实env-step；具体成功更新数以最终收据为准。
- 动力学：MuJoCo3.3.7 CPU原生并行rollout，1ms×20；保留实际A3接触和被动足；29维动作按既有A3 PD/力矩/armature映射，头部零位PD。摩擦/编码偏差/CoM随机化进入33维上下文。使用pelvis奖励锚点，不注入目标相对全局位置观测。
- 工件目录：`data/supermimic_for_a3/R01_20261023`；日志`logs/supermimic_a3_R01_20261023.log`；`step_000000.pt`为随机初始化基线；每500步保存，最终`final.pt`必须独立评测。一次独立作业，无自动恢复，不修改035和旧账本。
- 训练/评测入口、真实命令、停止/恢复字段见`script/supermimic_for_a3/README.md`。保护：非有限指标、物理不完整、GPU≥86℃或容器内存≥30GiB会停止，SIGTERM在当前更新后保存。原始工件均`LOCAL_ONLY`。

实现/合同已发布并独立核对main为`76ac42ef0a8e5d79af870dc9d8d29a26b2b76ae5`，随后实际启动R01，PID692029。启动前保存了step0和admission，但在第一次health检查因容器usage=34011369472字节超过30GiB保护而退出；0真实采集步、0模型/策略更新，没有最终模型或效果。只读检查该PID为Zs，已无计算活动，保留原目录与日志。

现场核查：原生A3每份MjModel buffer=86638912字节（约82.63MiB），64份模型约5.16GiB，额外占用导致启动保护。检查时共享容器既有usage约27GB，`memory.oom_control`历史oom_kill=14，此数字不是本轮新增OOM证明；R01实际错误是应用层保护。新独立R02采用16环境、4物理线程，保留30GiB/86℃保护；目标3000采集步=48000真实env-step，WM/想象批量、H/k、网络及四段训练/四段预留均不改。另修正初始化health失败也保存最终失败收据，避免只留admission。V04包源码未变；R02不覆盖R01。

## R02实际启动与恢复入口（01:11）

R01失败/资源调整已发布并独立核验`main=b6ebd8d9fb7a3fb12d99382eda6f93e5761a1063`。R02启动PID/PGID692767，同一boot；实际命令见[启动小清单](../data/manifests/supermimic_a3_R02_launch_20261023.json)。训练入口SHA-256 `a1fa89f2e5b9252319c9818452eb08c2230e73f9081ba40c6efcbb3ca76b857b`与服务器一致。

- 目录`data/supermimic_for_a3/R02_20261023`；日志`logs/supermimic_a3_R02_20261023.log`。独立初始化`step_000000.pt`已保存，不能用R01初始化替代本轮匹配基线。
- 首125个采集步/2000个真实env-step已完成，0模型/策略更新，随后进入首次世界模型编译。最近状态内存30206095360字节、GPU39℃；这些是启动快照，不是持续健康或训练完成证明。
- 启动后核查使用`status.json`、`metrics.jsonl`、`receipt.json`及实际PID/PGID；以真实终态为准，不因启动记录存在就续跑或称为完成。无自动重启。
- 下一步：完成预定3000步与联合更新；核验最终权重和收据；用本轮step0与final在相同八个完整片段进行确定性物理评测，保留早倒分母，再报告训练/留出集效果。原始权重/NPZ仍为`LOCAL_ONLY`。

## R02联合训练与文件缓存维护

R02已完成首次想象入口编译并进行真实策略/critic更新；最近核查850采集步、13600env-step、289次WM更新、89次策略更新，全部记录指标有限。每次联合更新约5.1秒；这是当前共享3090/CPU后端的实测吞吐，不使用论文H100耗时作本机承诺。运行进程未设置`CUDA_LAUNCH_BLOCKING`，GPU实际SM时钟1695MHz；当前不改变科学配置或中途缩减预算。

容器保护读取usage，包括本轮文件页缓存。为保留原30GiB保护又避免checkpoint缓存累计占额，新增`script/supermimic_for_a3/release_checkpoint_cache.py`，仅针对明确指定的本轮目录中已原子完成的常规PT文件，先`fsync`再`POSIX_FADV_DONTNEED`；不删除文件、不处理全局缓存或其他目录。首次对step0和step500执行成功，分别72021227/142947221字节；操作前后容器usage31744638976→31524446208字节（运行中快照，不能当精确独占回收量）。记录见服务器`logs/supermimic_a3_R02_cache_001.json`及[小副本](../data/manifests/supermimic_a3_R02_cache_001_20261023.json)。后续完成checkpoint按需同样处理，不放宽保护。

新增`script/supermimic_for_a3/audit_run.py`核验最终预算、三个网络实际变化/有限性、三个optimizer的精确step计数和有限状态、checkpoint/轨迹/参考SHA，以及两次评测完全匹配的片段与协议；从保存轨迹独立复算完整片段分母的回报和完成率，并复算分组统计。语法检查通过，实际权重/评测审计尚待R02完成，不能预先标成通过。

## R02半程与初始化基线完成

最近已核实1500/3000采集步、24000真实env-step、549次WM更新、349次策略更新，`step_001500.pt`已原子完成（216003303字节），温度41℃。训练仍进行中，没有最终模型或最终效果结论。

容器usage在后续快照中自行下降约2GB，具体原因未确定；原六个其他Python进程仍在，`oom_kill=14`及`memory.failcnt=158689333`与启动前相同。按新增余量运行初始化基线，没有停止/释放这些进程。R02本轮step0确定性八段评测实际退出0，25.587秒；后端源码/控制资产合同核验通过。目录`data/supermimic_for_a3/R02_eval_step0_20261023`，日志`logs/supermimic_a3_R02_eval_step0_20261023.log`；完整[评测收据](../data/manifests/supermimic_a3_R02_eval_step0_20261023.json)。step0为72021227字节，SHA-256 `009cacf08df91b04f7a21ed90221f10ab73e2782d53d04b283d6f7ba4a7959f3`。

| 未训练初始化组别 | 提前失败 | 平均归一化回报 | 平均完整片段完成率 |
| --- | --- | --- | --- |
| 四段训练片段 | 4/4 | 0.0181347619 | 2.537844% |
| 四段预留片段 | 4/4 | 0.0143205017 | 2.099767% |

八段实际执行75–179步（1.50–3.58秒），各原始片段2549–6607步（50.98–132.14秒）。回报分母包含早倒后丢失的整段，不能用存活段回报替代。此结果仅描述未训练step0；不代表R02训练后性能。后续最终策略使用相同八段和协议，预算/训练片段不因本基线改变。最终配对及轨迹复算审计尚未执行。

step1000后的缓存维护002同样仅作用于本轮已完成PT（新文件216003303字节），操作usage31755915264→31534579712，见[小清单002](../data/manifests/supermimic_a3_R02_cache_002_20261023.json)。本轮全部PT与NPZ保持`LOCAL_ONLY`；下一继续剩余预算，再最终评测和独立核验。

实际配置及输入身份已保存[admission小副本](../data/manifests/supermimic_a3_R02_admission_20261023.json)。step1500后的[缓存维护003](../data/manifests/supermimic_a3_R02_cache_003_20261023.json)仅处理本轮四个已完成PT，usage29851566080→29563531264，0文件删除；初始化基线对step0的读取也因此重新释放页缓存。所有处理均不改变数据、参数或训练预算。

## 后半程与存储入口复核（02:08）

R02最近2125/3000采集步、34000env-step、799次WM更新、599次策略更新，step2000已保存（216003303字节）。温度39℃，usage29568917504字节；在step2000附近复核历史oom_kill仍14，无新增计数，不将其扩称完整内核审计。step2000后的[缓存维护004](../data/manifests/supermimic_a3_R02_cache_004_20261023.json)仅同步并定向建议释放本轮五个PT页缓存，usage29789052928→29569486848，0删除。

本机存储入口已独立核查，[收据](../data/manifests/supermimic_a3_storage_20261023.json)固定UTC2026-10-22T18:08:46（北京时间10/23 02:08）、huanyu与当前boot：项目所在盘可用2056311394304字节，约1.87TiB；`/opt/baidunetdisk/baidunetdisk`不可执行，PATH中BaiduPCS-Go/baidupcs-go/bypy/rclone均未找到。范围仅为已知路径与PATH，未据此断言所有可能入口不存在。未读取认证材料、安装客户端或更换账号，原始执行收据在`logs/supermimic_a3_storage_20261023.json`。旧hp3090客户端8.7.0记录不能作为huanyu已具备传输的证明。

当前权重、动作和评测轨迹保持服务器原件及`LOCAL_ONLY`，没有网盘上传/下载校验或备份完成证据；Git仅保存代码、报告和小型清单。本机空间充足，无需删除checkpoint。下一完成剩余875个采集步，然后执行同八段最终评测和`audit_run.py`，先读实际终态，不能重启R01或复用旧STOPPED账本。

## R02训练预算完成（02:38）

实际终态`TRAINING_FINISHED_PENDING_EVALUATION`，error=null；3000采集步、48000真实env-step、1149次WM更新、949次策略更新，用时5268.134秒（87.80分钟）。PID692767已为Zs/RSS0，`/proc/692767/stat`退出字段0，无存活计算；历史oom_kill仍14、memory.failcnt仍158689333，未新增。训练完成不等于跟踪效果通过，最终权重和配对审计尚未执行。

- [训练终态/全部PT大小SHA清单](../data/manifests/supermimic_a3_R02_training_20261023.json)；[120条训练指标记录](../logs/session_records/supermimic_a3_R02_metrics_20261023.log)。原始JSONL与stdout保留服务器本轮目录及`logs/supermimic_a3_R02_20261023.log`。Git小副本使用仓库获准的`logs/session_records/*.log`位置；首次位置检查失败后已修正，未改变指标内容。
- 120个采样记录中GPU最高43℃、容器usage最高31748255744字节，均未越86℃/30GiB保护；这是采样最大值，不能代表未采样时刻的连续峰值。最终WM八步误差0.03613，高于热身末0.01573，联合训练中的分布变化和模型有效性仍需真实评测判断。
- `final.pt`大小215995283字节，SHA-256 `a491c07634a2fca4346b053e32d7d73f41f7f666864c2504f32b6ad1e59aa6d3`；step3000独立保存216003303字节，完整身份见清单。权重/输入/轨迹仍`LOCAL_ONLY`，没有Mac大型复制或Git权重提交。
- step2500后的[缓存维护005](../data/manifests/supermimic_a3_R02_cache_005_20261023.json)仍仅本轮六个PT、0删除；usage29775970304→29556363264。
- 下一立即在新目录`data/supermimic_for_a3/R02_eval_final_20261023`使用final.pt执行与step0相同八段/协议的真实MuJoCo评测，再`audit_run.py`核验参数/optimizer/保存轨迹与配对，最后报告真实效果。无需恢复或追加训练。

## R02最终物理评测与配对审计完成

最终八段评测实际退出0，42.260秒，使用相同第0帧初始化、无随机化/观测与动作噪声、完整参考片段分母。目录`data/supermimic_for_a3/R02_eval_final_20261023`、日志`logs/supermimic_a3_R02_eval_final_20261023.log`；[最终评测收据](../data/manifests/supermimic_a3_R02_eval_final_20261023.json)。不是历史035对照，不将随机初始化的提升解释为胜过035。

| 指标 | 四段训练片段：初始化→最终 | 四段预留片段：初始化→最终 |
| --- | --- | --- |
| 平均完整片段归一化回报 | 0.01813476→0.03459246 | 0.01432050→0.03824874 |
| 平均完整片段完成率 | 2.537844%→4.854001% | 2.099767%→5.119001% |
| 提前失败 | 4/4→4/4 | 4/4→4/4 |

八段中七段回报/完成率改善；前向慢走001由179步降至159步（3.58→3.18秒）。最终各段仅151–231步（3.02–4.62秒），对应完整参考50.98–132.14秒，**没有任何片段完成**。因此本轮结论是方法/训练/保存/真实评测闭环建立且出现有限学习收益，不能称可部署、稳定跟踪、论文性能复现或全面优于初始化。单seed/单固定起点/同源预留也不足泛化结论；这些预留已经用于诊断，后续若据此调参需视为开发评估，另留最终测试。

`audit_run.py`实际退出0，状态`TRAINING_AND_MATCHED_EVALUATION_VERIFIED`，见[配对审计](../data/manifests/supermimic_a3_R02_matched_audit_20261023.json)；这表示证据验收通过，**不是性能门通过**。

- WM14/15、策略20/22、critic6/7个state_dict张量实际改变；不变项为固定缩放buffer。最大绝对变化分别0.636013/0.065779/1.558208，最终参数和optimizer状态均有限。
- WM Adam14组各11490步（1149×10），策略20组各949步，critic6组各18980步（949×5×4）；与训练计数一致。
- 最终PT与训练收据身份匹配；两次评测16个轨迹和参考大小/SHA全部通过；83维状态/29维动作维数、完整片段分母回报、完成率及分组指标独立复算一致。原始审计日志`logs/supermimic_a3_R02_audit_20261023.log`。
- `plot_matched.py`已实际渲染PNG/PDF并检查PNG布局；服务器`data/supermimic_for_a3/R02_figures_20261023/`，身份见[图表清单](../data/manifests/supermimic_a3_R02_plot_20261023.json)。Mac仅复制117493字节PNG预览，权重/NPZ未复制。图中明确4/4→4/4终止和完整片段分母，不截取存活段制造提升。

下一先用已保存轨迹诊断世界模型1/8/32步预测偏差（无新仿真/optimizer），再选择独立下一轮训练方向；单凭本轮有限预算或模型内部回报不能断言失败根因。当前无活跃科学进程，R02本轮不自动恢复/追加，原件全部`LOCAL_ONLY`。可用策略目标仍未达成。

## D01：已存轨迹上的冻结模型多步诊断

`diagnose_model.py`实际退出0，输出`FROZEN_MODEL_TRACE_DIAGNOSTIC_COMPLETED`。仅载入R02 final.pt和八段已保存最终轨迹，按每个有效重叠窗口重放已执行动作，不调用物理步进、不更新参数；新增仿真0、optimizer0。逐文件核验checkpoint/资产/后端收据与源码/轨迹/参考身份，通过后分别计算1/8/32步开放环预测。上下文固定为实际评测摩擦1、零编码偏差/CoM偏移，动作按实际执行口径clamp[-20,20]。

日志`logs/supermimic_a3_R02_model_diagnostic_20261023.log`；完整[诊断收据](../data/manifests/supermimic_a3_R02_model_diagnostic_20261023.json)。组内先对每片段全部有效窗口取均值，再等权平均片段；下表为根位置三维欧氏RMSE，单位米：

| 组别 | 1步/20ms：学习模型/恒速度 | 8步/160ms：学习模型/恒速度 | 32步/640ms：学习模型/恒速度 |
| --- | --- | --- | --- |
| 训练片段 | 0.001399/0.000936 | 0.020102/0.029206 | 0.189660/0.129546 |
| 预留片段 | 0.000999/0.000636 | 0.016793/0.020935 | 0.172109/0.117682 |

整体标准化状态MSE在全部视野均优于恒速度（例如32步训练0.5305对6.2110、预留0.3900对2.7945），主动关节32步RMSE约0.0613/0.0560rad；因此不能概括成“世界模型没学会任何东西”。但在策略实际H32视野末，根位置误差约17–19cm且差于恒速度，8步拟合成绩不能保证32步策略梯度有效。恒速度是代码中的零残差积分基线，并非另一个训练策略。

当前支持的判断：长视野根部预测是应优先处理的明确短板。它只是在最终策略访问轨迹上的条件预测误差，尚未测梯度方向、反事实动作或独立因果，不把它直接认定为早倒的唯一原因。不存在由此证明“再训足够久必然成功”的证据。

后续独立实验方向：在保留H32策略思想的前提下，先单独检验更充分的世界模型采集/拟合能否改善长视野根部预测，再做与R02匹配的真实完整片段评测；固定预算/唯一主要变量/新输出目录，保留R02终态，不改旧训练权重或用已看过的预留当独立最终测试。R03尚未实现或启动；下次恢复先读本段与D01收据，核对实时资源及源码，不能把下一方向写成已通过或已训练。

## C01预登记：冻结R02策略的独立模型补充拟合

上一轮属实际进展：R02训练/评测闭环与D01已完成，目标尚未达到。本轮重新读取README、真实进程/GPU/内存、boot及final.pt身份：同一huanyu/boot，未发现本轮训练等活跃进程，GPU6348MiB/38℃/0%，usage27315347456字节、32GiB上限、历史OOM14；final.pt SHA仍为a491c07634a2fca4346b053e32d7d73f41f7f666864c2504f32b6ad1e59aa6d3。这是本次启动前快照，继续保留30GiB/86℃保护和其他进程。

假设：在相同动作接口和k8拟合下，更充分的真实数据/冻结策略模型训练能够改善H32视野末的根部预测。实施`calibrate_world_model.py`，仅把R02最终WM用于进一步拟合；冻结R02策略，不更新critic，不改变后端、H32目标、奖励、网络宽度、参考数据或动作比例。独立实验C01，不修改/续跑R02账本。

- 采集仅R02四段训练动作，各16环境/4线程、1024采集步；训练/验证种子4911/4912，各16384env-step，共32768个新物理步。顺序关闭两个模拟器以避免同时持有两组模型。
- 策略固定为R02最终参数，沿用原随机化/噪声。两组不同随机种子与重置episode，验证数据不进WM拟合；都来自训练动作，不使用四段已看预留。记录逐clip转移数，任何一段缺失则停止。
- 训练集固定保存完整replay；验证集预采样1/8/32各256个窗口并保存。模型初始/最终均在同一组窗口计算，比较R02 WM与额外1000次WM更新（每次10×128个k8窗口）。沿用R02 WM Adam状态与学习率，预期每参数step11490→21490；策略参数必须逐张量不变，0 actor/critic更新。
- 预先固定候选判据：最终32步根位置RMSE≤初始80%；一步整体状态MSE与32步主动关节RMSE均≤初始110%，同时有限性/计数/冻结证据通过。仅比较最终点，不按中间分数择优。判据是后续工程候选筛选，不是论文/性能成功门。
- 新目录`data/supermimic_for_a3/C01_20261023`，日志`logs/supermimic_a3_C01_20261023.log`，启动记录`data/supermimic_for_a3/C01_20261023_launch.json`；入口仍由launch.py显式`--entry calibrate_world_model`启动，无自动重启。
- 当前完成实现与语法检查，尚未实际运行C01或判据。原始replay/校准模型均`LOCAL_ONLY`，新模型尚不能作为可用策略；下一先实际执行、核验独立验证结果，再决定R03方向。

启动前复核：显式删除已关闭模拟器，确保第二组采集创建前释放第一组16份MjModel；加载所需状态后释放源checkpoint的CPU副本。两项仅调整生命周期，训练/验证采样、更新预算和判据不变。首份C01实现已同步并独立核验远端main=`3fe5a8e08a5ffbcaf012be12150203792dc9f4c8`；本次生命周期修正通过语法检查后同步，实际启动仍以新收据为准。

## C01实际启动（03:11）

生命周期修正已发布并独立核验main=`eafe0befe27bd311f2b9cd4724086a8189d8ac57`；本机/服务器入口SHA一致，`calibrate_world_model.py`为`bb4a044ef6562d38baedb3bc8308585db95d4ade2fa6baad7177174e0020571d`。本轮启动前boot/GPU UUID/输入SHA再次一致，usage27324932096字节、GPU38℃/6348MiB，无活跃SuperMimic作业；历史OOM14不变。原C01输出/日志/启动记录均不存在。

实际PID/PGID702742，独立作业、无自动重启；[启动清单](../data/manifests/supermimic_a3_C01_launch_20261023.json)与[admission](../data/manifests/supermimic_a3_C01_admission_20261023.json)已复制小文件留证。实际命令：

```bash
env TORCHINDUCTOR_CACHE_DIR=/opt/data/private/yuanqi_ops/data/tools/torchinductor_supermimic_a3 \
  envs/supermimic_a3_20261023/bin/python script/supermimic_for_a3/launch.py \
  --entry calibrate_world_model --python envs/supermimic_a3_20261023/bin/python \
  --log logs/supermimic_a3_C01_20261023.log \
  --record data/supermimic_for_a3/C01_20261023_launch.json -- \
  --checkpoint data/supermimic_for_a3/R02_20261023/final.pt \
  --root /opt/data/private/hechengzhu/ZHY \
  --output data/supermimic_for_a3/C01_20261023 \
  --collection-steps 1024 --num-envs 16 --threads 4 --updates 1000 --seed 4911 --compile
```

03:12快照：fitting第640采集步、10240env-step，0WM/actor更新；usage30325219328字节，GPU38℃，无新增OOM。此时仍在运行，尚无最终校准或策略效果。恢复以实际C01/status.json、receipt.json和PID为准，不把启动记录当完成证明。

C01运行期间新增独立`audit_calibration.py`：检查保存的数据计数/两种子、每个完整验证窗口属于验证replay且不属于fitting、输入/工件SHA、WM Adam实际计数，最后从初始/最终权重重新计算固定窗口误差与预登记判据。语法检查和服务器CPU三项控制通过（合法窗口接受、动作篡改拒绝、跨episode拒绝），不增加物理/optimizer或改变运行中的入口。此时尚未实际验收C01；actor冻结的独立证据限于hash绑定源码和其运行时逐张量检查，没有额外保存最终actor快照，明确不夸大为独立actor快照验证。

## C01实际完成与独立验收

C01原生wait status=0，PID702742为Z且RSS0；完整32768env-step、1000次WM更新（10000个Adam step），717.552秒；0 actor/critic更新，运行时逐张量确认actor不变。WM14个可训练张量变化，实际Adam11490→21490。四训练动作均出现在两组独立采集中，各组16384转移；两种子4911/4912。日志采样最高GPU42℃、容器30348595200字节；完成后OOM仍14，无新增。

`audit_calibration.py`实际退出0，状态`WORLD_MODEL_CALIBRATION_INDEPENDENTLY_VERIFIED`。1/8/32各256完整验证窗口逐一匹配validation replay，未出现在fitting replay；计数/大小/SHA/参数/优化器及重新计算指标一致，三项预登记判据全通过。

| 固定验证指标 | R02原WM | C01最终WM | 改善 |
| --- | --- | --- | --- |
| 1步标准化状态MSE | 0.06386335 | 0.03121058 | 51.13% |
| 8步根位置RMSE（米） | 0.02458826 | 0.01201112 | 51.15% |
| 32步根位置RMSE（米） | 0.17778267 | 0.08782202 | 50.60% |
| 32步主动关节RMSE（弧度） | 0.07648224 | 0.04535765 | 40.70% |

该结果只支持冻结R02访问分布上模型校准有效，未测新策略访问状态或反事实动作梯度。C01没有更新策略，机器人跟踪能力仍是R02的八段早倒；不能把这张表写成policy提升或论文结果复现。

- [实际校准收据](../data/manifests/supermimic_a3_C01_calibration_20261023.json)、[独立验收](../data/manifests/supermimic_a3_C01_audit_20261023.json)、[有界逐阶段指标](../logs/session_records/supermimic_a3_C01_metrics_20261023.log)。
- 校准WM：`data/supermimic_for_a3/C01_20261023/calibrated_world_model.pt`，106400425字节，SHA`3c4783ee92d90d0162d143898c8e48a799dc735b2ae301fe8364817be35a384c`。
- 两组replay各约24MB、固定窗口约6.45MB，精确大小/SHA见收据；全部留服务器且`LOCAL_ONLY`，未复制大文件到Mac。
- 复现验收：`audit_calibration.py --calibration data/supermimic_for_a3/C01_20261023 --output NEW_AUDIT.json`。原日志`logs/supermimic_a3_C01_audit_20261023.log`。

## R03实现与预登记：校准模型后的策略训练

依据C01通过结果推进实际策略闭环。新增`initialization.py`和`train.py --initialize-from ... --calibration-audit ...`：只接受已完成run的最终状态；资产、仿真、网络、训练器、引用和后端合同必须一致，采用校准WM须独立验收且候选门通过。继承R02 actor/critic及其Adam，WM/Adam改用C01；新replay、采样记忆、episode和显式种子，不伪装原进程恢复。step0是已训练父模型初始化；本轮计数归零，Adam累计计数和学习率起点继承并记录。R02原件/终态不改。

CPU控制验证继承张量/三种Adam状态与学习率偏移，拒绝仿真合同变化和失败校准；语法检查通过。[控制记录](../logs/session_records/supermimic_a3_initialization_controls_20261023.log)。修改`audit_run.py`逐张量/逐optimizer状态核对新run初始化来源并检查新增计数。原R02新输出回归验收退出0、效果数字不变，未重新仿真或优化：[回归收据](../data/manifests/supermimic_a3_R02_audit_regression_W01_20261023.json)。

真实R02/C01全宽网络初始化控制实际退出0：原A3资产/默认三网络配置匹配，actor/critic逐张量与R02相同，初始Adam21490/949/18980与schedule偏移3000一致，有限性/健康通过（GPU38℃、usage28894048256字节）。该控制仅分配/加载CUDA张量，0物理/optimizer更新；退出后再启动真实R03。

- R03固定3000采集步/16环境/4线程/48000新env-step，种子5921；四训练/四开发动作、H32/k8/网络/奖励/物理与批量保持。128步填新replay，已有校准WM不重复200次模型预热，`--model-warmup-updates 0`。预期1149次WM/actor更新，以实际收据为准。
- 初始Adam WM21490、actor949、critic18980；预期最终32980/2098/41960。学习率schedule起点3000，新增采集计数到6000。每500步保存，独立新目录`data/supermimic_for_a3/R03_20261023`和对应R03日志/launch.json，保持30GiB/86℃保护，无自动重启。
- 保存step0/final后按原协议八段完整确定性评测，独立审计继承状态/实际更新/早倒分母。当前四开发片段已看过，不当未触及最终测试。此轮比较自身step0/final检验组合训练是否改善，不能把新增策略预算的影响单独归因于C01。
- 初步可用门：四训练片段均完整不提前终止、均值完整片段归一化回报≥0.5、每片段存活段主动关节RMSE≤0.15rad；仍另报四开发结果。门仅限定这四片段的离线物理任务，不能称通用或真机通过。未达门则保存失败结论，再依据轨迹选择下一变量。
- 当前为实现/预登记，尚无R03训练或效果；下一核对源码/输入SHA与实时资源，发布后实际启动。

启动前补充：新run的每次PT保存后fsync并仅建议释放该文件页缓存，避免已完成检查点写入继续挤占32GiB共享额度；不删除文件或操作全局缓存。初始化manifest另绑定`initialization.py`本身大小/SHA，由后置审计核对。算法/预算/保护不变。

## R03实际启动与恢复入口（03:32）

实现/合同已发布并独立核验main=`5cc5557e2b6cf87ee092bf1cd2f75990a7a1fa8c`。服务器和发布副本三源码SHA一致：train `6ee24b056db6a2a5edea3c5086ded7c4d3b67f8748765f9904076f772ccd901a`、initialization `5e724982736c641af73e6c26e51090d827279d2fa6ce6f8cce5ac98518a4d0da`、audit_run `e0a051ed1b09c61f31bb05b68a1a43368b77e9264c1adaa54dc70a1ea7717507`。启动前同boot/GPU UUID、usage27487219712字节、GPU38℃/6348MiB、OOM14且无本轮作业；R03三个目标路径不存在。

实际PID/PGID703916；完整命令见[启动清单](../data/manifests/supermimic_a3_R03_launch_20261023.json)，[admission](../data/manifests/supermimic_a3_R03_admission_20261023.json)已保存实际初始化来源/计数。新目录`data/supermimic_for_a3/R03_20261023`，日志`logs/supermimic_a3_R03_20261023.log`，无自动恢复。

第一份125采集步快照：2000env-step，0WM/actor更新，正在进入编译/首次更新；step0已保存，usage30556114944字节、GPU38℃。这不是训练完成或策略有效证据。恢复先读本段，然后核对实际status.json/receipt.json和PID；不凭过期PID重复启动。

下一：完成既定3000步预算→独立step0/final八段物理评测→继承状态、Adam计数、轨迹及完整片段指标审计→按预登记可用门决定。当前所有大工件仍`LOCAL_ONLY`。

首轮联合更新已实际通过：[启动进度快照](../data/manifests/supermimic_a3_R03_startup_progress_20261023.json)记录275采集步、4400env-step、59次WM与59次actor更新，指标有限，GPU39℃/usage31240204288字节；最近核查OOM仍14。只是进行中证据，未完成3000步预算，没有R03最终权重/完整效果，目标继续保持未达。下一可执行步骤是读取R03/status.json、receipt.json和日志；收到完整训练终态后，按原README命令分别评测本轮step0与final，再用audit_run.py核验及计算上文门。

## R03完成流程实现（04时前后）

本次恢复实时确认R03进程仍活跃且更新增加，上一goal turn属于实际进展（完成C01并启动R03），本次不重启/复训。为使既定预算直接进入效果闭环，新增`complete_run.py`：绑定训练boot、PID/start ticks、完整命令及admission，等待该进程真实退出；仅在成功终态及预算匹配后，顺序执行本轮step0/final八段完整评测→匹配审计→`assess_policy.py`。各阶段一次，失败留证退出，不启动或恢复训练；完成流程本身不是目标达成声明。

`assess_policy.py`逐条核对审计/评测/轨迹身份并重算回报和关节指标，按R03预登记门输出结果。四开发片段另报，不当未触及测试。真实R02存量轨迹门复核实际退出0，完整性/均回报/每段关节三个门均false：[R02门回归](../data/manifests/supermimic_a3_R02_policy_gate_regression_20261023.json)，0新物理/optimizer。

语法检查和服务器3个聚合控制测试通过：全门达标接受，早倒/低回报/某段高关节误差拒绝，篡改轨迹拒绝；进程真实身份接受，boot/命令/起始tick变化拒绝。对真实活跃R03的同类只读控制也通过。见[控制记录](../logs/session_records/supermimic_a3_completion_controls_20261023.log)。源码已实现但此时完成流程尚未实际启动或完成；先发布并核验后启动独立等待进程，不把测试替代真实八段评测。

## R03完成流程实际启动（03:49）

源码/记录已发布并独立核验main=`01570666a5c1b84a906501dc1bca2c884276864b`；complete_run/assess_policy/launch两端SHA一致。新增完成进程PID/PGID704836实际活跃，RSS约18MiB，`waiting_for_training`；绑定训练PID703916/start_ticks3598955095/既有boot，尚未执行新评测/optimizer。[启动清单](../data/manifests/supermimic_a3_R03_completion_launch_20261023.json)、[admission](../data/manifests/supermimic_a3_R03_completion_admission_20261023.json)、[启动快照](../data/manifests/supermimic_a3_R03_completion_startup_20261023.json)已保存。

实际R03训练快照600/3000采集步、9600env-step、189次WM/actor更新；GPU40℃、usage29615190016字节，OOM仍14。训练和等待进程都已重新查验为活跃，不因旧状态文件假定存活。此时仍无R03最终效果，不能称可用策略。

```bash
envs/supermimic_a3_20261023/bin/python script/supermimic_for_a3/launch.py \
  --entry complete_run --python envs/supermimic_a3_20261023/bin/python \
  --log logs/supermimic_a3_R03_completion_20261023.log \
  --record data/supermimic_for_a3/R03_completion_20261023_launch.json -- \
  --run data/supermimic_for_a3/R03_20261023 \
  --launch data/supermimic_for_a3/R03_20261023_launch.json \
  --output data/supermimic_for_a3/R03_completion_20261023 \
  --log-dir logs/supermimic_a3_R03_completion_20261023
```

恢复先读本节并实时核对两个PID/boot；查`R03_20261023/status.json`与`R03_completion_20261023/status.json`。后者成功后查看`matched_audit.json`、`policy_assessment.json`及eval_step0/eval_final/receipt.json；失败时读取自身receipt和逐阶段日志，不自动重跑失败阶段或训练。权重和轨迹仍LOCAL_ONLY，Git仅保存小记录。下一真正决定为完整物理效果是否达门，目标保持active。
