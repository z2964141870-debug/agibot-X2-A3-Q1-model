# SuperMimic for A3：论文、实现与训练

记录日期：2026-10-23（北京时间）。阶段：SM-A3-P01/P02/R01/R02/C01/R03/R04/R05/DATA01。最新关键纠正：固定vendor文档声明A3源CSV为120Hz，本转换按30Hz，R02-R05参考被放慢4倍；旧结果只支持慢放条件，不能当原速跟踪。R04已完成且八段仍早倒，模型不可用；R05于11:01正常保存STOPPED_SIGNAL并独立核验退出和最终SHA，未完成配对，不重启。见[训练原料与时间基准纠正](README_SUPERMIMIC_DATA_20261023.md)，本条覆盖下文历史30Hz假设；原始工件和记录保留，目标尚未完成。

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

## R03实际保存初态独立核验（03:54）

本轮step0为216005351字节，SHA`65a93e9570af4efce76fafaf0d95abdd7dd5a13fa22ec07d2f3b5cdb3eaf2962`。已只读CPU/mmap载入实际保存PT、R02父PT和C01模型，实际退出0；[独立收据](../data/manifests/supermimic_a3_R03_initial_state_audit_20261023.json)状态`R03_SAVED_INITIAL_STATE_INDEPENDENTLY_VERIFIED`。

- 本轮step/wm_updates/policy_updates均0；actor22个/critic7个状态张量逐一等于R02，WM15个张量逐一等于C01，全部有限。
- 三优化器param_groups与每个step/exp_avg/exp_avg_sq逐值匹配来源；WM14个参数状态21490、actor20个949、critic6个18980；学习率偏移3000正确。
- 初始化源码/来源checkpoint/来源终态/C01校准审计等五项身份重新通过。只读核验新增物理0、optimizer0；GPU40℃，健康检查前后usage29913178112/30499733504字节。
- 后置完整audit_run仍必须执行；本节只证明保存初态继承正确，不代表R03训练预算已完成或策略有效。当前真实训练750/3000步、249次联合更新，两个PID均活跃；不改变/重启运行中的作业。

下一继续等待原训练与已启动完成进程，完整效果出来后才选择新实验；不因短期采样reward或模型内部回报提前改预算/奖励。

## 保存轨迹的真实A3渲染与R03三分之一进度（04:04）

新增`render_trace.py`，从评测收据绑定的原始NPZ状态与目标NPZ、A3 MJCF生成同相机并排视频；只做保存状态的FK/渲染，不执行mj_step、policy推理或optimizer。头部明确完整参考长度、实际执行时间与提前失败；默认播放全部已执行轨迹，若显式截短则收据标注。它是效果判读工具，不代替原物理评测。

首个实际V01使用R02最终001走路轨迹，退出0；EGL/实际网格/Pillow/ffmpeg入口通过。生成80帧/25FPS/3.2秒、1280×544 H264，ffprobe独立确认帧数/尺寸/时长。两面板最小像素标准差26.58，实际检查PNG可见完整参考机器人与明显倾倒的策略机器人，文字无重叠，和原3.18秒提前失败一致。原参考132.14秒在画面保留，未把3秒预览当完整动作成功。0新物理/推理/优化；采样最高GPU40℃、usage30961815552字节，保护未放宽。

- [渲染收据](../data/manifests/supermimic_a3_R02_trace_render_V01_20261023.json)绑定源码/资产/轨迹/视频身份。
- 服务器`data/supermimic_for_a3/R02_trace_render_V01_20261023/`保存原件；日志`logs/supermimic_a3_R02_trace_render_V01_20261023.log`。
- MP4为183031字节、PNG186215字节；Mac仅复制这两个小预览至`data/research/2026-10-23/supermimic/R02_trace_comparison.mp4`和`R02_trace_last_frame.png`，未复制大模型/数据。Git只存代码/小收据，原件`LOCAL_ONLY`。
- 复现命令见代码README的render_trace入口，输入R02_eval_final_20261023/receipt.json、R02_20261023/admission.json、clip001_walk_front_slow。下一用于本轮完整评测后的实际轨迹，不改正在运行的训练/完成流程源码。

同轮实时R03已到1025/3000采集步、16400env-step/359次联合更新；训练703916和完成704836均活跃，后者继续waiting_for_training。step_001000.pt已保存216005351字节、SHA`25b5cb007dd9204754fdbe5d09ef7efacdb3dd182bc55f91b2f50006617c4997`。GPU39℃，usage30222344192字节，OOM仍14无新增；[本轮进度快照](../data/manifests/supermimic_a3_R03_progress_20261023.json)是进行中证据，未完成预算、尚无R03最终效果。上一goal turn完成实际保存初态核验，属进展；本轮又完成轨迹渲染入口及视觉验证，不将等待重复记成科学完成。

## R03半程检查点与真实进程核验（04:22）

训练持续按原合同运行，半程`step_001500.pt`已完成，216005351字节、SHA`4e39d0b15133ebd546eb9eacbe494d3ded0e48c556f5800b2d5dfd1aedd435a5`。北京时间04:22:41的[独立小快照](../data/manifests/supermimic_a3_R03_midpoint_20261023.json)记录1525/3000采集步、24400新增env-step、559次WM和actor联合更新；61条已记录指标均有限，GPU39℃、容器usage30654042112字节，历史OOM仍14。快照只覆盖已采样记录，不代表连续资源峰值或最终策略效果。

核对实际`/proc`状态、完整命令、boot及start_ticks：训练703916/3598955095与完成704836/3599058278均活跃，后者仍`waiting_for_training`；两个终态receipt尚未产生。检查未调用物理、推理或optimizer，未改配方/预算/源码，没有重启训练。原件保持服务器`LOCAL_ONLY`，Git只保存小快照与恢复记录。

下一继续原作业剩余预算；完成流程退出后先看自身receipt及逐阶段日志，再看eval_step0/eval_final、matched_audit和policy_assessment。仅成功训练或证据审计不能证明跟踪可用；必须依据同八段完整物理评测及原预登记门决定后续实验，目标继续active。

## R03训练、完整配对评测与可用门实际完成（05:16）

原训练正常退出0：3000采集步、48000新env-step、1149次WM和1149次actor更新，6038.980秒（100.65分钟），error=null。原完成进程也退出0，四个阶段依次完成：step0评测47.434秒、final评测90.617秒、匹配审计5.365秒、可用性评估2.512秒；没有新训练或自动重启。两个原PID均为Z/RSS0，start_ticks/boot匹配，原生wait status均0，见[终态核验](../data/manifests/supermimic_a3_R03_terminal_20261023.json)。120条记录指标均有限，采样最高GPU40℃、容器31242362880字节，历史OOM仍14；这是采样证据，不声称连续峰值。

| 完整片段指标 | 四训练片段：step0→final | 四开发片段：step0→final |
| --- | --- | --- |
| 平均归一化回报 | 0.03459246→0.08176777 | 0.03824874→0.09778724 |
| 平均完成率 | 4.854001%→10.764340% | 5.119001%→12.970858% |
| 提前失败 | 4/4→4/4 | 4/4→4/4 |

八段的回报与完成率全部提高，实际执行从3.02–4.62秒延长至5.28–10.66秒，但参考完整长度为50.98–132.14秒，**仍没有任何完整片段成功**。001走路475/6607步（9.50秒）、042下蹲389/4626（7.78秒）、058出拳264/2712（5.28秒）、077举臂525/2962（10.50秒）。四训练片段实际执行段关节RMSE为0.22603/0.23200/0.21890/0.34024rad，全部高于0.15rad；077比自身step0的0.25506rad更大，且两者存活长度不同，不据此直接归因。开发四段分别526/533/326/525步，仍是已看过的开发评估，不是未触及最终测试。

预登记三个门全部false：四训练片段完整不失败、均完整回报≥0.5、每段执行关节RMSE≤0.15rad。本轮组合训练确有物理效果改善，**策略仍不可用**；未做等额无C01对照，不能把增益单独归因于校准，也不能声称优于035、论文规模复现、泛化或真机能力。

`audit_run.py`确认三网络变化与有限性、实际初始化继承、精确Adam计数、16份轨迹/参考身份及完整分母独立复算通过：WM14/15、actor20/22、critic6/7张量改变；最终Adam32980/2098/41960，新增11490/1149/22980，与预算一致。`assess_policy.py`再核对实际轨迹和关节误差，门失败不是执行/审计失败。

- [训练收据/全部PT身份](../data/manifests/supermimic_a3_R03_training_20261023.json)、[120条原始指标小副本](../logs/session_records/supermimic_a3_R03_metrics_20261023.log)、[完成流程收据](../data/manifests/supermimic_a3_R03_completion_20261023.json)。
- [step0评测](../data/manifests/supermimic_a3_R03_eval_step0_20261023.json)、[final评测](../data/manifests/supermimic_a3_R03_eval_final_20261023.json)、[匹配审计](../data/manifests/supermimic_a3_R03_matched_audit_20261023.json)、[可用门结论](../data/manifests/supermimic_a3_R03_policy_assessment_20261023.json)。
- final.pt为215997267字节，SHA`11f7dbe9b397be1abf88802a56c6e538e79f719b4f270f6e1909d76f14eea51e`；原件`data/supermimic_for_a3/R03_20261023/final.pt`，评测/审计在`R03_completion_20261023/`，逐阶段日志在`logs/supermimic_a3_R03_completion_20261023/`。
- step0物理指标与R02 final完全一致，符合actor精确继承；R03 step0绑定本轮PT，未用旧文件替代。模型/轨迹原件仍在服务器且LOCAL_ONLY，Mac只接收小清单/日志。

下一先基于本轮保存轨迹核查实际跟踪和H1/H8/H32模型预测，再按证据选择下一独立训练变量。原R03已结束，不恢复或重跑其完成阶段；不得将执行完成/增益或三项门失败改写为目标完成。原始可用SuperMimic for A3目标继续active。

## D02新策略轨迹诊断与V02实际渲染

既有`diagnose_model.py`使用R03最终保存轨迹实际退出0，逐文件身份和V04后端源码合同通过；0新物理/optimizer。所有有效重叠窗口内先按片段平均，再等权汇总。本轮H32根位置RMSE训练0.06053148米、开发0.05481941米，低于同轨迹恒速度0.07914635/0.07263191米；H8为0.01501049/0.01262554米，H1为0.00092106/0.00090956米。H32主动关节RMSE为0.05205442/0.04660527rad。

这些数值来自R03访问轨迹，与D01的R02轨迹分布不同，不能直接当固定数据集上的模型改善比例或早倒唯一因果。它们支持模型在当前访问状态上已有实质多步预测能力，尚未测反事实动作梯度。见[D02收据](../data/manifests/supermimic_a3_R03_model_diagnostic_D02_20261023.json)，服务器原件`data/supermimic_for_a3/R03_model_diagnostic_D02_20261023.json`，日志`logs/supermimic_a3_R03_model_diagnostic_D02_20261023.log`。

`render_trace.py`对R03最终001走路完整已执行9.50秒实际退出0：238帧、25FPS、9.52秒H264、1280×544；ffprobe核对尺寸/帧数/时长一致，面板最小像素标准差22.90，PNG已实际目视检查，末帧策略明显侧倾而参考直立。画面保留完整132.14秒分母与提前失败说明，未截短或包装成成功。0新物理/策略推理/optimizer，采样最高GPU39℃、usage28740141056字节。

- [V02渲染收据](../data/manifests/supermimic_a3_R03_trace_render_V02_20261023.json)绑定实际模型、轨迹及工件；服务器`data/supermimic_for_a3/R03_trace_render_V02_20261023/`，日志`logs/supermimic_a3_R03_trace_render_V02_20261023.log`。
- PNG183618字节、SHA`32c5b92f961f6bb581824904c6a27860b19b3783f7f6a208f86fef7da85d9fd9`；MP4352833字节、SHA`81278752db58716592d8a48059ec0b4d701811111a4c88bc9c0e7479bd364dc7`。Mac仅复制这两个小预览至`data/research/2026-10-23/supermimic/R03_trace_last_frame.png`及`R03_trace_comparison.mp4`，大小/SHA再核对一致。

## R04预登记：保持配方，增加联合训练预算

选择理由：R03八段完整分母回报/完成率均改善，执行时间延长，D02未显示当前H32根预测比恒速度更差；目前优先检验更多策略/模型联合训练是否继续改善，不再重复单独WM校准，也不同时更换奖励、horizon或物理。此为工程预算检验，不是C01因果消融或保证收敛。

- 独立新run，全部三网络与三Adam继承已审计R03 final；不再注入C01旧WM，不改R03原件/终态。新replay、adaptive sampler和episode，显式新种子6921；学习率偏移6000，新采集计数归零，累计schedule到12000。
- 本轮6000采集步×16环境=96000新真实env-step，4物理线程；128步填新replay、模型预热0，预计2349次WM与2349次actor更新，以实际收据为准。初始Adam32980/2098/41960，预期最终56470/4447/88940。
- 四训练/四开发片段、83维状态/29动作、H32/k8、网络宽度、批量、奖励、控制/随机化/原1ms×20物理保持。检查点每500步；30GiB/86℃保护及定向已保存PT缓存建议保持，无自动重启。按R03吞吐粗估约3.4小时，不作完成时限承诺。
- 新目录`data/supermimic_for_a3/R04_20261023`，日志`logs/supermimic_a3_R04_20261023.log`，启动清单`data/supermimic_for_a3/R04_20261023_launch.json`。后续独立完成目录`R04_completion_20261023`，原`complete_run.py`等待本轮实际PID/boot/start_ticks，预算成功后各一次step0/final八段完整评测、审计与原三项可用门。
- 原三项门不放宽；同源开发片段不当最终未见测试。模型/轨迹原件继续LOCAL_ONLY，不复制大型文件到Mac或放Git。当前仅预登记，尚未启动R04；下一实时资源/输入SHA核对、发布并核验后实际启动。

```bash
env TORCHINDUCTOR_CACHE_DIR=/opt/data/private/yuanqi_ops/data/tools/torchinductor_supermimic_a3 \
 envs/supermimic_a3_20261023/bin/python script/supermimic_for_a3/launch.py \
 --python envs/supermimic_a3_20261023/bin/python \
 --log logs/supermimic_a3_R04_20261023.log \
 --record data/supermimic_for_a3/R04_20261023_launch.json -- \
 --root /opt/data/private/hechengzhu/ZHY \
 --motions data/supermimic_for_a3/motions_20261023_M02 \
 --backend-receipt data/supermimic_for_a3/source_audit_20261023/backend_V04.json \
 --output data/supermimic_for_a3/R04_20261023 --steps 6000 --num-envs 16 --threads 4 \
 --batch 64 --wm-batch 128 --wm-batches 10 --warmup-steps 128 \
 --model-warmup-updates 0 --checkpoint-every 500 --seed 6921 --compile \
 --initialize-from data/supermimic_for_a3/R03_20261023/final.pt \
 --clips 001_walk_front_slow 042_squat_shallow_repeated 058_punch_front_alternating 077_reach_overhead_both
```

## R04实际启动、联合更新与恢复入口（05:29）

预登记与D02/V02已发布，独立核验main=`d58786d4fe0b872ba0bf12dcfe1304de0201c489`。启动前[实时预检](../data/manifests/supermimic_a3_R04_preflight_20261023.json)通过：同boot/GPU UUID，usage27966128128字节、GPU38℃/6348MiB/0%，历史OOM14，无活跃SuperMimic作业；父R03 final大小/SHA与匹配审计相符，V04和执行源码SHA未改，所有R04新路径不存在。未干预其他占用。

原入口已实际单次启动训练PID/PGID709634、start_ticks3599637468；[launch](../data/manifests/supermimic_a3_R04_launch_20261023.json)与[admission](../data/manifests/supermimic_a3_R04_admission_20261023.json)保存完整命令、父工件和配置。全部三网络及Adam从R03继承，admission初始32980/2098/41960、LR偏移6000正确，未再次注入C01；实际保存step0的逐张量/Adam独立审计仍待后置audit_run，不能仅凭admission宣称已经完成。

完成进程PID/PGID709693、start_ticks3599641134也已实际启动；[完成launch](../data/manifests/supermimic_a3_R04_completion_launch_20261023.json)和[admission](../data/manifests/supermimic_a3_R04_completion_admission_20261023.json)绑定训练709634/既有boot/命令/起始tick及四个执行源码身份，当前`waiting_for_training`，不启动/重启训练。各评测/审计/可用门只在本次6000步预算成功后顺序各一次，失败留证停止。

[05:29启动核验](../data/manifests/supermimic_a3_R04_startup_20261023.json)记录175/6000采集步、2800新env-step、19次WM与19次actor联合更新；两进程实际活跃，7条指标均有限，GPU39℃、usage31720005632字节，OOM仍14。step0为216004839字节、SHA`19f37d105edeef5c7f2cab33a5a92d4adba05ac01e043cb2034328d04319cf36`；读取身份后仅对本文件fsync/fadvise，不删除或改变参数。当前没有R04最终效果、终态或可用证明，不能将首批更新当完成。

```bash
envs/supermimic_a3_20261023/bin/python script/supermimic_for_a3/launch.py \
 --entry complete_run --python envs/supermimic_a3_20261023/bin/python \
 --log logs/supermimic_a3_R04_completion_20261023.log \
 --record data/supermimic_for_a3/R04_completion_20261023_launch.json -- \
 --run data/supermimic_for_a3/R04_20261023 \
 --launch data/supermimic_for_a3/R04_20261023_launch.json \
 --output data/supermimic_for_a3/R04_completion_20261023 \
 --log-dir logs/supermimic_a3_R04_completion_20261023
```

下一继续原训练与等待进程。恢复先核对实际boot/PID/start_ticks/命令、R04_20261023/status.json及receipt.json、R04_completion_20261023/status.json及receipt.json；成功后读取eval_step0/eval_final/receipt.json、matched_audit.json、policy_assessment.json和逐阶段日志，失败不自动重跑。保持原6000步/96000新交互与三项可用门、30GiB/86℃保护，旧R03/STOPPED终态不改，所有大型工件LOCAL_ONLY。

## R04父检查点缓存维护与保存初态独立核验（05:36）

实时核查R04仍在运行，但容器usage约31.7GB，距30GiB保护约0.5GB。复用既有`release_checkpoint_cache.py --run data/supermimic_for_a3/R03_20261023`，只处理已完成R03的8份常规PT；先核对终态清单及大小，并确认活跃R04没有这些父文件的FD/maps。fsync/fadvise前后usage31722033152→29993619456字节，下降1728413696字节（约1.73GB，运行中快照，不声称独占精确回收），0文件删除、未操作全局缓存。见[维护001原始收据](../data/manifests/supermimic_a3_R04_parent_cache_001_20261023.json)，服务器日志`logs/supermimic_a3_R04_parent_cache_001_20261023.json`。

随后有余量执行只读CPU/mmap核验实际R04 step0及R03 final，实际退出0：[独立初态收据](../data/manifests/supermimic_a3_R04_initial_state_audit_20261023.json)，[实际执行命令记录](../logs/session_records/supermimic_a3_R04_initial_state_audit_command_20261023.log)。三网络WM15/actor22/critic7状态张量逐一完全相同且有限；三Adam的参数组与每个参数所有状态/矩完全相同，14/20/6组实际step32980/2098/41960。step/wm_updates/policy_updates均0，学习率偏移6000正确；父PT、step0和admission大小/SHA与登记一致。没有新增物理、推理或optimizer更新。

首个核验尝试在清单比较处因PT的tuple/TorchVersion和JSON的list/str表示差异而断言失败，尚未进行张量比较；随后逐字段确认仅有这些类型差异，使用保留全部字段的排序JSON序列化严格比较后通过，再执行完整张量/Adam核验。未改训练/输入/可用门，不能把这次读取器断言写成训练失败或隐藏它。

CPU核验后仅对读过的两份PT fsync/fadvise；健康前后usage30267146240/30294253568字节、GPU40℃，仍在原保护内，OOM保持14。核验结束实时400/6000采集步、6400新env-step、109次联合更新，训练709634与完成709693均真实活跃，完成仍waiting_for_training。初态通过不替代最终预算、配对效果和完整audit_run；R04仍没有最终结果，目标active，继续原作业。

## R04半程检查点与实际进程核验（07:05）

原训练已超过本轮半程，`step_003000.pt`为216004839字节，SHA-256 `ec95767e403928f3e0a128c171c22bba201e451ef5d461eaa915e82cab7ff876`。北京时间07:04:59的[半程核验快照](../data/manifests/supermimic_a3_R04_midpoint_20261023.json)记录3025/6000采集步、48400新env-step、1159次WM和1159次actor更新；121条已记录指标均有限，采样最高GPU40℃、容器usage31711920128字节，历史OOM仍14。核验时实际usage29813653504字节；这些是记录与现场快照，不代表连续峰值或最终策略效果。

训练709634/start_ticks3599637468与完成709693/3599641134在既有boot上均真实活跃，完整命令匹配launch，完成仍`waiting_for_training`。训练入口、V04绑定28份后端源码、完成流程绑定4份入口身份均未改；两个终态receipt尚未产生。检查只读取文件和进程，0新物理、0 optimizer；对本轮已完成半程PT读取后执行fsync/fadvise，0删除。原始[实际核验命令](../logs/session_records/supermimic_a3_R04_midpoint_command_20261023.log)与小快照一并保存。

下一继续原6000采集步/96000新env-step预算及30GiB/86℃保护，无自动重启；完成进程在成功预算后各一次八段step0/final物理评测、匹配审计与原三项可用门。半程文件完整或指标有限不能证明稳定跟踪，目标保持active。恢复仍核对`R04_20261023/`与`R04_completion_20261023/`的status/receipt及实际进程；模型与轨迹原件在huanyu，`LOCAL_ONLY`，Mac和Git只保存小记录。报告索引与当前状态同次更新；GitHub同步以本次提交后的独立远端核验为准。

## R04完整训练、配对评测与可用性结论（08:51）

训练正常退出0：6000采集步、96000新env-step、2349次WM与2349次actor更新，11872.690秒（197.88分钟），error=null。完成流程同样实际退出0：step0评测88.696秒、final评测233.466秒、匹配审计6.073秒、可用性评估2.768秒，各一次完成，未重新训练。两原PID实际为Z，boot/start_ticks匹配且原生wait status均0，见[终态核验](../data/manifests/supermimic_a3_R04_terminal_20261023.json)与[实际核验命令](../logs/session_records/supermimic_a3_R04_terminal_command_20261023.log)。

| 完整片段指标 | 四训练片段：step0→final | 四开发片段：step0→final |
| --- | --- | --- |
| 平均归一化回报 | 0.08176777→0.19936035 | 0.09778724→0.33974153 |
| 平均完成率 | 10.764340%→25.188887% | 12.970858%→43.156445% |
| 提前失败 | 4/4→4/4 | 4/4→4/4 |

八段回报与完成率均提高，实际执行延长到15.64–47.76秒，但参考完整长度为50.98–132.14秒，**仍无任何完整片段成功，策略尚不可用**。训练001走路782/6607步（15.64秒）、042下蹲974/4626（19.48秒）、058出拳1119/2712（22.38秒）、077举臂788/2962（15.76秒）；四开发片段分别1289/2388/1718/789步，048虽达到93.6838%完成率，仍在47.76秒提前失败，不能写成成功。

训练执行段关节RMSE为0.17404/0.19528/0.34386/0.50987rad，全部高于0.15rad；出拳与举臂分别高于自身step0的0.21890/0.34024rad。存活长度不同，不能当相同时间窗口的精度退化因果证明，也不能笼统声称动作精度全面提高。原三个门均false：四训练片段完整不失败、平均完整回报≥0.5、每段执行关节RMSE≤0.15rad。开发片段此前已用于诊断，不是未触及最终测试；尚无035等预算优势、泛化、真机或论文规模效果证明。

`audit_run.py`核对实际初始化继承、三网络有限性与变化、Adam累计与新增计数、16份轨迹及参考身份，并独立复算完整分母指标通过。WM14/15、actor20/22、critic6/7状态张量改变；最终Adam56470/4447/88940，新增23490/2349/46980，与2349次联合更新一致。step0物理结果与R03 final一致。审计通过证明证据一致，不替代性能门。

- [训练收据与全部PT身份](../data/manifests/supermimic_a3_R04_training_20261023.json)、[240条训练指标](../logs/session_records/supermimic_a3_R04_metrics_20261023.log)、[完成流程收据](../data/manifests/supermimic_a3_R04_completion_20261023.json)。
- [step0评测](../data/manifests/supermimic_a3_R04_eval_step0_20261023.json)、[final评测](../data/manifests/supermimic_a3_R04_eval_final_20261023.json)、[匹配审计](../data/manifests/supermimic_a3_R04_matched_audit_20261023.json)、[可用性评估](../data/manifests/supermimic_a3_R04_policy_assessment_20261023.json)。
- final.pt：215996819字节，SHA-256 `d7e14eeda38b2fd2400bb364281335167e0eecaede6d4204e092d41e53931e08`，服务器`data/supermimic_for_a3/R04_20261023/final.pt`。原始评测/审计在`R04_completion_20261023/`，逐阶段日志在`logs/supermimic_a3_R04_completion_20261023/`；训练及完成命令仍为前文预登记/实际启动命令。
- 240条已记录指标均有限，采样最高GPU40℃、容器31711920128字节，历史OOM仍14无新增；终态核验实际usage29317435392字节/GPU38℃。采样最大值不代表连续峰值。训练入口、28份后端源码、4份完成入口身份均未改。
- 大型模型与轨迹保留服务器原件且LOCAL_ONLY，无云端往返校验；Mac只复制小收据与321099字节指标日志，未复制权重或原始NPZ。GitHub同步以本阶段提交及独立远端核验为准。

下一先在已保存R04轨迹上核查H1/H8/H32模型误差与实际跟踪画面，0新增物理/optimizer，再按证据选择下一独立训练方向。R04及其成功完成阶段不恢复或重跑，原预算不追加、性能门不放宽；可用SuperMimic for A3目标保持active。

## D03与V03/V04：R04保存轨迹诊断及下一动作响应检查

`diagnose_model.py`实际退出0、9.514秒，D03绑定R04 final和八段最终轨迹，资产/后端/轨迹身份通过；只重放已执行动作到冻结WM，没有新物理或参数更新。每片段内先平均全部有效重叠窗口，再等权平均片段：

| 组别 | H1根位置RMSE：学习/CV（米） | H8：学习/CV（米） | H32：学习/CV（米） |
| --- | --- | --- | --- |
| 四训练片段 | 0.00045664/0.00027862 | 0.00630197/0.00953855 | 0.06173307/0.04554071 |
| 四开发片段 | 0.00043639/0.00020613 | 0.00492917/0.00651816 | 0.05952967/0.03638820 |

H32主动关节RMSE学习/CV为0.036994/0.098506rad（训练）、0.035966/0.075125rad（开发），整体标准化状态MSE也优于CV；不能概括成模型没有学会动力学。但H32根位置误差在八段上都大于各自同轨迹CV。该基线只是零残差恒速度积分，且R04与D02访问轨迹/存活长度不同，不能直接把跨轮数字当固定分布退化比例或把此误差当早倒唯一原因。**尚未实测真实物理下动作响应或反事实梯度。**

- [D03原始诊断](../data/manifests/supermimic_a3_R04_model_diagnostic_D03_20261023.json)；服务器`data/supermimic_for_a3/R04_model_diagnostic_D03_20261023.json`，原始stdout日志`logs/supermimic_a3_R04_model_diagnostic_D03_20261023.log`。
- V03走路实际退出0、18.887秒，391帧/25FPS/15.64秒；V04举臂退出0、18.829秒，394帧/25FPS/15.76秒，均1280×544。独立ffprobe确认帧数、尺寸、时长；面板最小像素标准差25.986/29.769，PNG已实际目视核对，均明显侧倾，举臂也未跟随目标双臂。画面保留原完整132.14/59.24秒与提前失败说明，没有截短。
- [V03收据](../data/manifests/supermimic_a3_R04_trace_render_V03_20261023.json)、[V04收据](../data/manifests/supermimic_a3_R04_trace_render_V04_20261023.json)及[三次原生执行与资源/ffprobe记录](../data/manifests/supermimic_a3_R04_diagnostic_render_execution_20261023.json)。渲染只对保存状态做FK和图像生成，0新物理/策略推理/optimizer，不是新的跟踪效果评测。
- 原件在`data/supermimic_for_a3/R04_trace_render_V03_20261023/`和`R04_trace_render_V04_20261023/`；对应stdout在`logs/supermimic_a3_R04_trace_render_V0{3,4}_20261023.log`。Mac仅复制四份小PNG/MP4，共1470262字节，分别在本机`data/research/2026-10-23/supermimic/R04_{walk,overhead}_{last_frame.png,comparison.mp4}`，逐文件大小/SHA与收据一致。原模型与NPZ未复制，所有工件仍LOCAL_ONLY。
- 三次监督执行原生退出0，均按原30GiB/86℃保护；D03记录最大usage30596104192字节，末次GPU41℃；综合渲染与监督器采样最高usage30104166400字节/GPU39℃。以上是采样而非连续峰值。

复现入口：`diagnose_model.py --evaluation data/supermimic_for_a3/R04_completion_20261023/eval_final/receipt.json --root /opt/data/private/hechengzhu/ZHY --output NEW_D03.json`；渲染入口为`render_trace.py --evaluation 同一收据 --admission data/supermimic_for_a3/R04_20261023/admission.json --clip 001_walk_front_slow或077_reach_overhead_both --output NEW_RENDER_DIRECTORY`。解释器与环境变量、实际命令数组及退出状态见执行清单；不得覆盖既有结果。

下一选择独立J01动作响应诊断，先验证BPTT依赖的动作敏感性，再决定是否增加联合训练或改变模型拟合。拟定范围为四训练片段各2个保存状态（有效H32窗口的25%/65%位置）、每状态4个固定seed的单位动作方向、±0.05/±0.10两幅度，只扰动首个动作、随后使用原已记录动作，统一H32回放并读取H1/H8/H32响应。8状态×(1无扰动+4方向×2幅度×2符号)×32=4352新控制步，即87040个原1ms物理子步；1环境顺序运行，0新policy推理、0 optimizer，不改R04/旧STOPPED或物理/奖励。

J01须先证明无扰动回放复现原保存轨迹，再比较真实物理有限差分与冻结WM响应/自动微分，分别报告根位置和29主动关节、两幅度一致性及退化响应；无效/终止样本必须留证。它是局部、固定后续动作的模型敏感性诊断，不是完整闭环策略梯度的正确性证明。当前仅登记下一范围，代码、输入/源码冻结、具体种子与数值控制尚未实施，**未运行J01或下一轮训练，也没有动作梯度错误结论**。目标继续active，原三项可用门不变。

## J01实现、数值控制与实际输入冻结（09:22）

状态：实现与准备完成，尚未实际执行物理诊断。新增独立`diagnose_action_response.py`、`audit_action_response.py`及四项科学控制，不改既有28份后端、R04模型或训练入口。服务器控制实际退出0：反向响应即使幅度一致也判为负cosine；零响应不报cosine；非有限值拒绝；提前终止留在分母、不产生无效响应分数。Mac仅AST通过，系统Python没有torch，未假称Mac数值通过。

准备子进程720509实际退出0、4.620秒，0新物理/策略推理/optimizer；[准备监督收据](../data/manifests/supermimic_a3_R04_action_sensitivity_J01_prepare_execution_20261023.json)。[冻结计划](../data/manifests/supermimic_a3_R04_action_sensitivity_J01_plan_20261023.json)绑定R04 final完整SHA、评测/轨迹/参考、28后端及独立入口身份，包含八状态索引和全部方向。PCG64种子20261023；每片段取`floor(fraction*(executed_steps-32))`，fraction=.25/.65，不替换采样。输出独立为`huanyu:/opt/data/private/yuanqi_ops/data/supermimic_for_a3/R04_action_sensitivity_J01_20261023/`；plan状态PREPARED_NOT_EXECUTED。

- 八个H32无扰动窗口须全部复现保存轨迹：42个位置/四元数坐标绝对容差2e-6，41速度坐标2e-5，不使用相对容差；任何一个未过立即停止，尚未准入扰动执行。保持保存的全83状态、前一动作、frame和原物理控制；后续动作固定，不调用policy。
- 每状态四单位方向，首个原动作±.05/±.10，仍按原生[-20,20]裁剪并报告裁剪坐标。1环境/1线程顺序运行，每段32控制步/640原1ms子步；上限4352控制步/87040子步，提前终止保留实际长度和掩码，不替换或补跑。根3和主动关节29分别报告幅度、相对误差和cosine；L2范数≤1e-6米/动作或1e-5弧度/动作时不报对应cosine。
- WM参数冻结，float32模型FD/JVP与实际float32控制边界配对；同权重float64副本对全部方向额外做±1e-4/±2e-4中央差分。双精度数值控制逐坐标要求`abs(FD-JVP)<=1e-7+1e-3*abs(JVP)`；float32/64 JVP要求`abs(diff)<=1e-5+.02*abs(JVP64)`。不同幅度的物理/模型响应一致性单独报告，不把数值控制通过当物理梯度正确。
- `audit_action_response.py`不运行物理，独立从NPZ重建方向/扰动、原状态与动作、基线误差、终止有效性、384响应行/192数值控制行、聚合指标和实际预算。原始数组留服务器，Git只存小JSON。实现冻结与逐张量参数未变的运行时证据不等同于policy成功。

复现命令（工作根`/opt/data/private/yuanqi_ops`，解释器`envs/supermimic_a3_20261023/bin/python`）：

```bash
python script/supermimic_for_a3/test_action_response.py -v
python script/supermimic_for_a3/diagnose_action_response.py --prepare \
 --evaluation data/supermimic_for_a3/R04_completion_20261023/eval_final/receipt.json \
 --root /opt/data/private/hechengzhu/ZHY \
 --output data/supermimic_for_a3/R04_action_sensitivity_J01_20261023
# 上一步已完成，不重复；下面须经原30GiB/86℃监督器单次执行。
python script/supermimic_for_a3/diagnose_action_response.py --execute \
 --output data/supermimic_for_a3/R04_action_sensitivity_J01_20261023
python script/supermimic_for_a3/audit_action_response.py \
 --diagnostic data/supermimic_for_a3/R04_action_sensitivity_J01_20261023 \
 --output data/supermimic_for_a3/R04_action_sensitivity_J01_20261023_audit.json
```

09:22同boot/RTX3090 UUID已实测，无SuperMimic活跃作业；GPU38℃/6348MiB为既有其他占用，容器29327175680字节，OOM14未增。执行前再次核对，外部监督维持30GiB/86℃并给单次900秒壁钟上限，失败保存不自动重试。准备日志`logs/supermimic_a3_R04_action_sensitivity_J01_prepare_20261023.log`。本节与报告入口、源码/清单同次提交并独立核验GitHub后再执行；原件LOCAL_ONLY，目标active。

## J01实际完成：局部动作响应有方向收益，但长期幅度不准（09:30）

状态：完成，策略目标仍未达。实现/预登记已推送`main=aaccc2208959a0da55cff2b2c7dc07931229d50b`并独立核验后，单次执行子进程720709正常退出0；诊断自身111.578秒，监督器114.902秒。八个基线H32窗口均与保存轨迹逐坐标完全相同（位置/速度最大差都0），后续全部128个扰动窗口完整32步，无终止、超时或动作裁剪；实际4352控制步/87040子步，0策略推理/optimizer，WM逐张量未变。

下表为真实物理中央差分对WM float32 JVP的64个响应中位数（8状态×4方向×2幅度）；同方向两幅度并非独立随机样本，不能据此当64独立统计重复。根与主动关节分别比较：

| 时域 | 根cosine / 模型与物理范数比 / 相对L2误差 | 29关节cosine / 范数比 / 相对L2误差 |
| --- | --- | --- |
| H1 | .8961 / 1.2971 / .6382 | .9410 / 1.5642 / .7107 |
| H8 | .9180 / 1.2118 / .6292 | .6492 / .9251 / .8659 |
| H32 | .8183 / 1.5960 / 1.0012 | .6401 / 1.5045 / 1.1307 |

根正cosine比例H1/H8/H32为95.3125%/92.1875%/85.9375%，关节100%/100%/95.3125%；全部响应非退化。方向多数正确，不能判成模型完全不懂动作或反向传播断开；H32幅度与误差也不足以信任长期物理响应。模型FD与JVP结果接近；192组数值控制均通过（各组含两个双精度FD控制及float32/64 JVP对照），排除了本次读出下明显的自动微分数值实现不一致，不能证明完整闭环actor梯度正确。

真实物理的.05与.10幅度响应也并非完全线性：根H1/H8/H32 cosine中位数.9896/.9911/.9829，相对L2差.3133/.2715/.3372；关节cosine .9856/.9723/.9558，相对差.3677/.3583/.4049。接触、摩擦等非平滑响应和离散化均可能影响有限差分，这里没有辨明其各自贡献。两幅度/局部状态的结果不能直接推断所有状态、动作或整条策略梯度，也不是早倒唯一原因证明。

`audit_action_response.py`子进程720870实际退出0，监督7.614秒，独立验收`ACTION_RESPONSE_EVIDENCE_INDEPENDENTLY_VERIFIED`：重新核对全部输入/代码/原数组、种子与采样、基线精度、384响应行、192数值控制行、聚合分母和4352/87040实际计数。此审计0新物理/optimizer。证据：

- [完整动作响应JSON](../data/manifests/supermimic_a3_R04_action_sensitivity_J01_result_20261023.json)、[独立验收](../data/manifests/supermimic_a3_R04_action_sensitivity_J01_audit_20261023.json)、[实际监督执行及资源](../data/manifests/supermimic_a3_R04_action_sensitivity_J01_execution_20261023.json)。
- [诊断stdout](../logs/session_records/supermimic_a3_R04_action_sensitivity_J01_20261023.log)、[验收stdout](../logs/session_records/supermimic_a3_R04_action_sensitivity_J01_audit_20261023.log)。原件`data/supermimic_for_a3/R04_action_sensitivity_J01_20261023/{plan,admission,receipt}.json`、八组`sample_XX_{baseline,native,responses}.npz`；服务器独立审计`R04_action_sensitivity_J01_20261023_audit.json`。复现入口仍为上节固定命令，不覆盖或重跑本次完成结果。
- 监督采样最高容器31812046848字节，GPU含最终快照最高41℃；诊断后29342928896字节，审计后29342470144字节/GPU39℃，OOM14未增。仍是采样峰值，不是连续峰值。没有放宽30GiB/86℃保护或自动重启。

下一先核对actor时域/critic/初始化合同，再登记H8对H32的受控后续试验：依据是J01短时根响应及长期联合误差，而非已证明H8会改善。用同一R04父模型、匹配新交互/更新预算和既定完整片段门，分别评估，避免把更多训练与时域变化混为单一因果。当前尚无该后续代码/输入冻结或训练，R04及旧STOPPED不改；目标active，模型仍不可用，原件LOCAL_ONLY。本节/清单同次提交，GitHub同步以实际推送与独立远端SHA为准。

## R05配对试验实现与冻结：H32对H8（09:44）

状态：实现/控制/准备完成，尚未启动训练。J01结果已发布`main=f62b9e82d517f6bd70bdbb031a4f0925c8fc11b4`并独立核验后，选择一次时域配对试验。它测试整个想象时域的训练效果，**不是只改变actor梯度长度而保持critic数据不变**；时域会同时改变critic目标、minibatch样本数、目标归一化和想象计算量。仍为原WM冻结权重、保留动作/状态梯度的BPTT及原critic/GAE，不替换PPO，也不改28后端/物理/奖励。

实现：`train.py --policy-horizon`仅支持8或32，默认32。新的`--allow-policy-horizon-change`仅允许从完整H32父训练显式初始化到H8；初始化逐字段核验其余合同、加载全部三网络/三Adam并记录唯一变化，禁止同时注入新校准模型。未显式声明仍要求全合同相同。`audit_run.py`独立复核唯一H32→H8声明及原全部张量/Adam/预算/效果，launcher新增配对入口；原历史工件和终态保留，源码历史可由Git恢复，不重跑旧成功阶段。

实际控制：四项合同控制服务器退出0，覆盖默认严格比较、显式唯一变化、无效时域及所有其他合同字段拒绝；Mac七文件AST通过，无本机torch数值结果。`check_horizon_gradient.py`721444实际退出0，监督12.280秒：在J01首个保存状态/原R04全宽三网络上，eager H8+critic bootstrap反向传播的20个actor梯度有限，范数.296353，WM梯度0，三网络逐张量未变。该控制有8次想象policy调用与1次critic调用，0新物理、0 optimizer；batch1/eager不冒称batch64编译训练已通过。

准备进程721516实际退出0、3.516秒，0新增物理/optimizer。[计划](../data/manifests/supermimic_a3_R05_horizon_pair_plan_20261023.json)固定39源码与21输入身份、父R04 final完整SHA、J01独立验收/H8梯度控制/父匹配审计及全部命令；[实际监督收据](../data/manifests/supermimic_a3_R05_horizon_controls_execution_20261023.json)、[梯度结果](../data/manifests/supermimic_a3_R05_horizon_gradient_control_20261023.json)、[合同控制stdout](../logs/session_records/supermimic_a3_R05_horizon_contract_controls_20261023.log)。

| 设置 | R05A_H32对照 | R05B_H8候选 |
| --- | --- | --- |
| 父初始化 | R04 final三网络及三Adam | 相同R04 final，不继承R05A |
| 随机种子 / 环境 / 采集步 | 8723 / 16 / 3000 | 相同 |
| 新物理训练env-step / WM及actor更新 | 48000 / 各1149 | 相同 |
| H / 想象转移数（非真实物理） | 32 / 2353152 | 8 / 588288 |
| 初始Adam WM/actor/critic | 56470/4447/88940 | 相同 |
| 预期最终Adam | 67960/5596/111920 | 相同 |
| 日程偏移起点→终点 | 12000→15000 | 相同 |

两臂采用相同batch64、WM窗口8/每更新10批128、critic每次20提交、填replay128步/无模型再预热，其他配置保持。每500采集步保存。真实交互/更新次数匹配，计算量和critic每minibatch样本数不匹配；改变时域后随机数流消耗及访问状态也会分叉。单种子只是方向筛选，不能当统计稳健或独立最终测试。

`run_horizon_pair.py`先R05A训练→自身step0/final各八段完整评测/独立审计/原可用门，再R05B同流程。任一失败或停止即退出，不自动续训或补跑，不在R05A未完成时并行R05B。两臂总新增训练预算96000env-step、2298次策略更新；四次全八段评测的最大预算134844控制步，提前失败按真实长度计数及完整参考分母保留，不能与训练交互相混。配对比较报告所有片段/组别和原三项可用门；只有一方训练完整回报与完成率均更高时登记探索性偏好，否则记MIXED，仍不能把门未过者称可用。

服务器独立目录`data/supermimic_for_a3/R05_horizon_pair_20261023/`保存plan、admission、各训练launch、status和receipt。两臂输出`R05A_H32_20261023`与`R05B_H8_20261023`，完成目录后缀`_completion`；日志`logs/supermimic_a3_R05{A_H32,B_H8}_{train,complete}_20261023.log`及各完成阶段目录。总控日志`logs/supermimic_a3_R05_horizon_pair_20261023.log`，启动记录`R05_horizon_pair_20261023_launch.json`。

```bash
# prepare已完成，不重复。工作根/解释器同前文，先发布并复核源码/资源。
python script/supermimic_for_a3/launch.py --entry run_horizon_pair \
 --python envs/supermimic_a3_20261023/bin/python \
 --log logs/supermimic_a3_R05_horizon_pair_20261023.log \
 --record data/supermimic_for_a3/R05_horizon_pair_20261023_launch.json -- \
 --execute --output data/supermimic_for_a3/R05_horizon_pair_20261023
```

39源码/21输入与实时boot/RTX3090 UUID在执行准入及各阶段前再核验；30GiB/86℃保护保持，每臂训练壁钟上限10800秒、每臂完成流程3600秒，无自动重启。准备末次29343494144字节/GPU38℃，不是未来实时启动证明。下一完成本节发布/远端SHA核验后实际启动；当前0新增R05训练或效果，目标active、R04尚不可用，原件LOCAL_ONLY。

## R05实际启动与恢复入口（09:54）

实现原提交9f6e2ee首次推送遇远端研究简报更新而拒绝；随后读取并普通合并770d5e5，保留该日报及全部工程记录，audit/diff检查通过后推送`main=fd091590e00c61ea56835658cd6454c68c36cbbe`，独立ls-remote全SHA一致。没有强推或覆盖远端内容。39源码/21输入重新逐一大小/SHA通过；实际启动前同boot/RTX3090 UUID、26320343040字节、GPU38℃/6348MiB、OOM14。该其他GPU占用保留，没有清理无关进程。

总控PID/PGID721781、start_ticks3601225703；R05A H32训练721791、start_ticks3601225810，实际命令/boot与launch均匹配且活跃。见[总控launch](../data/manifests/supermimic_a3_R05_pair_launch_20261023.json)、[总控准入](../data/manifests/supermimic_a3_R05_pair_admission_20261023.json)、[A训练launch](../data/manifests/supermimic_a3_R05A_H32_launch_20261023.json)、[A准入](../data/manifests/supermimic_a3_R05A_H32_admission_20261023.json)。A实际H32/batch64、初始化变化声明空、Adam56470/4447/88940和日程偏移12000均正确；实际保存step0为216004967字节、SHA`9d5cb7664a02c27c49ceca991cd18dc39385bb6d683f85e9ba3582ba092eba20`。这些启动检查不替代后置逐张量/Adam及实际效果审计。

[09:53:59启动快照](../data/manifests/supermimic_a3_R05_startup_20261023.json)最近落盘125/3000采集步、2000新env-step、0落盘WM/actor更新；首次编译尚未完成，不能据此称已发生策略更新或改善。5条落盘指标均有限，总控RUNNING/R05A_H32/train；两个进程真实活跃，无训练/配对终态，B尚未启动。CPU时钟只读核对训练进程累计218.33 CPU秒，未据此改变运行预算或重启。GPU38℃、容器29820243968字节，OOM14未增；读取已完成step0身份后仅对该文件fsync/fadvise，0删除/全局缓存操作，末次29815619584字节。以上为采样证据，不当连续峰值或完成证明；[实际检查命令](../logs/session_records/supermimic_a3_R05_startup_command_20261023.log)。

恢复先读本节和`R05_horizon_pair_20261023/{plan,admission,status,receipt}.json`、总控launch与A训练launch，重新核对boot/PID/start_ticks/命令。A当前原预算运行；若完成，总控将按计划首次执行A完整配对评测/独立审计/原门，通过流程后再从R04开始B H8，最后给配对比较。任一失败保留证据退出，不能重启旧失败阶段或把H8尚未启动记成已训。模型和大型轨迹LOCAL_ONLY，目标active，原三门保持，当前仍没有可用SuperMimic for A3。

首批更新已实际通过：[09:57:41首更新快照](../data/manifests/supermimic_a3_R05_first_updates_20261023.json)记录A采集175步/2800新env-step、19次WM与19次actor更新，actor梯度范数.11421、WM梯度11.23755，全部落盘指标有限。首次编译已越过，尚非3000步预算完成或策略改善证明；总控仍R05A_H32/train，H8没有启动。记录GPU39℃、末次现场40℃，OOM仍14，实际进程/启动tick/命令匹配，继续既定两臂顺序流程。完整来源/时间/资源以快照为准；训练、配对终态及最终评测尚未产生。

## DATA01训练原料盘点、120Hz纠正与R05停止（11:01）

用户要求明确训练原料，实际只读核查20份A3源CSV与M02身份完全相同；固定vendor文档236行明确源120Hz，本转换29行却按30Hz建立时间轴。旧R02/R03/R04/R05均为四倍慢放参考条件，已有真实完成比例/回报/执行秒数保留，但不能称原速动作跟踪或与原速035/论文作效率比较。详见[DATA01完整报告](README_SUPERMIMIC_DATA_20261023.md)及[数据盘点](../data/manifests/supermimic_a3_training_data_DATA01_20261023.json)。本轮未改M02、冻结源码或旧终态，未新增科学计算。

11:01:23重新核对总控/训练PID、start_ticks、boot及命令，仅向自有总控SIGTERM，原handler使子进程完成当前更新保存；11:01:31父Z/子消失。A真实退出0，收据STOPPED_SIGNAL、2002采集步/32032env-step/750次WM及actor更新、4224.596秒；最终PT215996947字节/SHA`e3e9be77c2a136db568c3284af9e7ef1023542aaae96ef250fe8232a24140e09`独立流式复算一致。总控实际退出1，保留原生FAILED/InterruptedError，非预算完成；B和A最终评测未启动。见[停止证据](../data/manifests/supermimic_a3_R05_timebase_stop_DATA01_20261023.json)。不得自动恢复这轮失败/停止阶段。

下一在用户明确数据路线后，用独立数据版本纠正源时间、重算导数/FK速度、核验参考质量与原速基线，再决定新训练。20份A3是最接近当前接口的原料；90份FGP历史拟合须重审，PHUMA两机器人各73076同名片段须A3重定向；人体标签/IMU/仿真转移用途分别登记，不能把197完整会话或146152文件当合格独立A3训练集。所有大型原件LOCAL_ONLY，目标未完成。
