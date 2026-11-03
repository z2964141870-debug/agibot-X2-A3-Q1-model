# Primebot训练中效果监控与提前停止

记录日期：2026-11-03。状态：20:23:39正式监控训练已从零启动；20:31:29核303完整更新/3264512真实转移，三个原进程保持、GPU健康、OOM0。第一次数值对比在1000轮，当前尚无训练后的步态收益结论。

## 用户新决定

用户允许训练期间监控效果，在已经足够好时提前终止。因此250000轮改为上限；其他原生学习公式、奖励、完整网络和学习率日程保留。提前停止明确记为未跑满原配方，不冒充完整复现终点。当前没有新建应用goal或agent，用户采纳仍null。

默认在零更新、1000轮以及每5000轮观测固定三条普通走路动作，各seed0/1/2。三次连续观测均9/9完整、且关节角与骨盆相对身体位置两项RMSE都较同次正式训练的零更新改善，才在完整优化边界保存并提前停止。比较采用每对共同有效前缀，排除初始化及首次终止帧，片段等权。缺失、非有限、失败或不完整观测不能算通过；监控故障保留并继续原训练，不以reward/loss早停。

这仍是“跑通和跟踪改善”门，不等于自然步态合格。每次另外保留真实脚底几何离地、左右抬脚事件、最后2ms求解器承重力与材料点滑速、侧倾、上肢角度/幅度/时间序列。接触力和末态存在子步时差，不声称连续卸载。seed0三个原速50fps实际/参考MP4保留提前跌倒，并在余下参考时长标明FAILED/HOLD。名义无随机扰动回放的三种子不当独立扰动试验。

## 实现与隔离

script/primebot_q1_review包含CPU MuJoCo原生回放、指标、视频、阻塞式观测钩子、独立监控源码副本和正式入口。训练仅在保存检查点的完整更新边界等待CPU观测；观测不修改replay、采样计数或优化次数。前后核训练参数和Python/NumPy/Torch/CUDA RNG哈希，任何变化均报错。原验证源码保持冻结，监控副本仅增加零更新和轮末钩子。

训练结束自动执行初末同固定9次、全入选393条各seed0的逐片评估、保存审计和原速媒体，保留完整片及共同前缀误差。最终全库/媒体/检查点门和Mac本地验收仍先于Morph共享训练。若早停未达410009600实际采样量，等采样量对照记缺失，不因凑比较而继续训练。

首选4096/WM2048/policy2048不变。八轮要求80/8/160实际更新与Adam保存计数，明确CUDA容量OOM才走2048/1024两档，设备错误不作容量回退理由。磁盘预算按真实学习器、rank元数据和满replay，计56次快照、两份replay、8GiB其他工件、25%余量及100GiB空闲保留。

## 已有证据

修正后原筛选441接受/50review/68拒绝，额外48修正审计排除，最终393条2503.92秒，固定三走均接受。冻结清单SHA256896738c54a039160334312bf43158ea2d159eed250a588594ceed0a2e066c167。

2026-11-03 19:40启动验证controller6176；原生16环境24控制步/384转移/240物理子步全部核验，最大限矩误差1.90735e-6Nm、目标误差0、0optimizer。原生guard完成后进入4096 profile，guard6457/worker6462；其首更新编译仍运行，尚不称资源验证通过或正式训练。

固定验证提交49b902142848d62f4a58f4277d1a25235782f74b已独立远端和11文件473021字节Mac/服务器逐大小SHA核验。后续监控提交/CPU真实回放/正式启动以各自收据为准。此前session15017句柄不存在，经原生Windows一次恢复至45124；核原boot/GPU/根，不是服务器或训练重启。

## 完整更新验证结果

2026-11-03 20:09:23北京时间，validation_v01为VALIDATED_READY_FOR_FRESH_FORMAL。4096环境、WM2048、policy2048保留，8轮逐轮日志完整，实际更新和保存Adam均为80/8/160，245760真实转移，58张模型及机器人张量有限。原controller6176/guard6457/worker6462均已退出，GPU空闲1MiB、27度、recovery=None。原生控制、profile、CPU审计均正常回收，原冻结输入检查通过；未使用容量回退。

首轮更新1114.953秒主要含首次编译；后六轮更新耗时中位1.265676秒，25万轮纯迭代外推87.894小时，尚不含重新启动、评估、保存与负载变化。此短测只证明原配方能完成连续更新，不是步态收益。profile checkpoint8为209943195字节，rank shard为393516595字节；实际初始化replay393461760字节、满容量6557696000字节，用于正式整轮磁盘预算。

已有checkpoint0的独立CPU smoke固定9次完成，评测链路PASSED但模型完整0/9，属未训练基线。保留原结果，不重做动力学。新增render_saved仅从这些已存轨迹生成三条原速MP4、探测和完整解码，0新物理步/optimizer。独立监控源码副本已生成，只有零更新及完整轮边界两个观测钩子，验证源SHA保持。

start_validated顺序等待原验证终态、检查已有轨迹媒体、仅一次调用全新正式入口；任何验证或媒体失败保留现场，不启动后续。正式目录为新launch下formal/train/run，profile权重不续。该入口不会自行启动Morph，仍须完整交付和Mac核验过门。

20:16:55首次monitored_launch_v01仅在视频工具检查退出，实际0新动力学/0正式训练，原失败根和栈保留。服务器PATH没有FFmpeg且没有ffprobe；定位到原MorphMimic环境已附带的FFmpeg7.0.2。新版本显式复用此二进制，不安装或改Primebot锁，完整RGB framehash解码核1280x392、50Hz、逐帧PTS/帧数与内容哈希，替代对服务器ffprobe的依赖。修复后使用全新monitored_launch_v02，不恢复失败根，不重做原生或八全更新验证；后续正式启动看收据。

## 正式训练已启动

2026-11-03 20:22:45启动monitored_launch_v02，已有轨迹三原速视频20:23:38检查通过，20:23:39自动进入全新正式训练。controller29435/start2148365、train guard29965/start2153863、worker29970/start2153901，冻结执行b8f920cbba9be3f7247bddbe6535febfbc06186b；该修复已独立40位远端及7份26108字节Mac/服务器逐大小SHA核验，服务器10项测试通过。前阶段73b43f9的13份512481字节三端核验亦已通过。后续报告提交不改变本次执行版本。

根目录：data/supermimic_for_q1/PRIMEBOT_Q1_STAGED01_20261103/monitored_launch_v02/formal/train/run。E4096、WM2048、policy2048、393条2503.92秒、完整网络/原三优化器/原学习日程/从第一轮更新策略；全新初始化，profile权重未续。56份快照加满replay等整轮预算41901082209字节，另保留100GiB空闲，启动时空闲约863GB。

20:31:29现场303完整轮/WM3030/actor303/critic6060、3264512真实转移，104冻结输入重新逐SHA一致。GPU21458MiB/66度/79%/recovery=None，容器总17108746240字节、OOM0；这些只表示训练在推进，不是动作质量。正式零更新checkpoint与观测均保留，参数/RNG前后同，CPU评测完整0/9为未训练基线。1000轮及每5000轮观测自动在完整更新边界执行，持续三次过门才提前结束；异常训练不自动重启。

已有轨迹的三视频已在Mac逐大小/SHA一致并完成ffprobe及全1587帧解码、九格图目视；总1061321字节，1280x392/50fps，普通前走20.64秒、起走3.04秒、停走8.06秒，保留摔倒和标注HOLD。这组是验证视频交付链路的零更新轨迹，不当训练后效果。

- [前走基线视频](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/PRIMEBOT_Q1_STAGED01_20261103/monitored_launch_v02/render_saved/media/walk_forward_loop_001__A022/tracking.mp4)
- [起走基线视频](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/PRIMEBOT_Q1_STAGED01_20261103/monitored_launch_v02/render_saved/media/walk_forward_start_002__A023/tracking.mp4)
- [停走基线视频](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/PRIMEBOT_Q1_STAGED01_20261103/monitored_launch_v02/render_saved/media/walk_forward_stop_002__A024/tracking.mp4)

启动原始收据：logs/session_records/primebot_q1_monitored_formal_launch_20261103.json；Mac媒体收据：logs/session_records/primebot_q1_initial_media_mac_check_20261103.json。原失败根也已取回Mac，未覆盖。

## 下一步

监督原训练进程、1000及每5000轮质量记录、参数与RNG隔离；不热改冻结代码或续用profile，不新增确认环节。训练后入口会完成初末固定9次和全393片各seed0、原速媒体及保存审计，结束在等待Mac交付核验；Morph共享控制器、共享评估和综合比较仍需后续接续，当前未启动Morph。完整五阶段目标尚未完成，不据启动或等待结算goal。
