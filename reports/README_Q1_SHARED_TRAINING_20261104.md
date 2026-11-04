# Q1共享训练与目标监督

2026-11-04用户已更新内存策略，覆盖下方旧4GiB缓存准入与按cache大小例行维护。现有五阶段应用goal实读仍blocked；原q1保持ACTIVE、每10分钟、同一线程。完整目标未完成，用户采纳null；本次只落实内存策略，未创建V04或启动验证/训练，原追加一次E1024八完整更新的决定仍待。

## 总量触发回收策略

用户原话：“ok，以后这样，不看文件缓存，重点要看训练可用内存，当总用量达到30G的时候，再做清理”。沿用原30GiB口径（32212254720B），V04配置取消profile_maximum_initial_cache_bytes，启用memory_pressure_reclaim；cache仍保留诊断记录，但不作独立准入或例行维护触发条件。

以真实cgroup memory.limit_in_bytes与usage_in_bytes记录容器限额、总用量、限额减总量的即时余量及距30GiB的余量，不用宿主机MemAvailable冒充容器可用内存，也不把全部cache减掉后当保证可分配的训练容量。最近只读实核限额34292629504B（31.9375GiB），总量7886807040B，kernel5.4/cgroup v1；cgroup及/proc/sys为只读，未尝试写全局drop_caches。

每2秒原guard检查总量，达到30GiB先检查设备/温度/时限及OOM状态，并通过pidfd/启动tick暂停本guard的子树，避免清理期间CPU继续分配。保留已停止进程状态，finally恢复本次暂停的进程；其他项目计算、身份变化或不可读取的非基础设施进程均停止而不扩大范围。先处理冻结输入，继而在唯一根data/logs/script/reports作有界扫描，跳过符号链接及打开/映射inode；使用既有mincore/FADV_DONTNEED，不读文件内容或删文件。

单次最多15秒处理预算/8GiB advice范围，回收目标总量28GiB；advice不等于实际释放量，记录总量/cache前后实差，不将其解释成唯一归因。回收后仍达到30GiB、OOM变化、恢复失败或30秒内再达阈值，执行原停止路径，避免反复暂停清理。原24GiB初始准入、30GiB最终停止线、OOM0、86度、100GiB磁盘余量、GPU锁和身份核验保留；清理后重查设备。两次原内存检查（含storage扫描后的检查）均接同一策略。

分项采样增加真实限额和即时余量；memory_reclaim.json保留最近32次事件与累计次数，guard终态带SHA身份。未来共享评估phase沿用同一配置，交付清单收录实际存在的分项/回收证据。未触发时不生成虚构的回收事件。

本地五阶段116项：108通过、8因真实Torch/物理依赖及Linux平台跳过；原Primebot14项、既有缓存工具6项全部通过，git diff --check通过。覆盖30GiB边界、cache大而总量低时不清理、storage期间越线、回收无效/OOM/重复压力停止、PID复用和异常恢复、原错误保留、回收后设备复核；另外核父级等待controller/shared_review的精确脚本/模式和启动tick，避免误当外来计算，仍拒绝其他训练或audit。Linux真实pidfd子进程与服务器CPU补验待发布并三端核同后执行；不以夹具等同真实30GiB压力成功或八轮容量通过。[准备收据](../data/manifests/q1_memory_pressure_preparation_20261104.json)。

## 阻塞审计与恢复条件（历史缓存策略）

连续三轮均核同一阻塞：原追加一次E1024八完整更新验证的决定仍待答复，V03 pipeline明确FAILED、五旧PID退出、V04目录不存在。两轮目标续行没有新的用户决定，不能代作授权。第三轮于北京时间05:48:23再经原Windows原生25362会话只读实核，设备身份匹配、GPU1MiB/25度/recovery=None、无计算进程、OOMkill0；容器总7886438400B/cache6698852352B（约6.24GiB）/匿名12824576B/shmem0。cache高于4GiB profile准入、低于原8GiB维护触发，不能强制清理或改门槛。没有仍运行的训练可作为verified wait。

独立评估、比较与Mac交付工具及其实际CPU补验均已完成；后续真实共享回放、36视频和综合比较必须等合格共享检查点。本轮未发现不依赖训练的新必做工作，重复状态、测试或小修不构成目标进展。已满足同一阻塞连续至少三轮且无可推进独立工作的条件，应用update_goal返回blocked；这不是结算完成、暂停监督或改写五阶段范围。[三轮审计与原现场](../logs/session_records/q1_goal_blocked_audit_20261104.json)。

原q1通过应用工具保留ACTIVE/每10分钟/同线程并独立读回保存字段，最新指令已覆盖旧active文字。未变材料保持安静，仅新明确决定、可推进的外部状态变化、新异常或结果才通知。用户作出追加验证的新明确决定后，先核现有goal状态、原设备/锁和实际cache<=4GiB，再按获准范围仅一次同档验证；八全更新、两rank共享参数、CPU保存与磁盘预算全过才从零正式160000轮。仍只有明确容量CUDA OOM才满足512回退，原30GiB线及其他保护不改，V03不恢复，其他自动任务不动。

## 失败终态与前序准备（历史缓存策略）

V04准备补齐了说明与执行不一致的两项：配置显式要求profile初始cache<=4294967296B，并启用record_memory_breakdown。controller在完整Primebot交付门之后、创建实验根之前读实际cache；profile guard完成输入哈希后、创建worker之前再次检查。缓存超线直接拒绝，不释放缓存、不降低环境数，也不绕过原24GiB准入和30GiB运行线。训练配方、393同片数据和两档容量规则保持原样；这次接入不是获准追加验证。

新的GuardMemoryRecorder直接接在现有每2秒guard检查之前，不依赖另起观察进程。它记录cgroup总量/cache/匿名RSS/shmem/脏页与guard子树RSS，独立核PID启动tick；最高总量对应的整条样本单独保存，不能把不同时刻的各项最大值拼成峰值组成。采样读取为顺序读取、并非内核原子快照，进程RSS共享映射可能重复计数；仍不能补造V03旧峰值。JSONL历史上限16MiB，满后停止追加历史但继续更新峰值摘要，退出写OBSERVATION_ENDED；该状态只表示采样结束，不推断训练成功或整个目标完成。V04的CPU审计和未来正式phase也记录分项，4GiB缓存条件仅用于profile。

本地91项CPU测试84通过、7项因缺依赖跳过，原Primebot14项测试另全通过。覆盖缓存边界、输入核验后的二次拒绝且不启动worker、整门失败/缓存超线不建实验、仅profile传入缓存阈值、旧配置不启观察器、原真实MemoryBoundBudget在30GiB拒绝前保存样本、峰值同行组成、PID复用、历史限量后继续峰值采样及结束状态。准备身份见[本轮清单](../data/manifests/q1_shared_v04_admission_preparation_20261104.json)，其中PENDING字段为发布前准备快照，以后续补验收据为准。原追加一次同档验证问题仍待用户回答，未新建或启动V04。

补验完成：3bfd06e8deac5d487e16713686bae5fc5edb3f1e已独立40位远端核同，12份556060B在权威Git、Mac主根和25362逐大小SHA一致。服务器29项CPU测试全部通过、无跳过，用时2.172秒；9项观察输入前后SHA同，实际guard/observer导入路径正确。只对本次CPU检查进程做两次实时分项采样，正常写OBSERVATION_ENDED；CUDA未初始化、0积分/优化。实际cache6698295296B高于4GiB，准入函数按设计拒绝；低于原8GiB维护触发，不强制释放。GPU1MiB/25度/recovery=None、无计算进程、OOMkill0，V03仍FAILED、五旧PID退、V04目录不存在。本轮不是八完整更新验证或正式共享训练。[CPU原结果](../logs/session_records/q1_shared_v04_admission_cpu_v01_20261104.json)、[发布与三端核验](../logs/session_records/q1_shared_v04_admission_publication_20261104.json)。

现有应用goal再次实读active；原q1经应用工具更新监督材料，并独立读保存字段ACTIVE、每10分钟、同一线程。没有创建替代goal或启动agent，追加同档验证仍待原问题，不因等待或单阶段交付结算整个五阶段目标。

V03于北京时间01:37:29被原30GiB守卫正常回收，01:37:30整个pipeline明确FAILED。触发总量32271097856B（约30.055GiB），cgroup OOMkill仍0，worker回收-15；截至01:40独立核1761冻结输入全部SHA未变、五原PID全部不存在、GPU1MiB/25度/recovery=None且无计算进程。没有正式训练或checkpoint，八轮完整更新尚未完成。两rank日志各409600预热明确已采集；旧worker JSON仍RUNNING/计数0是未刷新的原始收据，不能当作当前在运行或证明内存中零优化。[失败与独立核验](../logs/session_records/q1_shared_v03_memory_failure_20261104.json)。

退出后容器20459925504B，其中cache18980741120B、匿名RSS29421568B、shmem0。停止瞬间没有memory.stat分项，故不能据退出后值判定峰值仅由cache造成，也不能声称是CUDA显存OOM。原choose_profile只允许明确容量CUDA OOM降档；本次没有进入512或正式训练。原失败目录、日志及RUNNING旧收据均保留，以pipeline/guard终态和PID核验解释。

按既有授权于01:41:55执行一轮默认缓存维护：最多8GiB advice/120秒/1800秒冷却，实际63.498秒，cache18980741120→11153068032B，观察下降7827673088B（约7.290GiB）。工具收尾total13084680192B，01:47工具退出后12439904256B，shmem0/OOM0；未删除文件、未读大文件内容或改资源线。[维护原收据副本](../logs/session_records/q1_shared_v03_cache_maintenance_20261104.json)。尚未降至4GiB目标；后续只按原冷却规则继续，不连续绕过限量。

新增独立只读memory_watch.py，每2秒记录cgroup总量/cache/匿名RSS/shmem/脏页及原controller子树RSS，核PID启动tick、boot和唯一根；最大10800秒/16MiB，观察超时不判训练停止，不向进程发信号，也不写训练输入。6项本地CPU测试通过，覆盖PID复用、扫描中复用、退出、观察超时、错误边界和原记录保护；Linux实测尚待发布后。原V03失败前没有这些峰值分项，新增工具不能补造旧峰值。

补验完成：935d3978b230f14fc71c9a46c98de8940c299c37已推送并独立40位远端核同，9文件495452B在Git工作树/Mac主根/25362逐大小SHA一致。服务器标准库6项CPU测试全部通过；对本次CPU诊断进程做2秒只读实采，5条样本，正确返回OBSERVATION_TIME_BOUND_REACHED且原进程仍在，未发信号/导入Torch/新增物理或优化，前后GPU无计算。此为采样器验证，不是新训练或补测V03峰值。[Linux实测](../logs/session_records/q1_shared_memory_observer_linux_20261104.json)、[发布与三端收据](../logs/session_records/q1_shared_v03_memory_publication_20261104.json)。

同档V04配置已准备，唯一变化为独立新输出目录，仍各1024/WM512/motion-KP1024、393同片、八完整更新；通过共享/CPU保存/实存预算后才从零160000。拟在空闲cache实际<=4GiB、原身份/24GiB准入/30GiB运行/锁等复核后仅一次新验证，并同步只读采样。当前未启动；本次原phase明确automatic_retry=false，非CUDA OOM不满足自动降档，需用户决定是否追加这一次同档验证。目标和十分钟监督保持active，等待时继续评估工具，不结算目标。

02:17:46冷却到期后按原默认再做一轮空闲缓存维护，19.064秒达到8GiB advice限量；实cache11233218560→5713190912B，观察下降5520027648B，工具收尾total7537897472B/OOM0，仍高于4GiB目标。原8GiB触发阈值、120秒和1800秒冷却不改，未删除文件/读取大文件内容/改保护线；不把advice字节当作实际释放量。[本轮原收据](../logs/session_records/q1_shared_idle_cache_20261104_022300.json)。这是正常维护，没有触发额外训练或验证。

## 启动快照（历史）

V03已于01:18:47实际启动，执行冻结152f5e1afacb3e33bffbd1f191827a89d8d02071，独立40位远端及15文件511071B三端大小SHA均通过，新Runner环境和实际CPU保存补验2/2通过。新根MORPHMIMIC_Q1_CONDITIONAL_SHARED01_V03_20261104，各1024环境/393条共同参考，仍是八全更新资源验证，尚未正式训练。[发布](../logs/session_records/q1_shared_v03_publication_20261104.json)、[启动](../logs/session_records/q1_shared_v03_launch_20261104.json)、[CPU](../logs/session_records/q1_shared_v03_cpu_20261104.json)。

01:20:10只读核总controller47219/start3924514、guard47220/start3924727、parent47225/start3924829、G147226/Q147227均start3924841，五PID均在；pipeline VALIDATING、guard RUNNING、GPU7211MiB/40度/recovery=None、容器26997280768B/OOMkill0。随后两rank日志已越过原生导入和真实环境构造，完成各409600预热，首完整更新仍在编译。收据未刷完整轮不等于内存中零更新；不热改执行输入或插入额外回放。[有时间戳现场](../logs/session_records/q1_shared_v03_observed_20261104.json)。

应用get_goal再次实读active，q1保存字段ACTIVE/每10分钟/同线程；监督指令已指向V03根、冻结152f5e1和本页，不再盯已结束的Primebot/V01/V02。所有八轮、两rank实核及CPU保存审计通过后，原controller自动从全新初始化进入160000轮；本次没有因用户追问模式另建目标或启动agent，不在等待期间结算目标。

前序修复：源码预算修复4fc06b98df269b0b62b2e0ee053162f10a9fc73a已独立40位远端及10文件499878B三端核同，5项服务器CPU检查全过。V02于01:07启动，但两rank在bootstrap导入GitPython时因新Runner遗漏原生Git工具环境而退出；原五PID已退出、1761冻结输入未变、OOM0/GPU健康，0物理转移/优化，不解释成容量失败。新V03仅补与原生入口一致的GIT_PYTHON_GIT_EXECUTABLE，并使用独立新根；两型真实bootstrap＋原Trainer CPU导入均已通过、无CUDA上下文。新增子进程环境回归通过，本地controller共8过1待Torch；原真实保存CPU检查已经补过。原V01/V02失败完整保留。[V02终态](../logs/session_records/q1_shared_v02_import_failure_20261104.json)、[真实导入](../logs/session_records/q1_shared_native_imports_20261104.json)、[前次发布](../logs/session_records/q1_shared_source_fix_publication_20261104.json)。

## 首轮与转接

Primebot已在10000轮正常早停，固定9/9、全393/393和Mac完整交付通过；完整转接门重算passed/authorizes_morph_training=true。跟踪改善不等于自然步态合格，低脚底离地和承重滑移仍在。详见[首轮结果](README_PRIMEBOT_Q1_FINAL_RESULTS_20261104.md)。

## 首次共享入口失败

原根data/supermimic_for_q1/MORPHMIMIC_Q1_CONDITIONAL_SHARED01_20261103于00:45:59启动控制器、00:46:04结束。e1024/profile/guard.json明确STOPPED_STORAGE、worker_started=false、worker_reaped=true；没有启动训练worker，没有新增物理转移或优化。pipeline保留FAILED和原异常，不解释成容量CUDA OOM，也没有进入512回退或正式训练。

来源是通用Primebot guard的64MiB源码目录上限；实际MorphMimic发布目录分配142503936B、逻辑141858407B、341个inode。现场磁盘空闲525553401856B；这是源码目录预算不适配，不是磁盘耗尽。新GPU UUID和boot匹配，25度/recovery=None/1MiB，cgroup OOMkill0。原始guard、status和trace继续保留在原失败根。[诊断收据](../logs/session_records/q1_shared_source_budget_20261104.json)。

## 修正范围

guard增加可选maximum_source_bytes，原默认64MiB保持，参数必须为正整数且最多256MiB。共享V02配置显式设置256MiB，覆盖实测约136MiB的固定发布树，仍继续统计整个目录并拒绝超限。100GiB磁盘余量、24GiB启动/30GiB运行、OOM0、86度、GPU锁和设备身份检查保持。

controller将配置传入共享guard并记下源码预算。新配置script/q1_five_stage/conditional_shared_v02_20261104.json使用独立MORPHMIMIC_Q1_CONDITIONAL_SHARED01_V02_20261104根，其余数据、网络、环境和训练配方不变。首次失败发生在worker创建前，修复后仍按原授权完成首次实际共享验证；不续失败根或profile权重。

本地4项源码预算回归、7项controller与10项contracts检查通过，共21项；1项真实Torch保存测试本机缺Torch跳过，发布后服务器CPU补验。新测试复现原64MiB拒绝、256MiB接受实测树、超过256MiB仍拒绝、非法参数拒绝，同时核原资源线不变。

## 执行顺序

发布修复、独立40位远端和三端工件核验后，服务器CPU检查实际目录预算和既有门。通过后启动V02各1024环境、393条共同数据、8轮完整更新；两rank共享参数、五优化器80/8/160/8/160、CPU保存审计与实存磁盘核算全部通过，控制器才自动全新初始化正式160000轮。只有明确容量CUDA OOM且正常回收、输入同、OOM0和GPU健康才允许各512/320000；正式异常不自动重启。

新启动命令在唯一ZHY根执行，VERIFIED_FULL_COMMIT必须来自本阶段独立远端核验：

```sh
data/environments/morphmimic_original_20261029/bin/python -u script/q1_five_stage/controller.py run --config script/q1_five_stage/conditional_shared_v02_20261104.json --code-commit VERIFIED_FULL_COMMIT
```

上面命令及“尚未启动V02”为首次修复准备历史；V02现已失败，下一次入口为script/q1_five_stage/conditional_shared_v03_20261104.json、独立MORPHMIMIC_Q1_CONDITIONAL_SHARED01_V03_20261104根。实际执行与发布身份以后续launch/publication收据为准，不把计划写成训练进度。原生Q1 motion/KP、G1初末评估、原速媒体和综合比较仍须完成。早停Primebot缺少等410009600曝光检查点如实缺失，不追加训练。

只复用hp_han Windows原生SSH至25362的既有会话78774；本次连接可用，未重建SSH或重启服务器。目标模式持续监督当前实际pipeline、guard、PID/starttick、完整metrics、GPU/cgroup和保存；无变化安静，有新异常、质量变化或完成主动报告。不开agent，不改其他自动任务。
