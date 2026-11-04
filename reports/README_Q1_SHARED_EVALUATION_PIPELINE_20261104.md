# Q1共享评估、比较与本地交付串行入口

2026-11-04 12:11当前：V04已在11:06:15正式从零启动，原追加验证授权已解决，旧缓存门和下方“V04未启/等待训练授权”仅历史。现在因同一SSH观察故障连续三目标轮而将原完整goal实记blocked；现有Windows会话仅PowerShell，原一次自动恢复已用，V04实时状态UNKNOWN，不据失联断言训练停止。评估/交付代码和实际CPU验证已齐，真实共享回放、权重和媒体需恢复原25362连接后按原门核验；q1仍ACTIVE十分钟同线程，等待新的明确连接恢复指令，未变不重复测试或通知。[当前训练与连接状态](README_Q1_SHARED_TRAINING_20261104.md)、[三轮审计](../logs/session_records/q1_shared_v04_connection_blocked_audit_20261104.json)。下方05:48及更早为已完成准备的历史快照。

2026-11-04 05:48当前状态：现有五阶段goal经三连续目标轮阻塞审计已记blocked，原q1心跳仍ACTIVE每10分钟、同一会话。独立评估和交付工具及实际CPU验证已完成，真实共享比较必须等合格权重；没有可继续独立推进的必做工作。共享V03的30GiB内存失败保持，追加一次同档验证的原问题仍待答复，cache约6.24GiB也未达4GiB准入；V04和正式共享训练未启动。目标范围未缩减、未完成或暂停监督，后续状态见[当前训练页](README_Q1_SHARED_TRAINING_20261104.md)及[阻塞审计](../logs/session_records/q1_goal_blocked_audit_20261104.json)。下文active均为历史快照。

## 已完成的代码

综合报告生成补验：现有Primebot实际等曝光收据只含status，原writer可用；但标准exposure_comparison输出还含target_q1_transitions和additional_training_authorized，原dict(**exposure, ...)会因重复关键字抛TypeError。新增真实write_conclusions文件输出回归已先复现该错误，改为从mapping复制再显式覆盖目标和禁止追加训练字段；不改原收据或比较门。测试分别输入旧极简收据和标准完整记录，实际写出14项比较的JSON/Markdown，确认缺共同前缀保持Unavailable、完整配方与等曝光分开、102563840/410009600暴露量不混同、不授权补训或结算goal。共享比较数据为明确的合成测试夹具，不是实际共享评估结果。原串行测试曾替换write_conclusions，新增测试覆盖了该缺口。本地92项85通过、7项缺依赖跳过。[准备清单](../data/manifests/q1_shared_conclusion_writer_preparation_20261104.json)的PENDING为发布前快照，现以补验收据覆盖。

实际补验完成：b64a7ef403723a394c30c5c2241253a6a83fd67b已独立40位远端核同，7份516836B在权威Git/Mac主根/25362逐大小SHA相同。服务器20项CPU检查全部通过，用时0.769628秒；9项输入前后SHA同，4份原Primebot记录逐大小SHA核对原交付清单。另用真实原审计/固定及全库结果/旧极简收据，和从实际102563840转移审计生成的完整标准记录，分别执行writer；两份临时合成共享比较报告均含14比较、393库及9固定元数据，保留等曝光不可用/缺前缀Unavailable/禁止补训/goal未完成。临时输出已删除，只保留验证收据，未当作真实共享交付。[CPU原结果](../logs/session_records/q1_conclusion_writer_cpu_v01_20261104.json)、[发布与三端证据](../logs/session_records/q1_conclusion_writer_publication_20261104.json)。

本次进程未导入Torch/MuJoCo，无CUDA或新积分/优化。收尾原boot/GPU健康，GPU1MiB/25度/None、无计算、OOM0；V03仍FAILED、五旧PID退、V04目录不存在，追加同档验证的原问题仍待。现有goal与十分钟监督保持，不因完成报告工具而结算整个五阶段。

补齐完整片步态汇总：`comparison.py`和`native_compare.py`的每侧结果均附足部p95/参考差、>10mm事件/交替、承重滑速、倾角、上肢RMSE及幅度差的逐片等权分布；`shared_delivery.py`在综合README输出相应诊断表。失败片不进入完整片分布，缺载荷不填零，缺完整片诊断拒绝比较。使用现有Primebot788份初末JSON及原SHA清单得到全库最终306/393无交替事件、308/393双脚p95<=5mm，基线详见[首轮结果](README_PRIMEBOT_Q1_FINAL_RESULTS_20261104.md)。本地81项74过7依赖跳过；50afcfd独立40位远端及12份559976B三端后，04:43服务器15项纯CPU检查与原JSON独立汇总全部通过/.989702秒，分组逐值相同/八源前后SHA同。未新积分或优化，V04仍未启动。[本次发布与服务器证据](../logs/session_records/q1_library_diagnostics_publication_20261104.json)。

新增`script/q1_five_stage/shared_review.py`。入口先核正式训练pipeline、固定计划、双机型初末CPU审计、两rank完整训练与共享一致性、393同片数据和原Primebot/Mac交付门，再创建独立评估目录。未通过时不导入策略、不创建Runner或启动物理。保留训练pipeline、计划、CPU审计和数据清单的独立小副本，后续控制器写状态不会丢失评估时的证据。

评估严格串行，共24个回放阶段：共同Primebot物理环境下Q1的motion/human，原生mjlab下G1/Q1的motion/human，各自初始/最终、固定三走三种子/全393条seed0。固定seed0均保留原速视频，合计36份；失败终止帧和hold保留。视频不是独立随机扰动测试，不能将三种子名义重复解释为泛化。

回放后串行执行14项比较和两型训练曲线。原生初末只在同机型、同接口中比较；Q1直接对照沿用相同Primebot CPU物理环境，KP另列。完整片误差与共同有效前缀分开，缺前缀明确缺失。G1初末不能替代等训练量的G1单体受控实验，见过的Q1不称未见本体零样本。

`shared_delivery.py`核全40阶段成功回收和输入身份、4824例逐片JSON/脚底姿态CSV/上肢时序CSV、36视频及各三张联系帧、比较、曲线和综合报告，再生成本地交付清单。大NPZ、权重和场景MJB仅SERVER_LOCAL_ONLY。Mac独立逐文件大小/SHA、完整FFmpeg解码和每个确切视频SHA的目视收据全部通过才写VERIFIED；不装ffprobe。目标是否完成仍须后续审阅实际结论，不由工具自行结算。

每项回放的轨迹和媒体放在该guard实际统计的输出目录内；整轮按冻结帧数计算轨迹和原始RGB视频上界，启动前另保留100GiB磁盘。沿用原24GiB准入/30GiB运行/86度/设备身份/GPU锁/回收保护，不失败自动重试。V04仅增加未来正式完成后自动接入评估的配置，原各1024/160000训练配方不改；没有据此执行V04。

同物理Q1入口补齐正式完成和双机型审计门，并将渲染身份同步写回逐片结果。策略加载器额外核保存的154维形态描述等于实际固定资产，避免仅mask一致或两个保存副本同时错误时仍放行。CPU反例会同时修改两个形态副本；恢复原值使用clone，不用加减回算。

## 验证与限制

04:06实际CPU兼容性补验通过：80a2cdbb13d78e2ea7246efd8edbb3f1beddc1c8独立40位远端及五文件485658B三端核同后，Morph Python读取真实Primebot10000轮最终检查点，11.683秒完成三条冻结固定参考1032/152/403共1587帧的Q1姿态/FK和人体sidecar核验，269/93/670/1684输入与原基线三次CPU前向有限、RNG保持。十三源前后SHA同，物理积分及优化0；GPU1MiB/24度/None/无计算、OOM0。构造/reset被允许，所有积分入口在测试中被显式拒绝。此为基线及接口兼容性，不是共享权重或共享质量评估。[原CPU收据](../logs/session_records/q1_common_plant_cpu_v01_20261104.json)、[日志](../logs/session_records/q1_common_plant_cpu_v01_20261104.log)。

这次补验同时暴露比较链路的依赖差异：Morph环境实际Torch2.11.0/MuJoCo3.11.0，而原Primebot锁定Torch2.7.0/MuJoCo3.6.0。原代码若直接在Morph Python中执行共同物理回放，会改变求解器版本；当前没有共享检查点或已执行的错误共同回放，不需要重算既有Primebot结果。

已修共同入口：`shared_review.py`明确用原Primebot Python运行物理、终止、诊断和媒体；`q1_native_review.py`在正式完成/双审计门后核解释器及2.7.0/3.6.0，另起`policy_process.py` CPU子进程，仅在原Morph环境计算native观察/目标/动作。JSON逐控制帧传输79状态和22/29动作，模型数学及完整网络不改；真实共享检查点仍须先过原门。子进程有消息长度/超时/错误传播/退出回收保护，无重试，stderr留在guard实际统计的评估目录并进入交付清单。

04:22真实两环境CPU补验完成：aad854ebf62547b6afef62db2e7906ab9b1170c3已独立40位远端核同，15文件577714B在权威Git/Mac主根/25362逐大小SHA相同后，服务器20项测试全部通过，进程40.838秒（unittest主体38.434秒），十八输入前后SHA未变。原Primebot进程实测Torch2.7.0/MuJoCo3.6.0，策略子进程实测Torch2.11.0/MuJoCo3.11.0，三固定片×motion/human×前三帧18次动作与同权重直接CPU数学逐值相同，最大差0；子进程reaped0、两环境CUDA均未初始化，物理积分及优化0。

该测试使用明确标注的小型合成native权重，观察取自真实固定参考与原CPU reset，未加载不存在的正式共享检查点，不能当作共享策略质量或动力学验证。正式门仍在任何策略子进程/物理执行之前；新增退出/超时/非JSON/错误响应/回收与解释器选择覆盖。Mac本地74项67通过、7项依赖跳过；其中本次涉及的实际依赖与通信已在服务器20项中补验，不重复既有物理、媒体或保存审计。[真实两环境CPU收据](../logs/session_records/q1_policy_bridge_cpu_v01_20261104.json)、[完整测试日志](../logs/session_records/q1_policy_bridge_cpu_v01_20261104.log)、[本阶段发布证据](../logs/session_records/q1_common_plant_publication_20261104.json)。

收尾实际设备仍为原boot/GPU，GPU1MiB/25度/None、无计算进程；cgroup总7885058048B/cache6697660416B/匿名11165696B/shmem0/OOM0。cache低于原8GiB触发线，未强制维护。正式共享训练和V04未启，追加一次同档验证的原问题仍待答复，goal及十分钟监督继续。阶段代码和CPU证据已完成；真实共享回放、GPU接触读取和媒体仍待正式共享检查点。

本地68项测试，63通过、5项依赖跳过；新增串行替身完整运行覆盖24回放→14比较→2曲线→待Mac，首回放失败覆盖无重试/无后续比较。另验正式未完成时不建目录、guard回收/输入变化/设备错误、预算包含全部输出、缺片或重复/大型交付成员、视频联系帧缺失和本地SHA变化。`git diff --check`通过。

正式共享检查点尚不存在，真实共享策略回放、GPU求解器接触读取、36视频渲染和Mac验收均未执行，不称五阶段已经实际跑完。既有真实CPU运动学结果未重复。

03:40:56实际补验完成：执行源码7965f2bb054ebea09d05260ef944d8dbf6c95493已推送并独立40位远端核同，17份585412B在权威Git/Mac主根/25362逐大小SHA一致后，服务器16项CPU检查全部通过，进程14.742秒（unittest主体12.569秒）。实际v0.1策略加载、冻结资产形态、motion/human合成输入与原加载器逐值一致，normalizer/RNG和缺失关节mask保护均过；十一输入前后SHA未变，CUDA设备屏蔽、无物理积分或优化。原V03真实FAILED pipeline另作负例，正确拒绝且没有创建评估目录。[CPU原收据](../logs/session_records/q1_shared_evaluation_cpu_v01_20261104.json)/[测试日志](../logs/session_records/q1_shared_evaluation_cpu_v01_20261104.log)/[预期失败门日志](../logs/session_records/q1_shared_evaluation_real_failed_gate_v01_20261104.log)。

CPU后原boot/GPU/唯一根保持，GPU1MiB/24度/recovery=None/无计算进程、原五PID不存在、cgroup OOM0；总7307677696B、cache6119419904B、匿名RSS13115392B、shmem0，均为本次收尾快照，不能反推V03峰值组成。cache低于默认8GiB触发线，未强制缓存维护，也未启动V04。现有goal再次实读active，十分钟监督保持。[发布、三端和CPU证据](../logs/session_records/q1_shared_evaluation_pipeline_publication_20261104.json)。

Primebot10000轮正常早停及9/9、393/393交付保持，仍低离地/承重滑移，不能称自然步态合格。其102563840转移达不到410009600，当前等采样量明确不可用；本入口保留该负面结论，不额外训练补齐。若后续等曝光证据状态改变，先审计和规划对应比较，不静默忽略。

## 入口

V04的`evaluation_runtime_config`指向既有`data/manifests/primebot_q1_monitored_config_20261103.json`。原controller只有八全更新、共享参数、CPU保存和正式训练全部过门才调用本入口，评估失败单独记`SHARED_TRAINING_COMPLETED_EVALUATION_FAILED`，不把已完成正式训练改成未完成或自动重训。

如独立调用，`FULL_COMMIT`须独立远端核验：

```sh
data/environments/morphmimic_original_20261029/bin/python script/q1_five_stage/shared_review.py --config SHARED_CONFIG --pipeline COMPLETED_SHARED_PIPELINE --runtime-config data/manifests/primebot_q1_monitored_config_20261103.json --output NEW_SERVER_EVALUATION_ROOT --evaluation-commit FULL_COMMIT
```

实际执行后工具保存`evaluation.json`、`conclusions.json`、`README_RESULTS.md`和`delivery_manifest.json`。独立Mac验收入口为`shared_delivery.py --manifest ... --local-root ... --visual-receipt ... --output ... --ffmpeg ...`，不得伪造目视收据。代码发布和CPU兼容性已经完成；实际策略回放须等待共享正式训练产生合格检查点，追加一次V04验证的用户决定仍待答复。
