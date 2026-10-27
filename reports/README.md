# YUANQI 项目状态与重要文件

2026-10-28 02:36：[RF12负面结果与本地交付完成](README_SUPERMIMIC_Q1_RF12_RESULTS_20261028.md)。3503.475实训秒/812轮，首个三重复物理门根XY误差升至RF11的2.65至2.82倍、上肢增加13.3至14.0%、验证4/6/3条失败，按physical_regression停止，B/C及加训未执行。02:28全pipeline正常完成、原进程退出/GPU释放；37保存/12媒体检查通过，43小工件48549915字节已Mac大小SHA核验，初末54秒MP4和六格/PPO诊断齐备。记录发布核验后暂停本轮监督，不追加训练。[结果原值](../data/manifests/supermimic_q1_RF12_results_20261028.json)/[完成收据](../data/manifests/supermimic_q1_RF12_completion_audit_20261028.json)。

2026-10-28 02:15：[RF12阶段A物理门明显退步](README_SUPERMIMIC_Q1_RF12_PPO_20261027.md)。实训3503.475秒/812轮/79822848转移，根XY误差为RF11的2.65至2.82倍，上肢增加13.3至14.0%，验证4/6/3条失败，固定普通走三次均不完整。整组改善0/3、严重退步3/3，已决定stop/physical_regression，B/C及加训未执行。优化暂停、自动非训练六格/视频交付中，尚未全pipeline结束；不等待视频才报告量化结论。[小收据](../data/manifests/supermimic_q1_RF12_stage01_20261028.json)。

2026-10-28 01:49：[RF12桥接恢复](README_SUPERMIMIC_Q1_RF12_PPO_20261027.md)完成；原controller/worker、boot/GPU一致，三状态RUNNING，A阶段2613.658实训秒/606轮，1123冻结来源大小SHA全相同，资源正常且未重启训练。首次3500秒物理评测尚待，继续原监督；下方断线/缺认证均为历史。[现场收据](../data/manifests/supermimic_q1_RF12_connection_unknown_20261028.json)。

2026-10-28 01:44：[RF12连接恢复尝试](README_SUPERMIMIC_Q1_RF12_PPO_20261027.md)已按用户新指示执行；Windows可达但桥接账户免密认证被拒绝，缺该账户登录信息，现场仍UNKNOWN。训练/冻结来源/预算未改。前次中断记录9b6c658已发布并独立核远端、Mac双副本大小SHA相同，服务器镜像待恢复。

2026-10-28 01:36：[RF12监督连接中断](README_SUPERMIMIC_Q1_RF12_PPO_20261027.md)。18873无监听，容器与Windows桥控制socket均不存在，现场UNKNOWN；最后01:25快照为A阶段1237.220实训秒/287轮且资源正常，不推断现在停训或继续。一次有界核查后不重启训练、不改冻结来源/预算，恢复后先核现场；25386记录同步待连接恢复。[收据](../data/manifests/supermimic_q1_RF12_connection_unknown_20261028.json)。

2026-10-28 01:07：[RF12真实仿真PPO与分阶段训练](README_SUPERMIMIC_Q1_RF12_PPO_20261027.md)已启动4096环境，冻结684548c发布/独立核远端、22文件三端匹配；175.025实训秒/42轮，已完成critic预热并开始actor，三状态/1123来源/资源正常，尚无新效果结论。约1至3实训小时按固定三重复物理门决定额外5小时，总<=8小时，原监督已ACTIVE。AP03的883训练/原30验证及AMASS179.48GB原始motion清理完成，未恢复旧实验或agent。[启动收据](../data/manifests/supermimic_q1_RF12_launch_observed_20261028.json)。

2026-10-27最终：[RF11结果与交付](README_SUPERMIMIC_Q1_RF11_RESULTS_20261027.md)。4096环境实训3500.266秒，全pipeline COMPLETED；验证0/30提前终止（RF09近一小时2/30、RF10 29/30），但RF09共同前缀根XY0.07904→0.11446m、上肢0.93760→0.93577rad，存活恢复而跟踪仍差。固定抬手腕高不足，舞蹈仍失败。六格图/初末54秒多类MP4及29文件大小SHA交付，36训练/19结果/12媒体检查通过；结果已发布并三端核验，22:45:11本轮监督实际PAUSED，不续训或再开。[完成回执](../data/manifests/supermimic_q1_RF11_completion_audit_20261027.json)。

2026-10-27 21:55：[RF11训练完成/评测交付待续](README_SUPERMIMIC_Q1_RF11_SAMPLERONLY_20261027.md)，实际3500.266秒/3271轮，36项保存审计通过且正常reap。首轮验证0/30提前终止、完成100%，训练24/300提前终止；尚待共同前缀根部/上肢分析、六格和本地54秒MP4，不把存活等同行走跟踪成功，不追加训练。

2026-10-27 20:57：[RF11已开始](README_SUPERMIMIC_Q1_RF11_SAMPLERONLY_20261027.md)，20:56:44起实际训练，4096环境/恢复RF09配方并仅修尾bin，obs80原生转移通过。冻结7a3567c已发布并核部署，58轮/WM580，最多3600秒；效果与六格/54秒MP4待终后，不用训练日志替代结果。

2026-10-27：[RF11计划与预检](README_SUPERMIMIC_Q1_RF11_SAMPLERONLY_20261027.md)。用户新授权再训练一小时；恢复RF09原观测/奖励，仅保留尾bin合法转移修复，4096环境从零/≤3600秒。CPU五组和原生16env转移通过，正式训练待发布启动；RF10训崩的唯一因果未隔离，不保证修复即改善。

2026-10-27最终：[RF10结果与交付](README_SUPERMIMIC_Q1_RF10_RESULTS_20261027.md)、[MG05完成审计](../data/manifests/supermimic_q1_MG05_completion_audit_20261027.json)。25386已连通；唯一4096环境试验实际3500.993秒，全pipeline COMPLETED，验证29/30失败（RF09近一小时2/30），两次复评仍退步，已确认剧烈动作指令振荡但组合的唯一根因未隔离。六格图/初末各54秒多类MP4及30小文件大小SHA全部交付，不追加训练。最新恢复入口集中在[CURRENT_STATE](README_CURRENT_STATE.md)，下方RUNNING/待评测均为历史。

2026-10-27 19:56：[RF10训练完成且明显退步](README_SUPERMIMIC_Q1_RF10_C01_20261027.md)，3500.993秒/3284学习轮，36项保存审计全部通过。330片最终验证29/30提前终止、完成42.88%，旧一小时基线2/30/95.70%；不追加训练，不用早倒截断造成的上肢均误差下降宣称改善。初末视频/六格/共同前缀与动作罚项、原生方向复核进行中。

2026-10-27 18:55：[RF10实际启动](README_SUPERMIMIC_Q1_RF10_C01_20261027.md)，4096环境/C01/同300+30参考/从零；18:47:19进入训练，冻结83dcc76已发布独立核SHA、21文件三端相同。609轮/WM6090/actor109及真实进程/资源正常，≤3600秒预算不变；初末效果/六格/长MP4未完成，不以训练日志宣称改善。[启动现场](../data/manifests/supermimic_q1_RF10_launch_observed_20261027.json)。

2026-10-27：[C01完整profile与RF09近一小时基线](README_SUPERMIMIC_Q1_MG05_NATIVE_CANDIDATE_20261027.md)通过，4096环境/102维转移、WM/actor/H32/critic有限，峰值分配6.35GB、worker正常回收；基线ckpt3500内部3424.267689秒，330片评测完成。[RF10计划](README_SUPERMIMIC_Q1_RF10_C01_20261027.md)待源码发布/远端核验后独立从零一次，≤3600秒；正式训练尚未启动。

2026-10-27 18:32：[MG05原生诊断/C01候选](README_SUPERMIMIC_Q1_MG05_NATIVE_CANDIDATE_20261027.md)已取得真实受力/滑速与H32闭环方向；普通走已有单支撑/切换、存在较大接触点滑速，H32方向弱，未锁定唯一因果。重复不观察rollout亦有差异，496步接触读取未改所选状态。候选实际构建及CPU5组契约通过，4096完整profile进行中；MG05工具active，原paused条目为历史。尚无新正式训练，无新agent。

2026-10-27：[CS02恢复25386](README_Q1_CONTAINER_RESTORE_CS02_20261027.md)完成连接、共享数据与CUDA/模型检查；RF09权重/清单SHA保持、330参考无缺失，3090已换卡、boot更新/OOM0、盘余约1.65TB。停服前V02小结果与原生检查日志已取回Mac并核SHA；日志实际有后端对象清单，纠正此前prepare失败阶段的判断，原生受力/H32仍未验证。本次0积分/0optimizer，无新训练/agent，MG05工具仍paused；最新入口覆盖下方停服等待记录。[收据](../data/manifests/q1_container_restore_CS02_20261027.json)。

2026-10-27：[MorphMimic只读初评](README_MORPHMIMIC_REVIEW_20261027.md)完成，固定main提交`aa40fe46a7fa995ba7a7354ad065a2c8ad64a302`。值得借鉴多形态映射/坐标与模型稳定性处理；默认位置反馈语义、29关节强映射及显存要求均不能直接套用Q1。作者大规模结果与G1真机不作Q1复现证据；无新安装/仿真/训练/agent，服务器仍停服，MG05未恢复。

2026-10-27：[MG05离线分析](README_SUPERMIMIC_Q1_MG05_SIGNAL_ANALYSIS_20261027.md)与候选准备：验证手势81.72%上肢平方误差为片内持续bias；[上游核查](README_SUPERMIMICHP_Q1_UPDATE_20261027.md)确认共有尾bin跨片bug。独立候选构建/关节误差与tracking/采样overlay及16项离线测试完成，[小收据](../data/manifests/supermimic_q1_MG05_candidate_offline_20261027.json)保存。实际runtime/Torch/原生接触/H32未验证，不能宣称效果改善。用户确认25245停服，等待新入口，停止连接/训练；MG05工具本次paused，目标未完成，服务器同步待。

2026-10-27：[MG05](README_SUPERMIMIC_Q1_MG05_DIAGNOSE_TRACKING_20261027.md)开始：上肢策略/奖励、真实足地接触/滑移、模型动作方向三类诊断；唯一子agent核师兄库更新，据证据选观测/tracking修正，随后一次4096环境最多一小时训练。此为最新授权，旧暂停与无目标条目仅历史；尚未启动本轮训练。

2026-10-27：[CS01切换25245容器](README_Q1_CONTAINER_SWITCH_20261027.md)完成连接与必要环境核验。`interactive63856`直接共享原ZHY挂载，RF09最终权重/BD01诊断/参考清单SHA匹配，330参考路径无缺失；CUDA/模型/权重加载通过，0积分/0optimizer，无需复制大数据。工作入口已更新，继续暂停训练；下一步优先上肢策略/奖励、足地真实接触与模型动作方向诊断。[小收据](../data/manifests/q1_container_switch_CS01_20261027.json)。

2026-10-27：[RF09暂停后的BD01瓶颈诊断](README_SUPERMIMIC_Q1_RF09_BOTTLENECK_20261027.md)完成。496次保存轨迹策略前向、0积分/0optimizer，重建回报差小于1.66e-5；验证手势策略目标误差1.184rad、关节相对目标0.062rad，首先是策略目标偏差。手势仍获69.1%最高步奖励、身体朝向项仅满值2.6%；Q1统一模型PD及参考足地关系仍需验证。用户明确先别继续训练，不启动RF10/续训；MG04已完成/get_goal为null，旧active为历史。[小收据](../data/manifests/supermimic_q1_RF09_BD01_diagnosis_20261027.json)。

2026-10-27研究：[五项简报](research/2026-10-27/README.md)已保存；Q1当前MotionLib静态存在末尾补帧复用末帧速度的模式，实际影响未测、未归因跌倒。另记SONIC朝向修复、第三方资产和三篇方法边界；无新增训练/仿真或原创假设。完整历史/去重由main维护，工程以Q1最新状态为准；[检查](README_RESEARCH_CHECK_20261027.md)/[同步回执](../data/manifests/research_watch_sync_20261027.json)。

2026-10-27：[MG04完成要求审计](../data/manifests/supermimic_q1_MG04_completion_audit_20261027.json)24项证据齐备，[RF09结果](README_SUPERMIMIC_Q1_RF09_RESULTS_20261027.md)提交913b2601d87934865b171064bbccfb7e16ad5b9b独立核远端、10文件307967字节三端大小SHA一致；08:00:27本轮q1-rf08-4096实际PAUSED并结构化回读确认，其他自动任务保留。扩库/重定向/五小时4096训练/统一基准及54秒MP4与六格图交付完成，仍明确普通走失稳与上肢短板；最后记录发布核验后结算目标，不自动续训。

2026-10-27 07:54：[RF09最终结果与本地交付](README_SUPERMIMIC_Q1_RF09_RESULTS_20261027.md)已完成：4096环境实训18000.931秒/17895学习轮，07:33全pipeline COMPLETED；890物理rollout/固定115基准/共同前缀与全程上肢支撑分析、36+31项训练/结果核验通过。旧验证根XY误差40.15→9.23cm、原地漂移大降，但提前终止0→2/15、手势上肢仍差，两条普通走演示仍倒地。初末54秒八段六类MP4和六格PNG/PDF/CSV等41文件39411623字节已Mac/服务器大小SHA一致，[交付](../data/manifests/supermimic_q1_RF09_local_delivery_20261027.json)/[量化](../data/manifests/supermimic_q1_RF09_results_20261027.json)。MG04内容完成，Git/监督收尾中，无自动续训或新训练。

2026-10-27 07:20：[RF09五小时训练正常完成](README_SUPERMIMIC_Q1_RF09_SOMA300_5H_20261027.md)，18000.931468秒/17895学习轮/183504896转移、WM178950/actor17395，worker退出0并reap，[36项保存审计V02](../data/manifests/supermimic_q1_RF09_training_audit_V02_20261027.json)通过。330片初末动力学已保存，验证提前终止30→1；统一RF08基准/共同前缀、长视频/六格图及本地交付正在执行，MG04 C完成、D/E未完成。V01序列化比较误报原件保留，无新增训练或重试。

2026-10-27 04:50：[RF09半程](README_SUPERMIMIC_Q1_RF09_SOMA300_5H_20261027.md)9018.543秒/9045学习轮、92893184转移，WM90450/actor8545；身份/三状态/资源正常，ckpt_009000.pt约68.73MB及SHA已登记，模型不拉Mac。[小收据](../data/manifests/supermimic_q1_RF09_checkpoint_009000_20261027.json)。继续原五小时预算，终后效果/六格图/长MP4仍待，不把文件保存当效果改善。

2026-10-27 03:13：[AP02展示参考步态核对](README_SUPERMIMIC_Q1_AP02_SOMA_RETARGET_20261027.md)完成八片CSV SHA/30Hz只读FK；两条普通前向走193/127帧均无踝中心侧序反转，最小横距14.88/14.29cm。舞蹈交叉单独标类，不把参考几何当策略成功或接触力，不改训练/既定视频清单，[收据](../data/manifests/supermimic_q1_AP02_demo_geometry_20261027.json)。RF09五小时训练仍进行中。

2026-10-27：[MG04新目标](README_SUPERMIMIC_Q1_MG04_EXPAND_RETARGET_20261027.md)active，[RF09五小时](README_SUPERMIMIC_Q1_RF09_SOMA300_5H_20261027.md)02:19:56已实际启动，4096环境/300训练30验证/根原点校正/从零；冻结a553b6a已发布并核远端、24文件三端大小SHA一致，153来源锁定。206学习轮/WM2060时仍正常预热，无新效果结论；沿用q1-rf08-4096已ACTIVE每10分钟监督，预计07:20左右训练结束后自动初末/固定RF08对照/六宫格/长MP4。不开子agent、不自动重试或续训。

2026-10-27：[AP02扩库/重定向](README_SUPERMIMIC_Q1_AP02_SOMA_RETARGET_20261027.md)330片全部转换，300训练/30演员分离验证；Q1静止pelvis原点重复偏移已修正，同源腕位置/峰高误差小幅改善。8段53.83秒展示清单固定，4096完整P02已通过并回收；[RF08补充55.24秒多类MP4](../data/manifests/supermimic_q1_RF08_multitype_local_delivery_20261027.json)3文件13795681字节已本地交付/大小SHA/解码目视通过，原片保留。RF09真实效果仍待。

2026-10-27：[MG03完成审计](../data/manifests/supermimic_q1_MG03_completion_audit_20261027.json)20项要求证据齐备，[RF08最终结果](README_SUPERMIMIC_Q1_RF08_RESULTS_20261027.md)提交bb8c909e44badf8346b6a8d08b5246aa75d48c67独立远端核验、19文件三端大小SHA通过，25个本地图/视频等小文件已交付。本轮q1-rf08-4096实际PAUSED，保留其他两任务；没有新增训练。存活/移动改善和全身跟踪仍不合格分别记录，目标工具最后结算随收尾提交核验后执行。

2026-10-27：[RF08最终结果与本地交付](README_SUPERMIMIC_Q1_RF08_RESULTS_20261027.md)完成：实训58分21秒、3510轮/3010actor，训练硬终止100→10、验证15→0；验证walking3/3完成，沿目标1.123m/参考2.394m，仍有根漂移和上肢/相位误差。初末115片、共同前缀/全程支撑与肩肘腕分析、六宫格PNG/PDF/CSV和初末MP4完成，[25文件本地大小SHA](../data/manifests/supermimic_q1_RF08_local_delivery_20261027.json)、20560条图表数值与全帧解码通过。MG03发布收尾，不自动延长；全部旧RUNNING为历史时点。

2026-10-27 00:28：[RF08训练正常结束](README_SUPERMIMIC_Q1_RF08_HP_SOMA_20261026.md)，实际3500.923046秒/3510学习轮36204544转移、WM35100/actor3010；worker退出0并reap，[26项训练保存审计](../data/manifests/supermimic_q1_RF08_training_audit_20261027.json)通过（146冻结来源、115输入、张量/计数/初始从零等）。初末115片评测与最终媒体进行中，图/本地MP4、共同前缀与效果结论待；MG03 A/B/C完成、D进行中/E待，没有新增训练。

2026-10-26 23:59：[RF07曲线判读](README_SUPERMIMIC_Q1_RF07_ROOT_FEEDBACK_20261026.md)补充原论文核验：配置正确不要求逐点单调，模型loss下降不等于真实控制改善，过拟合须看训练/验证差距；RF07只有129次actor更新、500轮平滑接近整段575轮，真实113/365步倒地仍为问题。[RF08](README_SUPERMIMIC_Q1_RF08_HP_SOMA_20261026.md)23:57已2091学习轮/1591actor、实训2036秒，现场身份/资源正常，继续冻结的一小时试验，终后效果待。

2026-10-26 23:27：[MG03观察目标](README_SUPERMIMIC_Q1_MG03_OBSERVATION_20261026.md)active，[RF08](README_SUPERMIMIC_Q1_RF08_HP_SOMA_20261026.md)23:23:07实际启动4096环境GPU训练，最多一小时；冻结d8b1dcf…已发布/独立核远端、40文件大小SHA同步三端，现场三status/进程身份/4096配置通过。23:26已271学习轮/3047424转移、WM2710/actor0，仍在500轮WM-only阶段，资源正常无效果结论。新q1-rf08-4096每10分钟监督已实际ACTIVE；终后初末115片、图与本地MP4，2/4/8小时不自动延长。[AP01](README_SUPERMIMIC_Q1_HP_ALIGNMENT_20261026.md)准备已完成，不重复转换/profile。

2026-10-26 23:39：[RF08终后共同前缀工具](README_SUPERMIMIC_Q1_RF08_HP_SOMA_20261026.md)已补齐并通过不等长终止/时轴/算术小测试，独立读取初末保存轨迹，0新积分/策略推理/优化。训练冻结源码未改；23:35现场已373次actor更新、资源正常。真实初末效果/交付仍待，不把工具准备和训练进度当完成。

2026-10-26 21:43：[SOMA接收与全面对齐准备](README_SUPERMIMIC_Q1_HP_ALIGNMENT_20261026.md)：用户提供的canonical骨架23062字节/SHA与师兄输入一致，已复制25252统一inbox并核对；元数据已在，SOMA主包由用户上传。精确复用100/15动作可补pilot_manifest.json/selection.json，生成来源记录可补generation_manifest.json；后两项不作额外数据门。整体Q1/GPU适配尚未实现或启动新训练，25677登录未成功，数据状态UNKNOWN。旧q1-2048更新返回不存在，未创建替代任务，独立仓库跟踪未改。

2026-10-26 20:45：[RF07完整交付](README_SUPERMIMIC_Q1_RF07_ROOT_FEEDBACK_20261026.md)：全部10阶段完成，3058.032232秒/575轮2355200转移、WM179组/actor129次，13项保存/初末身份算术审计通过。最终113/365步2.26秒倒地，净移0.50865m含失稳，不能称稳定行走；[RF06/RF07比较](../data/manifests/supermimic_q1_RF06_RF07_comparison_20261026.json)共同前缀根XY误差0.10987至0.03109m，上肢0.71598至0.68749rad，但完整抬手峰未到。11项PNG/PDF/CSV/初末MP4/代表帧1539139字节已在Mac逐大小SHA通过，权重留服务器。用户据此要求全面对齐师兄方法，不冒称多变量整体适配是单因素消融。

2026-10-26 19:50：[SuperMimicHP与Q1双向核查](README_SUPERMIMICHP_Q1_REVIEW_20261026.md)完成并采用原可微根反馈，6测试/真实CPU评测审计通过；冻结代码8ba7f820cffbe0e2f0ece7a3b34ce79d0683f7a6发布/远端核验/部署后，[RF07](README_SUPERMIMIC_Q1_RF07_ROOT_FEEDBACK_20261026.md)19:48:55启动，goal540→549、4096环境/原RF06控制/软50分钟最多1小时，现场身份匹配/step0/OOM0。q1-2048改为RF07每10分钟监督。师兄短片无H32起点时无限采样已隔离实际函数复现，非当前100条数据触发证据；每小时仓库跟踪继续双向核查，不保证Q1改善。

2026-10-26 19:00：[SuperMimicHP增量巡查](README_SUPERMIMICHP_WATCH_20261026.md)确认310244b只发布已有X2演示和文档，运行源码/配置/原24k结果未变，没有新的Q1相关技术变化，按偏好静默；状态文件推进已检查HEAD，每小时跟踪继续，不新增训练。

2026-10-26：[MG02最终要求审计与监督收尾](README_SUPERMIMIC_Q1_MG02_LEG_GOAL_20261026.md)完成，A至E的17项证据全部通过，最新一小时预算覆盖目标卡旧两小时文字。RF06真实训练52分13秒，未学会向前行走或完整抬手；定量比较、2×3图和本地初末MP4已交付。结果提交373088b2d75bd50827f0c857b005d9ba6b826631独立远端核验、16文件232478字节在Mac/25252大小SHA通过，见[最终收尾收据](../data/manifests/supermimic_q1_MG02_completion_audit_20261026.json)。q1-2048训练心跳已暂停，supermimichp-x2每小时仓库跟踪与yuanqi仍ACTIVE；未新增训练，下一候选为actor根反馈接口。

2026-10-26 18:35：[RF06全流程、完整对照和本地MP4交付](README_SUPERMIMIC_Q1_RF06_LEGS50_20261026.md)完成，18:21:52全部10阶段正常结束、controller/worker退出。实训52分13秒/600轮2457600转移，满足最多1小时；最终365步不倒但仅移1.49cm/参考1.60m、双支撑98.904%，抬手高度不足，腿部限矩提高未修复行走。[RF05/RF06共同365步比较](../data/manifests/supermimic_q1_RF05_RF06_comparison_20261026.json)0新积分/推理/更新，时长差异如实保留。11项图表/初末MP4共1706855字节本地大小SHA通过，六宫格129条CSV独立核对通过；[MG02](README_SUPERMIMIC_Q1_MG02_LEG_GOAL_20261026.md)A至E实测证据齐备，最终发布/三端核验收尾中。下一候选为策略根反馈接口，未新开训练；独立SuperMimicHP每小时跟踪继续。当前状态已压缩，原时点记录保存在[历史快照](README_CURRENT_STATE_ARCHIVE_20261026.md)。

2026-10-26 18:15：[RF06腿部50%限矩训练完成及初末评测](README_SUPERMIMIC_Q1_RF06_LEGS50_20261026.md)：实训3132.86秒/600轮2457600转移、WM189/actor139，保存13项/初末算术身份审计通过。最终365步不倒但只净移1.49cm、双支撑98.904%，未有效换脚；回报0.47534→0.43743，上肢误差略恶化，限矩扩大尚未修复行走。worker回收/GPU14MiB/OOM0，MG02 A至D完成；现有渲染和本地交付/跨轮比较待，保持目标active。

2026-10-26 17:56：[SuperMimicHP师兄X2仓库跟踪](README_SUPERMIMICHP_WATCH_20261026.md)已启用独立每小时心跳`supermimichp-x2`，初次Git已核默认分支`feat/x2-embodiment`及`649d1b1165bc08fdcf4468e596161db73d93d7b9`。源码显示X2 actor显式接收根XY误差，当前Q1该误差仅供critic；X2正式入口1024环境、100条SOMA训练，且采用逐关节PD与完整effort限矩。列为后续受控对照候选，非Q1已证实根因；RF06继续冻结、一小时上限不变，0新增训练。

2026-10-26 17:22：[唯一RF06腿部限矩对照实际启动](README_SUPERMIMIC_Q1_RF06_LEGS50_20261026.md)，RF05已完整结束并本地交付后顺序执行。4096环境、12腿关节限矩25%→50%、其他控制保持，从零，软50分钟/最多1小时；controller3301/worker3304真实身份、控制manifest及预算已核，初始RUNNING/step0/OOM0，尚无效果结论。MG02 A/B/C完成，D训练中，禁止重复启动或热改。

2026-10-26 17:10：[RF05全流程和本地交付完成](README_SUPERMIMIC_Q1_E4096_20261025.md)，9阶段完成/训练7237.63秒、GPU释放，11项图表及初末MP4共1.72MB大小SHA一致。双支撑99.178%、单支撑0、沿目标方向-1.68cm，仍未行走；腿部限矩边界饱和仅0.274%，不能套用RF04的髋饱和解释。[独立RF04/RF05比较](../data/manifests/supermimic_q1_RF04_RF05_comparison_20261026.json)共同122步误差已核，MG02 A/B/C完成，下一唯一一小时RF06。

2026-10-26 17:05：[RF05-E4096训练与动力学结果已出](README_SUPERMIMIC_Q1_E4096_20261025.md)，实际7237.63秒/4608000转移、WM399/actor349，保存及初末算术审计通过；最终365步不倒但仅净移1.87cm，目标1.60m，上肢/腿部完整段误差均略恶化。初始策略已能存活，尚未学会行走/抬手。管线仍在渲染，全部本地交付和RF06启动尚待。

2026-10-26 16:33：[MG02-C发布部署核验完成](README_SUPERMIMIC_Q1_RF06_LEGS50_20261026.md)，Q1代码`0b1da033148fff0b705fd85cdac5a773b5a7f332`已独立核远端、13项代码/报告/小收据的Mac及25252大小SHA一致，目标AGENTS已补一小时预算。[部署收据](../data/manifests/supermimic_q1_RF06_C_deployment_20261026.json)保留当时文件身份；后续纯文档更新不改代码。RF05仍运行，RF06尚未启动。

2026-10-26：[MG02-C独立RF06实现与验证](README_SUPERMIMIC_Q1_RF06_LEGS50_20261026.md)完成：8测试通过，C02真实CPU评测/审计/全身/腿部/两帧MP4编码退出0，状态/动作与D06B差0；首轮媒体入口故障保留，复用已有字体/ffmpeg修复，RF05的78项冻结来源未改。RF06只提高12腿关节限矩，最多1小时，尚未启动。RF05验证环境/更新批量规模，不能称腿部修复。

2026-10-26：[MG02预算修订](README_SUPERMIMIC_Q1_MG02_LEG_GOAL_20261026.md)：用户要求以后每轮训练最多1小时，RF06改为最多3600秒、软目标3000秒并加训练阶段退出保护。当前已运行RF05保留原两小时基线；后续不自动延长，比较如实说明预算差别。

2026-10-26：[MG02-B完整闭环腿部诊断](README_SUPERMIMIC_Q1_LEG_CONTROL_20261026.md)：同RF04固定策略，原25%122步倒地，腿部50%与全身50%均365步不倒；腿部50%仅移动4.04cm、双支撑99.18%，不称走起来。原基线状态/动作误差0，852控制步/0optimizer/无CUDA；下一独立RF06只改腿部限矩，当前RF05仍原控制。

2026-10-26：[MG02腿部闭环与独立训练目标](README_SUPERMIMIC_Q1_MG02_LEG_GOAL_20261026.md)已创建active。当前RF05未做腿部特调，继续作为4096原控制基线；并行准备固定RF04策略的三组闭环CPU诊断，RF05结束后仅将12个腿关节限矩25%提高到50%的独立RF06。PD、动作尺度、上肢/腰、奖励及参考保持，下一轮仍4096环境至少两小时；尚无腿部修正训练结果。

2026-10-26 15:02：[CM01全量迁移完成](README_Q1_CONTAINER_MIGRATION_20261026.md)，283995文件/327.64GB全部内容SHA、原树结构一致，外部依赖和链接核验、CUDA/三个Q1模型/EGL实际通过。用户自己的25252容器已启动[RF05-E4096](README_SUPERMIMIC_Q1_E4096_20261025.md)，4096环境、原参考/控制/seed/从零、至少7200秒真实训练；新OOM基线0，运行初始状态已核，无效果结论。新源码及27项小文件部署来自已发布`abda76e7bc873405b4321b4d9a6f5991e7290828`，不恢复旧失败轮。

2026-10-26：[RF04-D05腿部动力学对照](README_SUPERMIMIC_Q1_LEG_CONTROL_20261026.md)完成。原保存动作精确重现；25%限矩时髋横滚频繁饱和，50%同动作122步内未倒但仍未行走，单独降低参考高度也未修复直接PD早倒。用户已明确允许迁移同期分析及运行就绪后训练，后续正式RF05只在自己的25252容器执行；下方“仅迁移、不恢复训练”为较早范围。

2026-10-26：[CM01迁入用户25252容器](README_Q1_CONTAINER_MIGRATION_20261026.md)进行中。用户澄清25651是师兄容器，授权全部项目文件直传到自己的`/opt/data/private/ZHY`；源原件保留，补必要外部依赖，再核SHA/链接/运行环境。MG01仍paused，无新训练。下方25651为当时的历史入口。

2026-10-26 13:36：[新入口25651及RF04同参考失稳诊断](README_SUPERMIMIC_Q1_FAILURE_REVIEW_20261026.md)完成。原ZHY、环境、BONES与RF04权重均可见，3090空闲、余盘约1.87TB；MG01仍paused，RF05未启动。失败同片Q1参考整段抬高36.44mm，90.14%帧双脚足底均高于15mm，前122帧52.46%均高于30mm；0积分/0更新。优先验证重定向足地关系与控制/学习衔接，几何证据不冒充唯一倒地因果；不恢复额外数据门。以下active/盘满等为此前历史快照。

2026-10-26 02:02：[共享盘可用空间已为0](README_Q1_STORAGE_GROWTH_20261026.md)，RF05仍未创建。元数据与profile/排查记录已发布Q1并独立核验提交`81db2e1fee2828e9ce59d0271c7d2f2cde56cee1`，服务器同步和代码部署暂缓，待空间恢复后补齐；本机RF04曲线和MP4已交付，不受影响。没有删除文件、停他人任务或降低100GiB保护。

2026-10-26 01:52：[共享盘快速增长排查](README_Q1_STORAGE_GROWTH_20261026.md)完成：两次private各一级目录和703216345088字节总量完全相同，底层可用空间却持续减少；private只是宿主机子树，增长超出当前可见范围，具体目录/进程需平台侧定位。01:52余约30GiB，未删除文件或停止其他任务。[4096完整批量实测](README_SUPERMIMIC_Q1_E4096_20261025.md)已于01:35通过：16轮65536转移、WM2048×10及actor4096/H32/critic有限，CUDA分配/保留峰值15.60/16.54GiB，worker正常回收/OOM14。MG01 A/B/C完成，RF05尚未创建，待100GiB余盘恢复；正式目标保持active。RF04本地六宫格和初末MP4链接见[交付记录](README_SUPERMIMIC_Q1_E2048_20261025.md)。

2026-10-26 01:23：[RF04-E2048整轮完成与本地交付](README_SUPERMIMIC_Q1_E2048_20261025.md)：真实训练7244.11秒/1775轮/3635200转移、WM659/actor609，保存13项和初末算术身份审计通过；六宫格/初末MP4的11项1512643字节已在Mac逐大小/SHA核对。最终净位移0.45659m，但122/365步、2.44秒提前倒地；整片回报0.47538→0.19674，共同上肢误差略改善但抬手主峰未到达。GPU回落14MiB/OOM14；共享盘余183.36GiB持续下降，本实验约0.483GB，已询问外部空间。MG01 A/B完成，下一4096完整实测和独立对照仍待，目标active。

2026-10-26 00:31：[RF04首个1000轮检查点已登记](README_SUPERMIMIC_Q1_E2048_20261025.md)，保存205591335字节/完整SHA；实际已1050轮、2150400转移、WM369/actor319、训练4085.71秒，身份匹配、OOM14/盘余507.86GiB，本实验约274.96MB。仅文件身份和现场观测，未重复张量审计；两小时和终后动力学评测未完成，4096未启动，MG01继续active。

2026-10-25 23:59：[MG01正式监督目标已创建active](README_SUPERMIMIC_Q1_MONITOR_GOAL_20261025.md)，明确2048两小时/评测交付→4096完整批量实测→独立4096两小时/评测交付→移动/上肢定量比较与Q1归档，证据齐备才完成。沿用原监督任务，正常静默；现场RF04625轮/128万转移、WM199/actor149、资源保护正常，4096未启动。未新增数据门、训练或子agent。

2026-10-25：[用户追加后续4096环境](README_SUPERMIMIC_Q1_E4096_20261025.md)，对齐原论文Table A.5的4096/WM2048×10/actor4096-H32/UTD0.4。当前RF04的2048两小时继续，完成评测后再顺序实测4096并独立RF05，暂沿用同片/从零/两小时；没有启动第二训练。18GB仅估计，尚无4096容量或效果结果；继续原资源保护和唯一监督任务。

2026-10-25 23:35：[RF04-E2048-UP01实际启动](README_SUPERMIMIC_Q1_E2048_20261025.md)，23:21:31起从零2048环境/8槽/8线程、WM1024×10、actor2048/H32、UTD0.4，真实训练至少7200秒。300轮/614400转移、WM69组/actor19次，critic和梯度有限、78项源码入口身份保持；profile完整更新及两端8控制通过，冻结f03db75，温度/内存/OOM/盘余保护正常，`q1-2048`每10分钟静默监督。唯一子agent完成[旧RF02位移诊断](README_SUPERMIMIC_Q1_DISPLACEMENT_REVIEW_20261025.md)：目标移动速度未丢失，实际速度仅参考约5.7%、双支撑97.86%，支持站立局部最优，未确定单一原因；没有新增GPU或仿真。训练终后初末动力学比较/上肢位移/六宫格/本地MP4待，不能提前称扩容有效。

2026-10-25 22:51：[2048环境至少两小时新要求](README_SUPERMIMIC_Q1_E2048_20261025.md)已登记；旧RF03按用户要求STOPPED_SIGNAL于667轮/42688转移、WM216/actor166，worker已回收。新RF04计划2048环境/WM1024/actor2048、按论文比例采样，正式训练前先核资源；唯一子agent只读分析位移，主代理监督训练，尚未启动新轮。

2026-10-25 22:35：[RF03-UP01双臂单动作实验已启动](README_SUPERMIMIC_Q1_RF03_UP01_20261025.md)：既有365帧双臂抬起加移动参考，从零64环境/12000轮，仅改参考范围，实际配置/共同源码与RF02相同。两端3控制通过；375轮/24000转移、WM99组/actor49次，GPU40/OOM14/盘余1.111TB。流程在后台独立运行，终后自动接本轮初始/最终MuJoCo、上肢/移动分解、六宫格和本地MP4，尚无本轮效果结论。

2026-10-25：[RF02全身跟踪诊断D02完成](README_SUPERMIMIC_Q1_RF02_FULLBODY_20261025.md)：4条专项上肢片平均误差改善但中途抬手幅度仅目标8.45–13.16%；1024来源全是移动候选，源G1同样少抬手。原8片双脚承重97.86%、腿饱和0.51%；6状态H1/8/32局部奖励方向各12/12一致，剥离动作率仍成立，但根XY微调仅约0.07mm，不能证明行走方向有效。10483控制步/4聚焦控制/原数组审计通过，0optimizer更新、工件约6.73MB/OOM14，作业全部退出。下一具名双臂单动作学习对照建议已登记，未启动新训练。

2026-10-25：[RF02奖励、重定向与接触梯度归因](README_SUPERMIMIC_Q1_RF02_GRADIENT_REASONING_20261025.md)完成原论文/当前源码/历史诊断只读核对。全身与全局根奖励确实存在；低状态误差没有验证动作收益方向。旧R03局部开环检查不能套用本轮BONES；下一优先原轨迹奖励分项、同参考接触对应和本轮模型/真实动作微调收益比较，不直接改权重、接触或追加训练。本次0新仿真/网络前向/更新。

2026-10-25 21:03：[RF02首版基线交付](README_SUPERMIMIC_Q1_BONES_BASELINE_20261025.md)完成：1024段/108.62分钟、12000轮/768000转移，主训练6小时31分35秒；最终保存核对及16阶段退出0。固定8片初始/最终回报0.379982至0.654285、提前终止2/8至0/8，但最终净位移均值仅0.02035m/参考1.81529m，尚未学会行走。两段真实动力学MP4已保存Mac并核大小/SHA，按用户要求关闭本轮临时网页服务及18863转发，18864训练桥保留。[最终2×3训练图](README_SUPERMIMIC_Q1_TRAINING_DASHBOARD_20261025.md)及1425点独立数值核对完成，前三回合指标明确留缺。终后盘余1.682TB/OOM14；代码、结论与小收据同步Q1，权重仅留服务器。以下RUNNING/待交付条目均为各时点历史记录。

2026-10-25 20:03：[RF02第11000轮保存身份登记](README_SUPERMIMIC_Q1_BONES_BASELINE_20261025.md)，704000真实转移、WM4349组/actor4299次，完成本次预算91.67%；205.92MB保存及完整SHA已读，未重复张量审计。[保存后精简观测](../data/manifests/supermimic_q1_BONES_RF02_checkpoint_11000_observation_20261025.json)已继续11050轮/707200转移、四进程身份匹配、RUNNING/OOM14，训练分配2.336GB/盘余1.916TB。前台会话78493继续观察整个pipeline，训练与绘图回调独立运行；完整12000轮及终后交付仍待。

2026-10-25 19:33：[RF02第10000轮保存身份登记](README_SUPERMIMIC_Q1_BONES_BASELINE_20261025.md)，640000真实转移、WM3949组/actor3899次，完成本次预算83.33%；205.92MB保存及完整SHA已读，未重复张量审计。[保存后观测](../data/manifests/supermimic_q1_BONES_RF02_checkpoint_10000_observation_20261025.json)已继续10100轮/646400转移、四进程身份匹配、RUNNING/OOM14，实测RTX3090为41摄氏度，训练分配2.130GB/盘余2.065TB。目标交付进度表已关联数据、训练、固定8片比较、视频、曲线和资源证据；完整12000轮及终后交付仍待，训练独立继续。

2026-10-25 18:58：[RF02第9000轮保存身份登记](README_SUPERMIMIC_Q1_BONES_BASELINE_20261025.md)，576000真实转移、WM3549组/actor3499次，完成本次预算75%；205.92MB保存及完整SHA已读，未重复张量审计。[保存后观测](../data/manifests/supermimic_q1_BONES_RF02_checkpoint_9000_observation_20261025.json)已继续9050轮/579200转移，四进程身份及68源码SHA保持、RUNNING/OOM14，训练分配1.924GB/盘余2.066TB。观察正常退出，训练及绘图回调独立继续；完整12000轮/固定8片物理比较、最终图与回放仍待。

2026-10-25 18:23：[RF02第8000轮保存身份登记](README_SUPERMIMIC_Q1_BONES_BASELINE_20261025.md)，512000真实转移、WM3149组/actor3099次，完成本次预算三分之二；205.92MB保存及完整SHA已读，未重复张量审计。[原样观测](../data/manifests/supermimic_q1_BONES_RF02_checkpoint_8000_observation_20261025.json)已继续8025轮/513600转移、四进程身份匹配、RUNNING/OOM14，训练分配1.718GB/盘余2.067TB。观察正常退出，训练与绘图回调独立继续；完整12000轮/固定8片物理比较及回放仍待。

2026-10-25 17:52：[RF02第7000轮保存身份登记](README_SUPERMIMIC_Q1_BONES_BASELINE_20261025.md)，448000真实转移、WM2749组/actor2699次，205.92MB保存及完整SHA已读，未重复张量审计。[原样观测](../data/manifests/supermimic_q1_BONES_RF02_checkpoint_7000_observation_20261025.json)已继续7075轮/452800转移，四进程身份及68源码SHA保持、RUNNING/OOM14；训练分配1.512GB/盘余2.068TB，final.pt仍不存在。绘图等待、完整12000/固定8片物理比较及回放未完成。

2026-10-25 17:18：[RF02第6000轮保存身份登记](README_SUPERMIMIC_Q1_BONES_BASELINE_20261025.md)，384000真实转移、WM2349组/actor2299次，本次采集预算一半；205.92MB保存及完整SHA已读，未重复张量审计。[原样观测](../data/manifests/supermimic_q1_BONES_RF02_checkpoint_6000_observation_20261025.json)已继续6050轮/387200转移，四进程身份保持、RUNNING/OOM14；RTX3090实际39摄氏度、训练分配1.306GB/盘余2.076TB，final.pt尚不存在。绘图继续等待，完整12000/固定8片物理比较及回放未完成。

2026-10-25 17:03：[RF02固定八片具体名单已保存](README_SUPERMIMIC_Q1_BONES_BASELINE_20261025.md)，沿原计划前8train排序规则，2409帧/48.02秒，实际NPZ身份与合库一致；瘦身收据8057字节/完整SHA绑定来源。0新动力积分/更新，训练代码和选择规则不变，生成时尚在5600轮；结束后核对两份实际评测及第一片视频，不以名单准备宣称效果改善。

2026-10-25 16:45：[RF02第5000轮保存身份登记](README_SUPERMIMIC_Q1_BONES_BASELINE_20261025.md)，320000真实转移、WM1949组/actor1899次，205.92MB保存及完整SHA已读，未重复张量审计。[原样观测](../data/manifests/supermimic_q1_BONES_RF02_checkpoint_5000_observation_20261025.json)已继续5075轮/324800转移、四进程身份匹配、RUNNING/OOM14，训练分配1.100GB/盘余2.077TB。训练和绘图回调独立继续，完整12000轮/固定8片物理比较及回放仍待。

2026-10-25 16:27：[每轮六宫格训练图标准已落地](README_SUPERMIMIC_Q1_TRAINING_DASHBOARD_20261025.md)，细原始/粗500训练iteration平均、PNG/PDF/CSV/收据、叠加预算如实标注。服务器5控制过（含真实MuJoCo状态/奖励/RNG不变），本轮4325预览已目视通过；缺失三项回合统计明确留缺，下一轮入口补记录。本轮唯一终后回调已实际等待；RF02已4525/12000轮、289600转移，四进程身份/68冻结源码保持、OOM14/盘余2.077TB，完整训练/评测/回放仍待。

2026-10-25 15:38：[RF02第3000轮保存登记](README_SUPERMIMIC_Q1_BONES_BASELINE_20261025.md)完成，192000真实转移、WM1149组/actor1099次，保存205.92MB及完整SHA已读；三进程身份匹配，未重复张量审计。原样状态已继续3050轮/195200转移，RUNNING/OOM14，训练分配687.84MB；12000完整预算及固定8片动力学比较/回放仍待。

2026-10-25 15:21：[RF02回放入口源码准备](README_SUPERMIMIC_Q1_BONES_BASELINE_20261025.md)完成，沿用既有Q1视频页、绑定原评测字段，Node语法/服务器逐SHA通过；尚未部署真实RF02视频或完成浏览器画面/控件核验。训练已继续2525轮/161600交互，RUNNING/OOM14；15:20三进程身份匹配、盘余2.087TB，完整训练和评测仍待。

2026-10-25 15:04：[RF02第2000轮自动保存已登记](README_SUPERMIMIC_Q1_BONES_BASELINE_20261025.md)，128000转移/WM749组/actor699次、三进程身份匹配；保存205.92MB及完整SHA已读，没有重复张量审计。取回原样状态已继续2025轮/129600转移、RUNNING/OOM14，训练分配481.86MB/盘余2.089TB；12000完整训练和既有动力学比较/回放待。

2026-10-25：[RF02核心链路只读审阅完成](README_SUPERMIMIC_Q1_RF02_METHOD_REVIEW_20261025.md)，未发现要求中止的8步/H32/critic或replay错误；确认500步截断对采样回报的偏差，86/1024段存在长度触发条件，影响尚未量化，保持本轮冻结代码继续。原样进度1625轮/104000交互、WM599组/actor549次，RUNNING/OOM14；唯一子agent已结束，完整训练/动力学比较待。

2026-10-25 14:30：[RF02第1000轮保存真实核对通过](README_SUPERMIMIC_Q1_BONES_BASELINE_20261025.md)，64000转移/WM349组/actor299次，独立CPU13项、三Adam3490/299/5980、三网络变化及有限/1024输入与源码身份保持；保存205.92MB、原样收据逐SHA取回。14:31已继续1025轮，训练分配275.88MB/盘余2.091TB/OOM14；完整12000及动力学比较/回放待。

2026-10-25 14:17：[RF02中间保存核对入口准备完成](README_SUPERMIMIC_Q1_BONES_BASELINE_20261025.md)，独立CPU只读脚本已用RF01真实500轮控制验证13项，三Adam/三网络变化与原独立核对相同，3.22秒/0新积分更新；RF02继续训练，1000保存实际核对待。

2026-10-25：[BONES PF02完整合库及RF02真实更新](README_SUPERMIMIC_Q1_BONES_BASELINE_20261025.md)：1024段/326883帧/108.62分钟、0转换/IO错误、冻结来源缺项/额外0；新7批1.031GB、旧128复用。14:02:53进入64环境12000轮训练，已存350轮/22400转移快照、WM89组/actor39次、H32 BPTT及critic真实更新、初始模型实存；16份静态小收据逐SHA一致，仍待完整训练与固定8片动力学比较、回放，不称效果改善。

2026-10-25 13:38：[BONES G02复现基线流程已启动](README_SUPERMIMIC_Q1_BONES_BASELINE_20261025.md)，代码d66ac1c已核远端Q1，计划1024段/12000轮及固定参考比较；最多一个子agent。本条为启动快照，实时恢复见[当前状态](README_CURRENT_STATE.md)。

2026-10-25：[RF01 E01两片真实动力学比较](README_SUPERMIMIC_Q1_BONES_BASELINE_20261025.md)完成，两次CPU MuJoCo及算术复算通过，整片完成比例62.03%→32.49%、最终两片均提前终止；首片视频实际倒地71/266步。尚无整体改善，五份原样小收据逐SHA取回；继续原定扩库/训练，不恢复自定几何门。

2026-10-25 12:53：[RF01取消额外数据门后训练完成](README_SUPERMIMIC_Q1_PAPER_FIRST_20261025.md)，128片BONES/13.19分钟，64环境500轮/32000真实转移，约11.49分钟；WM149组/actor99次，三网络真实变化、三Adam1490/99/1980与有限性核对通过。guard完成且worker回收、盘余2.094TB/OOM14；尚无本轮固定动力学效果结论。当前用户决定优先于下方历史条目的旧19/22门、规模配额及诊断下一步，恢复见[当前状态](README_CURRENT_STATE.md)。

2026-10-25 12:37：[PF01取消额外门槛后128片全部组库](README_SUPERMIMIC_Q1_PAPER_FIRST_20261025.md)，39694帧/791.32秒=13.19分钟、0转换/IO错误；统一2rad/s裁速已取消、原几何flags仅诊断。分配142.74MB、盘余2.094TB/OOM14；下一直接64环境500轮世界模型/BPTT有限训练。

2026-10-25：[用户纠正：取消额外数据硬门槛，优先SuperMimic复现](README_SUPERMIMIC_Q1_PAPER_FIRST_20261025.md)。旧19/22门只作历史诊断；新入口取消统一2rad/s裁速，先128冻结train来源原速Q1准备与64环境500轮有限训练，真实执行待。后续不以代理自定支撑/间隙/自碰撞或规模配额阻止训练。

2026-10-25：[BONES D01足底比较准备](README_SUPERMIMIC_Q1_BONES_SUPPORT_20261025.md)，同8个冻结train来源比较原G1、Q1映射脚刚体与导出前后实际足底，逐帧几何闭合及原指标复核；真实执行待，0新参考/训练。

2026-10-25 12:03：[BONES首8个train真实原速小批完成](README_SUPERMIMIC_Q1_BONES_RECEIPT_20261025.md)，3098帧/61.80秒，源FK/映射/时间/导出通过；独立22门0/8准入，支撑与最大间隙全部失败、自穿透7/8失败。下一同批来源/映射/Q1足底诊断，不扩大不合格库；A02共446.51MB、盘余2.094TB/OOM14，0新训练。

2026-10-25 11:58：[BONES整包完整接收通过](README_SUPERMIMIC_Q1_BONES_RECEIPT_20261025.md)，完整SHA/EOF-CRC与142220 CSV目录覆盖保持，提取冻结1280候选402.82MB；guard完成回收、整个实验分配435.14MB、盘余2.094TB/OOM14。下一首8个train来源原速转换和独立22门，尚无BONES合格参考或训练。

2026-10-25 11:48：[BONES原包完整本地SHA/实际CSV表头预览通过](README_SUPERMIMIC_Q1_BONES_RECEIPT_20261025.md)，23.50GB同盘归档无副本、SHA52580ea8…，真实36列与V2一致。下一全包gzip CRC与1280冻结候选流式接入，再做8个train来源原速小批；尚无新BONES合格参考或训练。

2026-10-25：[BONES真实原包已发现](README_SUPERMIMIC_Q1_BONES_RECEIPT_20261025.md)，用户已拷贝`ZHY/g1.tar.gz`，实测23.50GB/两次稳定/无可见写FD。准备同盘归档、完整SHA/CRC及冻结候选原速小批；尚无新BONES参考或训练。[P15R八组](README_SUPERMIMIC_Q1_RELOCATED_DIAGNOSTICS_20261025.md)及独立756项核验完成，两8rad/s几何子集通过但滑脚/原速度门不通过，0正式参考。

2026-10-25 11:31：[DEV-B最新策略实际开发配对完成](README_SUPERMIMIC_Q1_RELOCATED_DIAGNOSTICS_20261025.md)，两次MuJoCo/两次独立重算均通过。父5000与累计10282均6/6完整、0/6跟随；移动0.07825至0.07258m/参考0.52117m，额外训练未改善移动。新实验约4.09MB，原冻结/终态保持；下一P15R足地恢复八组。

2026-10-25：[统一目录后开发评测与足地修正](README_SUPERMIMIC_Q1_RELOCATED_DIAGNOSTICS_20261025.md)实现与预检中。显式M01路径映射/内容核验，拟独立比较5000父与累计10282退出策略，再执行P15八组；旧终态保持，尚无新动力学结果。

2026-10-25 10:52：[旧资产清理与统一目录完成](README_Q1_STORAGE_CONSOLIDATION_20261025.md)。实际删除178938项/文件块180.70GB，剩余约299.58GB全部同盘归拢至`/opt/data/private/ZHY`四目录，旧根仅兼容链接。280057常规文件、3703链接、41364目录与516冻结/5关键文件核验通过，依赖导入及三份Q1模型加载/forward通过；无新训练，原停止/失败保持。共享盘余约2.142TB，其中清理前外部已恢复约1.96TB。

2026-10-25 09:42：[private与项目存储盘点完成](README_Q1_PRIVATE_STORAGE_INVENTORY_20261025.md)。五个一级目录合计831.41GB；Q1/yuanqi_ops/旧ZHY相关目录合计480.36GB，补齐此前仅Q1约20GB口径。两组旧A3权重SHA确认双份、环境亦有重复；尚未清理。private两次总量一致、同期整盘已用增27.73GB，当前可用0，需平台侧查可见目录之外占用；具体写入者未定位。

2026-10-25 09:02：[P17真实父库只读预览通过](README_SUPERMIMIC_Q1_DIRECTORY_EXCLUSION_VIEW_20261025.md)，两份各2885行变换保持十九门/旧来源链/held-out，P10十五预留目录排除；候选593 train/35.27分钟，正式库未建。共享盘仍0，Q1 data自身约19.95GB不足恢复100GiB；后续科学执行等待外部空间。

2026-10-25：[P17目录排除视图准备](README_SUPERMIMIC_Q1_DIRECTORY_EXCLUSION_VIEW_20261025.md)实现及本地五项控制通过，保持旧HumanML来源链/十九门/held-out，重核P10预留来源，盘余预检在建目录前。真实父库预览待，正式新库未建/未训练；08:47盘余仍0。

2026-10-25 08:40：[P16全625片命名目录复核](README_SUPERMIMIC_Q1_SOURCE_DIRECTORY_AUDIT_20261025.md)发现GRAB/s1及s10共19片train目录候选与held-out重叠，其中原R02 train11片；保守排除后候选593片/35.27分钟，尚未建库/准入。0服务器新文件，旧输入/终态保持；盘余0继续等待，P15八组及动力学评测未执行。

2026-10-25 08:32：[共享盘0可用及停止身份复核](README_SUPERMIMIC_Q1_STORAGE_FULL_20261025.md)，08:26/08:32两次确认、原三终态保持、P15目录未创建；正式计算等待100GiB盘余，继续已有小型元数据核对。0服务器新输出，不重复资源询问。

2026-10-25 08:22：[最新共享盘资源](../logs/session_records/supermimic_q1_P15_final_resource_20261025.log)余17.74GB，原训练已停止且进程消失、P15正式目录未创建；100GiB保护不变。P15五控制及退出权重十五项核验已过，正式转换/评测等存储恢复，用户已收到资源询问和下降通知。

2026-10-25 08:19：[P15-C五项服务器控制全部通过](README_SUPERMIMIC_Q1_FLOOR_RECOVERY_20261025.md)，修正实际地面缓存/验证mesh+sphere导数及恢复QP；A/B失败保留，正式八组因余盘不足100GiB未启动。[R02D退出保存十五项只读核验通过](README_SUPERMIMIC_Q1_R02D_BOUNDED_20261025.md)，计数/516冻结/停止身份保持，0服务器新文件；盘余35.71GB，原STOPPED_STORAGE不恢复，目标未达。

2026-10-25 08:05：[R02D存储保护停止](README_SUPERMIMIC_Q1_R02D_BOUNDED_20261025.md)，5282/7000新轮/338048转移，guard STOPPED_STORAGE、controller/train STOPPED_SIGNAL，退出权重保存、最终配对未执行，不自动恢复。共享盘随后约73GiB，不删未知文件。[P15首轮控制失败及地面缓存定位](README_SUPERMIMIC_Q1_FLOOR_RECOVERY_20261025.md)：配置floor移动但sameframe仍使实际Z=0，且Q1脚含mesh；正式八组0启动，下一修正新入口控制。

2026-10-25：[P15恢复型足地约束](README_SUPERMIMIC_Q1_FLOOR_RECOVERY_20261025.md)适配器/四控制已实现，本地两解析控制通过，服务器控制与八组待；保持P14固定输入/原限制及R02D预算。08:02原训练5200新轮/RUNNING，盘余123.97GB，100GiB保护不变。

2026-10-25 07:56：[R02D保存5000独立十三门通过](README_SUPERMIMIC_Q1_R02D_BOUNDED_20261025.md)，三Adam计数38980/3798/75960、原340输入/516冻结保持，实际5050新增轮/RUNNING，数值有限不代表步行有效；盘余152.34GB，本训练2.259GB，100GiB停止线不变，若触发不自动恢复。

2026-10-25 07:53：[P14真实八组及独立复算完成](README_SUPERMIMIC_Q1_WALKING_ROOT_OFFSET_20261025.md)，降根52至107mm后全部穿地20至63mm/滑动，0准入；16状态只读核实原生足地不等式允许已有穿地零步，下一先验证恢复约束/可行初始化，不能把必要腿长条件当可行证明。原R02D4975新轮/三进程身份保持/OOM14，盘余168.26GB、100GiB保护不变。

2026-10-25：[P14整段常数根高度适配](README_SUPERMIMIC_Q1_WALKING_ROOT_OFFSET_20261025.md)实现/本地三控制/服务器四控制完成，代码逐SHA部署；固定四train/2和8rad/s，绑定P12C/P13，以整段共同Z区间选单个delta，保留足世界锚点与原时轴。八组IK及独立核验待，0新准入；R02D原预算保持，BONES交师兄。

2026-10-25 07:31：[P13独立6068核验/4314 fresh FK完成](README_SUPERMIMIC_Q1_WALKING_REACH_20261025.md)，根与修正踝目标外球冲突确认，下一预定整段常数高度适配P14，尚未实施/0新参考。原[R02D](README_SUPERMIMIC_Q1_R02D_BOUNDED_20261025.md)4375新增轮/280000转移、三原进程身份保持/OOM14；盘余283.46GB，本训练1.849GB，100GiB停止线不变，尚无最终跟随效果。

2026-10-25 07:25：[P13真实24组/4314帧完成](README_SUPERMIMIC_Q1_WALKING_REACH_20261025.md)，原目标未超保守腿长球、全部修正变体每帧至少一踝超球，最大101.98mm；必要降低方向未被几何排除，尚不等于可达/动力学成功。独立复算待，0新参考；原R02D预算不改。

2026-10-25：[P13首次预检失败及包装器修正](README_SUPERMIMIC_Q1_WALKING_REACH_20261025.md)，错误在建目录前完整存储检查，0worker/正式数组读取；原四控制通过不代表端到端通过。对齐既有guard顺序、停止线不变，正式结果待。

2026-10-25 07:20：[R02D保存4000十三门通过](README_SUPERMIMIC_Q1_R02D_BOUNDED_20261025.md)，三Adam计数/516冻结/原340输入保持，执行前4075新轮，盘余340.03GB/OOM14。另[P13四个服务器控制](README_SUPERMIMIC_Q1_WALKING_REACH_20261025.md)全过/0跳过，正式24组只读可达性分析待；没有新动力学跟随结论。

2026-10-25：[P13步行可达性诊断准备](README_SUPERMIMIC_Q1_WALKING_REACH_20261025.md)，保留P12C24组，比较原/修正踝目标与真实Q1腿链外球及必要根Z区间；只读/0新参考，实际控制和正式结果待。07:10原R02D3825新增轮继续，BONES仍交师兄。

2026-10-25 07:06：[P12C真实24组与独立核验完成](README_SUPERMIMIC_Q1_WALKING_CONTACT_20261025.md)，无穿地但悬空8至34mm、近地0至15.08%，0准入；三轮共72组，后续量化可达性/根足兼容。原[R02D](README_SUPERMIMIC_Q1_R02D_BOUNDED_20261025.md)3700新轮/236800转移、原进程身份保持/OOM14/盘余418.10GB，原预算及100GiB保护不变，尚无有效步行结论。

2026-10-25：[P12C足地控制D十项通过](README_SUPERMIMIC_Q1_WALKING_CONTACT_20261025.md)，保留A/B/C原件及零距离退化证据，0.5mm余隙/双地面各30次向内求解与真实Q1十个配对通过，24组真实动作待。06:57原[R02D](README_SUPERMIMIC_Q1_R02D_BOUNDED_20261025.md)3475新轮/222400转移、三实际身份保持/OOM14，盘余464.32GB，100GiB停止线不变。

2026-10-25 06:42：[R02D保存3000独立十三门通过](README_SUPERMIMIC_Q1_R02D_BOUNDED_20261025.md)，三Adam计数、原340/6/7与516冻结保持，执行前3050新增轮，尚无最终动力学效果。[P12C足地原生约束](README_SUPERMIMIC_Q1_WALKING_CONTACT_20261025.md)准备，固定原根/时轴，真实控制/24组待；P12/B失败原件均保留。

2026-10-25 06:35：[P12B速度条件24组完成](README_SUPERMIMIC_Q1_WALKING_CONTACT_20261025.md)并独立8940核验/8436新inverse通过；锁脚失真减轻，但近地提升仍伴4.61至22.13mm穿地/高滑动，0准入。两轮48组/16个加避碰变体自穿透0；下一原生足地/可达性约束，原R02D继续。06:38现场2950新轮/188800转移、OOM14/盘余565.15GB。

2026-10-25 06:27：[P12独立FK/inverse核验完成](README_SUPERMIMIC_Q1_WALKING_CONTACT_20261025.md)，24组/8940核验及8436新inverse与保存结果一致，高度单判锁脚失真结论保持。P12B同train新增0.15m/s低速支撑条件，准备验证，原训练不改。

2026-10-25 06:22：[P12首轮24组完成但锁脚方案拒绝](README_SUPERMIMIC_Q1_WALKING_CONTACT_20261025.md)，高度单判把81至100%帧当支撑、脚目标改动最高1.04m，0新增准入。八baseline复现P11，原生腕部避碰八组自穿透0但未修复落地；下一独立重算/同train速度条件。R02D原预算继续。

2026-10-25 06:17：[P12步行接触对照准备](README_SUPERMIMIC_Q1_WALKING_CONTACT_20261025.md)，复用P11四个train来源，固定根/时轴比较支撑脚目标及Mink避碰，2/8rad/s共24组；本地三项来源/标签控制通过，服务器几何/原生不等式控制和真实结果待。R02D最后实读06:03为2050新增轮，唯一训练保持，BONES交师兄。

2026-10-25 05:55最新：[R02D](README_SUPERMIMIC_Q1_R02D_BOUNDED_20261025.md)1825新增轮/116800转移、三实际进程身份保持，越过原1810数值失败位置但效果仍待；保存1500审计B十三门全过/516冻结保持，A元数据断言失败保留。[P11](README_SUPERMIMIC_Q1_WALKING_RETARGET_20261025.md)188项保存数组独立重算误差0，限速单独不足以解决步行支撑；[P10/B](README_SUPERMIMIC_Q1_WALKING_HOLDOUT_20261025.md)0/15新walking准入，不以test调参。共享盘余785GB/本轮0.822GB，100GiB停止线保持。当前恢复入口已精简，旧全文移[历史归档](README_CURRENT_STATE_ARCHIVE_20261025.md)。

2026-10-25 05:40：[P11真实十二组对照完成](README_SUPERMIMIC_Q1_WALKING_RETARGET_20261025.md)，2rad/s确实截断部分步态，但4/8仍支撑不足、两片约9mm自穿透；不采用提高上限即准入，下一train接触一致性。服务器3控制0跳过、guard退出0/worker回收/0新参考。R02D1400新增轮继续，中间保存独立审计入口已准备、结果待。

2026-10-25：[P11训练walking限速诊断准备](README_SUPERMIMIC_Q1_WALKING_RETARGET_20261025.md)以固定四个train目录比较2/4/8rad/s，只改内存中IK限速，原root/目标/时轴/位置限位保持；排旧dev/test及P10预留目录，不导出准入、不改R02D。Q1 URDF标32/36仅元数据，真实控制与诊断结果待。

2026-10-25 05:30：[P10B十九门完成、0/15准入](README_SUPERMIMIC_Q1_WALKING_HOLDOUT_20261025.md)：原速步行候选全部足底支撑比例失败，14最大间隙/6自穿透失败；补检/guard原生退出0并回收，原P10 wrapper失败保留。没有可用补充walking评测，不放宽门；下一只用train来源区分映射与限速，不以新test调参。R02D最近1100新增轮继续，OOM14。

2026-10-25 05:27：[P10首次wrapper失败](README_SUPERMIMIC_Q1_WALKING_HOLDOUT_20261025.md)：15转换完成后独立检查同名模块误导入，0十九门准入，原WORKER_FAILED保留/worker回收。新wrapper已修为绝对路径加载，可在P10B新目录复用原转换补检；尚待执行。R02D1000新轮/64000转移继续，OOM14。

2026-10-25 05:25：[P10来源选择完成](README_SUPERMIMIC_Q1_WALKING_HOLDOUT_20261025.md)源码afb29d9已核远端/部署SHA、服务器4控制0跳过，52未用walking块/17原事件候选/15目录候选固定选15片，原split保留、全部只供补充test；转换/十九门与动力学待，全源隔离仍false。R02D950新增轮继续、未改输入。

2026-10-25：[P10独立步行补充测试准备](README_SUPERMIMIC_Q1_WALKING_HOLDOUT_20261025.md)新增固定HumanML来源/镜像/原事件/目录候选排除和原split保留的选择、转换入口，本地四项来源控制通过，服务器真实选择及十九门待。只供补充test、不改[R02D](README_SUPERMIMIC_Q1_R02D_BOUNDED_20261025.md)340/6/7与7000新增轮；05:16实读725轮、500保存已存在、进程身份匹配，尚无修正后移动效果。BONES交师兄。

2026-10-25：[S02动作域修正验证完成](README_SUPERMIMIC_Q1_ACTION_DOMAIN_20261025.md)服务器5控制、1500保存15门、真实replay H32/BPTT7门均通过，原496冻结保持；正常探针未重现崩溃，下一原5000父/340片/相同seed/7000轮独立R02D，登记启动待，不宣称跟随改善。

2026-10-25 04:17：[R02C实际数值FAILED](README_SUPERMIMIC_Q1_R02_PARTIAL_CONTINUATION_20261025.md)于04:07:44新增1810轮退出，累计6810、未完成最终配对；496冻结保持、原worker回收/OOM14。最后日志动作惩罚/critic爆大，先定位真实/想象动作域不一致，不盲目续跑；step1500待张量审计。下方RUNNING为历史快照。

2026-10-25：[P09来源排除视图登记](README_SUPERMIMIC_Q1_SOURCE_VIEW_REGISTRATION_20261025.md)独立入口已发布，两端四项控制及真实预检通过，显式绑定排除前/后来源证据且保留原几何/held-out；612训练/dev6/test7，未登记或启动第二训练。R02C已FAILED，先处理数值异常，BONES交师兄。

2026-10-25 04:01：[PHUMA P08B-V2真实完成](README_SUPERMIMIC_Q1_PHUMA_LIBRARY_20261025.md)，排除4条训练目录候选冲突后train612片/36.52分钟，dev6/test7不变；保存时轴/原trainer准入及独立HumanML复审通过，全源隔离仍false。[R02C](README_SUPERMIMIC_Q1_R02_PARTIAL_CONTINUATION_20261025.md)1625新增轮/累计6625、三进程身份及496冻结通过，GPU40/OOM14/盘余1.385TB，未热改340输入，无新效果结论。

2026-10-25 03:41：[PHUMA B019完成](README_SUPERMIMIC_Q1_PHUMA_CONCURRENT_EXPANSION_20261024.md)，额外1024来源全部处理完，算术616训练片/36.73分钟待完整组库。[PHUMA-D04](README_SUPERMIMIC_Q1_PHUMA_REFERENCE_DYNAMICS_20261025.md)8片/1441帧及736独立重算通过，发现静态过门参考仍浮脚/近地滑动，不据此宣称唯一失败原因。[R02C](README_SUPERMIMIC_Q1_R02_PARTIAL_CONTINUATION_20261025.md)继续1025新轮/累计6025，OOM14，尚无新效果证明。下一P08组库与新增来源审计。

2026-10-25 03:30：[PHUMA B018完成/B019启动](README_SUPERMIMIC_Q1_PHUMA_CONCURRENT_EXPANSION_20261024.md)，B018新增94片/328.50秒，worker回收及四绑定SHA通过；各批算术507训练片/30.35分钟，B019正处理最后352候选，完整组库/新增来源审计待。`locomotion`仅位移阈值标签，不代表步行语义。[R02C](README_SUPERMIMIC_Q1_R02_PARTIAL_CONTINUATION_20261025.md)575新轮/累计5575，固定340输入与7000预算保持，OOM14。BONES交师兄，无新效果结论。

2026-10-25 03:16：[R02C新轮真实更新](README_SUPERMIMIC_Q1_R02_PARTIAL_CONTINUATION_20261025.md)step0继承12门/496冻结通过，300新轮/19200转移、WM69组/actor19次、三进程身份核验通过，OOM14/内存18.64GB/盘余1.620TB；未有续训后效果。另[B018](README_SUPERMIMIC_Q1_PHUMA_CONCURRENT_EXPANSION_20261024.md)固定offset288/count384单worker在跑，不能预计合格时长。目标未达、原FAILED不重启、BONES交师兄。

2026-10-25 03:12：[R02C真实登记/启动](README_SUPERMIMIC_Q1_R02_PARTIAL_CONTINUATION_20261025.md)从5000保存继承，固定7000新轮、496冻结；03:12:55实读50轮/3200新转移，CPU初始继承审计进行中，WM/actor尚在预热。发布d21a880已核Q1，原FAILED不重启，OOM14/盘余1.640TB。后续配对评测尚未执行。

2026-10-25：[R02C独立续训准备](README_SUPERMIMIC_Q1_R02_PARTIAL_CONTINUATION_20261025.md)从已审计5000保存追加原剩余7000轮，旧FAILED不重启；新loader/控制器/初始继承检查已实现，服务器验证与实际启动待。[B017B](README_SUPERMIMIC_Q1_PHUMA_CONCURRENT_EXPANSION_20261024.md)真实新增33片/123.70秒，累计算术413片/24.88分钟未合库，OOM14。

2026-10-25：[R02第5000轮真实开发对照](README_SUPERMIMIC_Q1_R02_DEVELOPMENT_20261025.md)四阶段退出0，完整5→6/6、跟随仍0/6；移动只走0.078/参考0.521m。下一PHUMA扩库及移动模型诊断，原失败不重启。[P07](README_SUPERMIMIC_Q1_SOURCE_ALIASES_20261025.md)02:57正式审计完成，HumanML内部原动作跨split0但全源隔离未证明。BONES继续交师兄。

2026-10-25 02:53：[5000轮保存真实审计通过](README_SUPERMIMIC_Q1_RESOURCE_RECOVERY_20261025.md)，16门/469冻结/三Adam计数通过，原R02失败保留。六段development初始/5000动力学对照入口已准备，真实结果待；不以保存可用宣称策略成功。

2026-10-25 02:51：[六个旧失败G1残留已精确回收](README_SUPERMIMIC_Q1_RESOURCE_RECOVERY_20261025.md)，容器内存下降21.20GB至12.76GB，GPU6348→14MiB/OOM14，0文件删除；原R02 FAILED保留。下一审计5000保存、六段开发初末动力学对照，尚无新训练或效果结论。

2026-10-25 02:46：[连接与资源恢复](README_SUPERMIMIC_Q1_RESOURCE_RECOVERY_20261025.md)18864已恢复，R02 FAILED原样确认；六个10月21日G1失败评测残留身份/错误日志/无进展已核，尚未回收。下一原样证据→精确回收→5000保存及六段开发配对评测。BONES继续交师兄，原FAILED不重启，盘余约1.807TB/OOM14。

2026-10-25 02:20:13最新实核：[原PHUMA R02内存保护FAILED](README_SUPERMIMIC_Q1_EXISTING_RESOURCES_20261024.md)，02:08:57第5300轮/180.61分钟退出1，未执行配对评测；5000权重待独立核验。OOM14/盘余约1.965TB，之后SSH转发及直连超时，Mac Tailscale中继异常，现场归因待恢复。[P07](README_SUPERMIMIC_Q1_SOURCE_ALIASES_20261025.md)两端4控制过但正式审计内存预检退出1，未创建输出。已保存[衍生故障收据](../data/manifests/supermimic_q1_R02_failure_observation_20261025.json)，不重启/改写失败实验或放宽保护；下方RUNNING条目为历史快照。

2026-10-25：[P07来源别名审计准备](README_SUPERMIMIC_Q1_SOURCE_ALIASES_20261025.md)严格以353片独立准入名单连接95 HumanML，区分事件名/镜像/裁片/受试者候选，4项本地控制通过，正式服务器收据待；不把名字不同当独立测试。原R02继续，AMASS P06 0/4不采用，B017保护暂停。

2026-10-25：[AMASS P06局部偏移真实对照](README_SUPERMIMIC_Q1_AMASS_LOCAL_OFFSET_20261025.md)01:57完成、0/4准入；两圆走残差下降但慢走支撑恶化/新增自碰撞，新配置不采用。服务器4控制通过、worker回收/OOM14。下一已有PHUMA来源审计，不盲目恢复B017；原R02继续，BONES交师兄。

2026-10-25：[AMASS P05B真实归因](README_SUPERMIMIC_Q1_AMASS_MAPPING_20261025.md)804帧/20对账通过，实际固定肘腕12.698cm而目标P95为16.31–25.08cm，目标长度不兼容；三圆走足底目标本身与IK均有偏差。A错误“共用连杆”判断已撤回，原映射/参考未改；下一独立小样本尺度/校准、跨源隔离核对。

2026-10-25：[AMASS P05A失败与B修正](README_SUPERMIMIC_Q1_AMASS_MAPPING_20261025.md)：交接肘腕共用判断有误，实际映射已用独立腕部连杆，A前提检查退出1/回收。B查实际固定连杆与足底，四项Mac控制通过，真实执行待。原R02 01:27继续4175轮/139.12分钟/OOM14；参考与预算不改，BONES交师兄。

2026-10-25：[AMASS P04真实检查](README_SUPERMIMIC_Q1_AMASS_RETARGET_20261025.md)16段Q1候选转换/独立22门完成，0/16全通过，足底支撑15/残差11/最大间隙10项失败，0新增训练准入。人体FK/时轴正确，但仍须修适配质量，下一源/目标/IK归因。[P03](README_SUPERMIMIC_Q1_AMASS_PILOT_20261025.md)固定HumanML索引01:07接收成功，来源别名/受试者隔离未核完，不合库。[原R02](README_SUPERMIMIC_Q1_EXISTING_RESOURCES_20261024.md)01:10:25继续3650轮/233600交互、121.33分钟、GPU40/OOM14，最终动力学待。[P06](README_SUPERMIMIC_Q1_PHUMA_CONCURRENT_EXPANSION_20261024.md)40新增合格/算术22.81分钟未合库，B017待资源释放。BONES交师兄不再获取，原预算保持，移动目标未达。

2026-10-24当前：[现有资源R02实际启动](README_SUPERMIMIC_Q1_EXISTING_RESOURCES_20261024.md)。23:08独立PHUMA340训练片/20.45分钟、64环境/4槽/H32/12000轮启动；23:12核验300轮/19200交互、WM69组/actor19次、critic与梯度有限、469冻结文件及进程身份通过/OOM14。冻结spec、启动及核验收据已保存，尚无效果结论。BONES交师兄处理，临时凭据/18866已清理；AMASS与16维SMPL-X到位但未Q1准入。[D03B](README_SUPERMIMIC_Q1_PHUMA_MODEL_20261024.md)和[模型槽](README_SUPERMIMIC_Q1_EXPANDED_RUN_20261024.md)实际完成证据已恢复；下方等待条目为历史状态。

2026-10-24：[BONES V01独立准入准备](README_SUPERMIMIC_Q1_BONES_INDEPENDENT_20261024.md)22项源FK/映射/原速/几何/导出门及数值指纹已实现，服务器8控制通过；不等于真实参考或动力学跟随。用户确认Windows下载失败，要求代理Git获取；下一官方Git/LFS/Hub权限/服务器直连，0新真实BONES参考/训练，22:07盘余1.965TB/OOM14。

2026-10-24：[BONES C01专用原速重定向](README_SUPERMIMIC_Q1_BONES_RETARGET_20261024.md)实现及服务器5控制通过，含合成tar→名单→G1 FK→Q1 IK/导出，训练标志保持false。收据路径类型故障A保留，修正B通过；原120Hz时间/位移及既有门不改。真实包与独立BONES准入仍待，0新真实参考或训练。

2026-10-24：[BONES M01源G1模型核对](README_SUPERMIMIC_Q1_BONES_G1_MODEL_20261024.md)。现有候选与宇树官方固定版本骨架/关节轴/限位/足碰撞数组差0，64随机姿态FK差0；不证明BONES生成用FBX身份，实际动作待核。Windows D:\tmp最大partial已6.895GB，下载文件写入推进，仍无完整包。下一BONES专用重定向/独立质检，0新真实参考或训练。

2026-10-24：[BONES P02-F02列名勘误](README_SUPERMIMIC_Q1_BONES_FORMAT_V2_20261024.md)。V1误用Hips及短关节名，真实官方G1协议为root_*与完整joint名加_dof；此前测试共用错误常量。现V2以独立表头及全29关节回归，服务器11控制/SHA通过，旧收据保留。此前无真实BONES输入/Q1参考/训练；实际原包仍待接收。

2026-10-24：[BONES P02-A01原包接入准备](README_SUPERMIMIC_Q1_BONES_ARCHIVE_20261024.md)。完整SHA/gzip CRC、成员索引和仅冻结名单提取两端11控制通过；真实原包尚未到位、0新Q1参考/训练。21:16 Windows最大partial与20:51同大小，未验证浏览器下载状态；21:22服务器无原包/OOM14/GPU38/盘余1.965TB。现有G1模型仅载入/FK候选，尚非源模型身份通过。下一源模型与独立重定向准备。

2026-10-24：[BONES P02-F01独立CSV/原速准备](README_SUPERMIMIC_Q1_BONES_FORMAT_20261024.md)。固定官方查看器/NVlabs来源确认文档120Hz/厘米/度，按列名解析及120→50时间插值服务器9控制通过；Mac缺scipy仅语法通过。实际原包/模型未验证，事件时段存在端点偏差、不参与裁片。下一流式原包接入，0新BONES参考或训练。

2026-10-24：[BONES-SEED主线目标](README_SUPERMIMIC_Q1_BONES_GOAL_20261024.md)已开始。[P01元数据审计](README_SUPERMIMIC_Q1_BONES_METADATA_20261024.md)142220行/352703事件及71117来源组隔离通过，固定train1024/dev128/test128独立候选。[P00存储/worker保护](README_SUPERMIMIC_Q1_BONES_RUNTIME_20261024.md)两端8控制及真实预算非GPU探针通过，磁盘余1.965TB/OOM14；仅对显式新调用生效。用户确认完整包仍在D:\tmp下载，下一P02原包接入，尚无BONES参考/训练。未知fps不算合格秒数，此前AMASS优先建议被新目标覆盖。

2026-10-24：[PHUMA/BONES-SEED/AMASS价值与优先级](README_SUPERMIMIC_Q1_DATASET_PRIORITY_20261024.md)。建议当前新增准备优先AMASS小移动库，BONES-SEED负责论文数据对齐，PHUMA保留现成基线；严格来源复现优先BONES、立即现成闭环PHUMA成本最低。尚无Q1公平排名，未新转换/训练。

2026-10-24：[SMPL-X身体模型接收与AMASS验证](README_SUPERMIMIC_Q1_SMPLX_INTAKE_20261024.md)。D:\tmp原包411448739字节直接迁服务器且完整SHA一致；三模型MD5、实际16betas和有限样本FK通过，固定版本与默认入口就绪。原包/模型约0.823GB，0新Q1参考或训练；lockedhead原拟合模型身份与Q1质量待验证。下方身体模型缺项为接收前快照。

2026-10-24用户已恢复探索与训练：[资源、Windows接收和官方入口](README_SUPERMIMIC_Q1_RESOURCES_FIXES_20261024.md)。实际基础组库19:13:16退出0，train340/1227.10s（移动62/221.76s）、dev6/21.26s、test7/27.56s，完整时轴/来源及原trainer准入通过；新训练未启动。Windows指定lzl目录访问被拒绝，已验证C:\Users\yu\YuanQi_inbox可读写接收新数据。下一后端与模型诊断、磁盘预算及独立训练，下方暂停/准备条目为历史记录。

2026-10-24当前执行：[Q1扩库训练准备](README_SUPERMIMIC_Q1_EXPANDED_RUN_20261024.md)。原1861候选全部完成，批次算术累计train340/1227.10s（移动62/221.76s）、dev6/21.26s、test7/27.56s，尚待完整时轴重算/真实组库。新增1024固定训练来源已选出、未转换。7份代码发布f039d77并核SHA、组库两端6控制通过；后端数值/梯度/资源与新R02尚未执行。18:40:50实核无相关计算进程、OOM14/GPU38。下一基础组库与模型槽验证，再单路处理B015至B018，详见[P04](README_SUPERMIMIC_Q1_PHUMA_EXPANSION_20261024.md)。

2026-10-24当前优先：[AMASS与BONES-SEED接入盘点](README_SUPERMIMIC_Q1_SOURCE_EXPANSION_20261024.md)。当前服务器已找到AMASS原库，17928 NPZ中17265标准动作/标称56.34h，490静态/173待处理；全头部索引及独立重算完成。缺SMPL-X男女模型及完整BONES G1包/metadata，精确清单已登记；BONES匿名下载401。用户已授权三源扩库，尚未新增Q1参考/训练。

2026-10-24：[SuperMimic论文数据用法与PHUMA扩量](README_SUPERMIMIC_Q1_PHUMA_DATA_USE_20261024.md)。论文主库264.56h、LAFAN1对照2.45h；当前实际11训练片/41.30s。已有1秒50/40/10采样，首轮配额与保守准入限制有效量；现有温和训练候选标称26.44h，尚未去重/Q1准入。完成论文/源码/索引复核，下一独立P04扩量协议，0新训练。

2026-10-24：[PHUMA D03共同轨迹奖励/模型诊断](README_SUPERMIMIC_Q1_PHUMA_MODEL_20261024.md)：首次真实执行因评测划分标签适配失败退出1，未加载模型张量/创建输出；已修新入口且本地四控制通过。D03B延后，无真实模型结果、新训练。

2026-10-24：[PHUMA R01完整训练与配对结果](README_SUPERMIMIC_Q1_PHUMA_20261024.md)：14:36:50全部7阶段原生退出0；3000轮/48000转移、97.6787分钟，11项保存/两份18片轨迹审计、121冻结及三划分算术通过。最终16/18完整、0/18过跟随门；test非移动片关节误差0.272425→0.240971rad，开发移动仍明显不足。V01图表已完成，下一D02C，原终态保留，不盲目追加预算。

2026-10-24：[PHUMA V01配对评测图表](README_SUPERMIMIC_Q1_PHUMA_PLOTS_20261024.md)：真实18片绘图退出0、36组根/关节误差重算一致；两图目视与大小/SHA通过，3段合格移动均纳入，test无移动片。图像支持移动仍不足，不新增物理/策略/训练；下一D02C。

2026-10-24：[PHUMA D02C足底高度映射诊断](README_SUPERMIMIC_Q1_PHUMA_SUPPORT_MAPPING_20261024.md)：8训练片/1536帧原生退出0，32几何检查/264统计重算及121冻结通过，OOM14。支撑失败片在源固定落地/映射已有缺口，IK/目标曲线差0.56–3.04mm；不能据此解释合格移动片学习失败。D02/D02B历史失败保留，下一PHUMA奖励/模型诊断。

2026-10-24：[PHUMA→Q1首轮训练](README_SUPERMIMIC_Q1_PHUMA_20261024.md)13:21实核875/3000轮、14000转移/249次actor，原进程/冻结条件通过、GPU40摄氏度/OOM14。[D01](README_SUPERMIMIC_Q1_PHUMA_DIAGNOSIS_20261024.md)第500轮15项保存审计通过，三网络变化、Adam1490/99/1980；112候选统计/独立重算确认支撑与自碰撞是筛选主瓶颈，18合格/94拒绝不变。仅11训练/4开发/3测试，test无移动片；完整配对效果待原流水线。

2026-10-24 11:26：[Q1 R04V2用户停止并切换PHUMA](README_SUPERMIMIC_Q1_HORIZON_PAIR_20261024.md)：A932轮/14912交互后正常停止、B未启动；11:27进程组无活跃成员、权重SHA/冻结核验通过。旧A3来源参考不进入新实验，后续PHUMA训练以上方最新条目为准。

2026-10-24 10:24：[Q1 D08动作响应完成](README_SUPERMIMIC_Q1_ACTION_RESPONSE_20261024.md)：4基线精确复现、2176控制步/192响应独立审计通过；走路根位置H8为16/16正向、相对误差0.441，H32为11/16/0.760。下一同R03父权重/种子的H32-H8等预算匹配试验，0新增训练。

2026-10-24 10:05：[Q1 D07奖励/模型诊断完成](README_SUPERMIMIC_Q1_REWARD_MODEL_20261024.md)：两端3控制、六段奖励/18项预测完成；走路回报增量95.47%为身体朝向项，根位置项下降。最终H32走路根坐标预测误差41.37mm/CV9.67mm，根速度误差6.23倍CV，整体loss掩盖根部问题。下一有限预算动作响应核查，0新增训练。

2026-10-24：[Q1 V02/R03回放完成](README_SUPERMIMIC_Q1_VISUAL_R03_20261024.md)：三段826/579/339帧完整EGL渲染、ffprobe/身份、九帧目视与浏览器播放通过；新入口18865，原18863为R01。走路仍近原地，下一核查参考/奖励/世界模型响应，不追加原R03预算。

2026-10-24 09:28：[Q1 R03最终结果](README_SUPERMIMIC_Q1_R03_CONTINUATION_20261023.md)：追加12000轮、累计15000轮训练及八阶段配对评测全部完成；保存13门/冻结61项/终态事件与summary重算通过。三片全身RMSE降至0.1885/0.2011/0.2625rad，均未达0.15门；走路最远2.91cm/参考1.126m，仍未学会移动。下一实际V02渲染；以下Q1进度为历史快照。

2026-10-24 02:38：[Q1 D06追加预算在线趋势](README_SUPERMIMIC_Q1_BUDGET_TREND_20261024.md)：350条原日志前缀完整/有限，150条选定记录45个均值独立重算及前缀SHA通过；R03早/后窗口在线关节误差0.376→0.250rad、奖励+18.61%，不替代固定动作评测。原训练追加5975/12000轮、累计8975轮，进程/冻结条件通过、GPU40摄氏度/OOM14，继续原预算。

2026-10-24：[Q1 D05/D05B根高候选诊断完成](README_SUPERMIMIC_Q1_SUPPORT_PROJECTION_20261024.md)：三片生成、54项导出审计/276项动力学重算通过。足地接触100%但走路超cap帧0.06%→29.43%、根外力矩P95 2.67→96.09Nm，不直接接入训练；原失败保留。02:03:18实核原R03追加4950/12000轮、累计7950轮，冻结条件通过、GPU39摄氏度/OOM14，尚无最终效果。

2026-10-24 01:14：[Q1 D04参考接触/逆动力学诊断](README_SUPERMIMIC_Q1_REFERENCE_DYNAMICS_20261024.md)两端控制、3475帧/6950次inverse及276项统计重算通过；另三峰值帧inverse精确复现。走路较低脚底平均离地19.68mm、实际接触1/1648；出拳7.64秒左腕/左髋穿透约5mm，左肩233.68Nm异常主要来自自碰撞约束，不能直接增cap。0积分/推理/optimizer，学习失败主因未定。原R03仍3525/12000追加轮、累计6525轮，原进程/冻结条件通过，GPU39摄氏度/OOM14，尚无最终效果。

2026-10-24 00:28：[Q1 V02新协议渲染准备](README_SUPERMIMIC_Q1_RENDER_PREP_20261024.md)两端各4项控制及R02B三片真实只读预检通过，取回小证据逐项大小/SHA一致；预期826/579/339帧，0新物理/推理/渲染。EGL和完整视频仍待执行。原R03实核2150/12000追加轮、累计5150轮，原进程/冻结条件通过，GPU39摄氏度/OOM14；两个训练源码目录保持冻结。

2026-10-23 23:41：[Q1 R03固定追加12000轮](README_SUPERMIMIC_Q1_R03_CONTINUATION_20261023.md)实时800轮/12800新交互；第500轮独立CPU核验14项通过，真实Adam12980/1198/23960、三网络有限且变化，原进程/冻结合同一致。父网络/完整Adam/bin共8项继承精确一致，训练进行中、尚无最终效果。此前[R02最终对照](README_SUPERMIMIC_Q1_BATCH_TRAINING_20261023.md)18阶段通过，B三片完整、腿部较好，但走路仍约3cm；D03四份前缀分析完成，原终态保留。

2026-10-23：[Q1 D03相同前缀诊断](README_SUPERMIMIC_Q1_PREFIX_DIAGNOSIS_20261023.md)代码2ca9661发布/部署、两端3项控制和A真实只读分析完成。相同走路前14秒全身0.3748→0.3466rad、腿部0.2167→0.2380rad，最终仍提前失败；浅蹲/开发出拳水平漂移增大。22:33核实B2225/3000轮，原进程及冻结条件通过，原训练继续。

2026-10-23：[Q1 E01/E03长预算续训入口准备](README_SUPERMIMIC_Q1_EXTENSION_PREP_20261023.md)：状态/Adam继承及11项控制通过；固定预算登记→训练→审计→三片配对流程另5项两端通过，未完成pair实际登记拒绝。代码d5f9476已发布，61项release逐文件部署通过；尚无真实续训，先等B及匹配比较。

2026-10-23 22:08：[Q1 R02修正摩擦批量对照](README_SUPERMIMIC_Q1_BATCH_TRAINING_20261023.md)B已1525轮/24400交互/509次actor且存活，step1500大小/SHA、冻结源码/输入通过，OOM仍14；无B/pair终态。A完成3000轮及完整配对/重算但未过门，走路700/1652步失败，三段关节RMSE0.3466/0.3693/0.3004rad。两臂结束再比较并选固定长预算，旧停止保留。

2026-10-23：[Q1 S01改进方案筛选](README_SUPERMIMIC_Q1_IMPROVEMENT_20261023.md)完成：摩擦修正后端、J01响应与独立重算、T01低边界饱和、B01全宽batch128资源通过，C01新入口绑定预检及旧协议拒绝通过。下一修正摩擦下batch32/128匹配对照，计划已保存，未启动训练。

2026-10-23：[Q1 D02学习效果诊断](README_SUPERMIMIC_Q1_LEARNING_DIAGNOSIS_20261023.md)完成：六段既有轨迹奖励拆分，回报增量89%/89%/94%来自软限位惩罚减少，走路全局位置项几乎不变；训练后段在线均值仍改善，但时间/超参数均未因果归因。0新增物理/推理/训练，R02未启动。

2026-10-23：[Q1 V01动力学可视化](README_SUPERMIMIC_Q1_VISUAL_20261023.md)完成：新跑R01最终策略三段完整MuJoCo闭环3487控制步，独立重算/全时长视频/9帧目视及浏览器三动作播放通过。结果与原评测一致、仍未达到跟随门；0新增训练，HTTP/SSH视频入口保留。

2026-10-23：[SuperMimic for Q1](README_SUPERMIMIC_FOR_Q1_20261023.md)：R01训练/配对整片/独立重算及D01模型、位移、三视频/9帧目视完成；关节误差改善但未过0.15rad。走路最大位移reference1.126m/policy0.041m，仍非可用跟随；下一P06摩擦修正接入与新后端核验，A3保持停止。

更新日期：2026-10-23（Q1阶段更新；通用研究条目仍以各自记录为准）。21:42恢复Windows跳板后核验B为825轮/13200交互/229次actor，原进程身份/冻结源码/输入通过，OOM仍14；R02继续运行，无停止或重启。

P05历史限制：R01脚地摩擦固定0.6、随机化被floor优先级覆盖，旧结果保留；S01/P07已在独立适配器修正并接入新入口，不追改R01。

## 恢复上下文先读

1. [当前状态与下一步](README_CURRENT_STATE.md)：目标、完成项、未验证项和下一条工作步骤。
2. 下方阶段索引中与当前工作对应的 README，再查看其中引用的日志与版本。
3. [训练工作包](训练工作包.md)及任务所需的服务器、存储说明。

用户明确要求：每完成重要部分立即写 README，不能只保留聊天。执行规则见根目录 [AGENTS.md](../AGENTS.md)，新记录使用[阶段模板](README_STAGE_TEMPLATE.md)。

## 阶段记录索引

| 日期 | 阶段 | 状态与证据 |
| --- | --- | --- |
| 2026-10-26 | 每日研究静默检查 | 46研究请求全部成功，关注源无实质变化；两份10/23旧v2仅HTML元数据等变化，去重111→113，候选保持、不新增简报；见[检查](README_RESEARCH_CHECK_20261026.md)，main/Q1文档已推送，09:23:32独立内容核验通过 |
| 2026-10-25 | 每日研究静默检查 | 43请求均成功，列表/主要关注源未变，SMR仅页面样式；111条去重保持，不新增简报，见[检查](README_RESEARCH_CHECK_20261025.md)及[回执](../data/manifests/research_watch_sync_20261025.json)，main/Q1文档已推送，09:21:54独立内容核验通过 |
| 2026-10-25 | P15足地间隙恢复 | C五控制全过、A/B失败保留，修正实际floor缓存/mesh+sphere导数；八组因余盘不足未运行，见[阶段记录](README_SUPERMIMIC_Q1_FLOOR_RECOVERY_20261025.md) |
| 2026-10-25 | P14整段常数根高度适配 | 八组1438帧/独立3188核验和2812新inverse完成，0准入；16状态确认原生约束允许已有穿地零步，见[阶段记录](README_SUPERMIMIC_Q1_WALKING_ROOT_OFFSET_20261025.md) |
| 2026-10-25 | P13步行目标可达性 | 实际24组/4314帧及独立6068核验/fresh FK完成，修正目标每帧有外球冲突；P14常数高度候选待，见[阶段记录](README_SUPERMIMIC_Q1_WALKING_REACH_20261025.md) |
| 2026-10-25 | P12/P12B/P12C训练步行接触一致性 | 三轮72组及三次独立FK/inverse通过；足地约束消穿地但更浮脚，均0准入，见[阶段记录](README_SUPERMIMIC_Q1_WALKING_CONTACT_20261025.md) |
| 2026-10-25 | R02D动作域匹配试验 | 08:05存储保护停止于5282/7000新轮；5000保存十三门/退出保存十五门通过，配对未执行，不自动恢复，见[阶段记录](README_SUPERMIMIC_Q1_R02D_BOUNDED_20261025.md) |
| 2026-10-25 | S02动作域一致性与数值失败 | 5控制、1500保存15门、真实replay H32/BPTT7门通过，未精确重现崩溃/证明移动改善，见[阶段记录](README_SUPERMIMIC_Q1_ACTION_DOMAIN_20261025.md) |
| 2026-10-25 | P09来源排除视图训练登记 | 两端四项控制及真实服务器预检通过，612训练/dev6/test7；无第二训练，见[阶段记录](README_SUPERMIMIC_Q1_SOURCE_VIEW_REGISTRATION_20261025.md) |
| 2026-10-25 | PHUMA P08/P08B扩展库与来源排除 | P08完整组库，P08B-V2排除4条冲突后612训练片/36.52分钟；准入/保存时轴及独立HumanML复审通过，全源隔离仍false，见[阶段记录](README_SUPERMIMIC_Q1_PHUMA_LIBRARY_20261025.md) |
| 2026-10-25 | PHUMA D04参考接触与逆动力学 | 8段train/1441帧/2882次inverse及736独立重算完成，浮脚/近地滑动缺口确认，见[阶段记录](README_SUPERMIMIC_Q1_PHUMA_REFERENCE_DYNAMICS_20261025.md) |
| 2026-10-25 | R02C独立续训 | 新增1810轮数值FAILED，未执行最终配对；最后1500保存已独立15门通过，原终态保留，见[阶段记录](README_SUPERMIMIC_Q1_R02_PARTIAL_CONTINUATION_20261025.md) |
| 2026-10-25 | R02六段开发对照 | 真实初始/5000动力学及两次轨迹审计完成，完整5→6/6、跟随0/6，见[阶段记录](README_SUPERMIMIC_Q1_R02_DEVELOPMENT_20261025.md) |
| 2026-10-25 | G1失败残留资源恢复 | 精确回收六个旧失败进程，释放21.20GB，0文件删除/无新OOM，见[阶段记录](README_SUPERMIMIC_Q1_RESOURCE_RECOVERY_20261025.md) |
| 2026-10-25 | P07跨源原动作别名 | 正式353参考审计完成，HumanML95段内部跨split0，全源隔离仍未证明，见[阶段记录](README_SUPERMIMIC_Q1_SOURCE_ALIASES_20261025.md) |
| 2026-10-25 | AMASS P06局部偏移对照 | 真实4片/22门完成，0准入，新配置不采用，见[阶段记录](README_SUPERMIMIC_Q1_AMASS_LOCAL_OFFSET_20261025.md) |
| 2026-10-25 | AMASS P05映射/支撑归因 | A错误前提退出1/回收；B804帧/20对账通过，目标刚性长度与支撑存在缺口，见[阶段记录](README_SUPERMIMIC_Q1_AMASS_MAPPING_20261025.md) |
| 2026-10-25 | AMASS P02原速移动候选 | 43探查选16个KIT四秒原始窗口/64秒，16betas/120Hz，尚未FK/Q1准入或分配split，见[阶段记录](README_SUPERMIMIC_Q1_AMASS_PILOT_20261025.md) |
| 2026-10-24至25日 | PHUMA P06独立扩库与资源隔离 | 额外1024来源全部转换，新增276合格/976.48秒；算术616训练片/36.73分钟待完整组库，见[阶段记录](README_SUPERMIMIC_Q1_PHUMA_CONCURRENT_EXPANSION_20261024.md) |
| 2026-10-24 | Q1扩库训练与低内存仿真准备 | 多名单组库两端6控制通过，真实组库/后端验证/新训练待执行，见[阶段记录](README_SUPERMIMIC_Q1_EXPANDED_RUN_20261024.md) |
| 2026-10-24 | PHUMA P04分批扩量 | 原1861候选转换完成；train340/1227.10s为批次累计，新增1024训练候选已选出；0新训练，见[阶段记录](README_SUPERMIMIC_Q1_PHUMA_EXPANSION_20261024.md) |
| 2026-10-24 | AMASS/BONES-SEED来源扩展 | AMASS17265动作/标称56.34h索引与独立算术通过；缺身体模型/BONES原包，0新训练，见[阶段记录](README_SUPERMIMIC_Q1_SOURCE_EXPANSION_20261024.md) |
| 2026-10-24 | SuperMimic数据用法与PHUMA扩量路线 | 原文/源码/既有索引复核完成，41.30秒小库差距与标称26.44h源候选已登记，扩量/训练未运行，见[阶段记录](README_SUPERMIMIC_Q1_PHUMA_DATA_USE_20261024.md) |
| 2026-10-24 | PHUMA D03奖励/共同轨迹模型 | 真实执行元数据失败保留，新入口适配及本地四控制通过，D03B延后，见[阶段记录](README_SUPERMIMIC_Q1_PHUMA_MODEL_20261024.md) |
| 2026-10-24 | PHUMA V01配对图表 | 真实18片/36组误差重算及两图目视/大小/SHA通过，移动仍不足，见[阶段记录](README_SUPERMIMIC_Q1_PHUMA_PLOTS_20261024.md) |
| 2026-10-24 | PHUMA D02源/目标/IK支撑诊断 | D02C八片1536帧/264统计重算通过，源/映射贡献筛选缺口但策略失败主因未定，见[阶段记录](README_SUPERMIMIC_Q1_PHUMA_SUPPORT_MAPPING_20261024.md) |
| 2026-10-24 | PHUMA D01保存/数据筛选诊断 | 第500轮15项保存审计通过；112片/19门/12组统计独立重算，支撑与碰撞为筛选主瓶颈，见[诊断记录](README_SUPERMIMIC_Q1_PHUMA_DIAGNOSIS_20261024.md) |
| 2026-10-24 | PHUMA数据切换与Q1首轮训练 | 3000轮及全部配对完成，最终16/18完整、0/18过关；保存/两次轨迹/121冻结核验通过，见[阶段记录](README_SUPERMIMIC_Q1_PHUMA_20261024.md) |
| 2026-10-24 | Q1 R04/R04V2 H32/H8等预算对照 | 用户停止切换PHUMA；A932轮/14912交互、B未启动，无完整对照，旧终态保留，见[阶段记录](README_SUPERMIMIC_Q1_HORIZON_PAIR_20261024.md) |
| 2026-10-24 | Q1 D08最终模型动作响应 | 2176步/192响应与数组独立审计完成，支持H8/H32匹配试验，见[阶段记录](README_SUPERMIMIC_Q1_ACTION_RESPONSE_20261024.md) |
| 2026-10-24 | Q1 D07最终奖励/世界模型诊断 | 六段奖励与18项模型预测完成，走路根位置H32误差4.28倍CV，见[诊断记录](README_SUPERMIMIC_Q1_REWARD_MODEL_20261024.md) |
| 2026-10-24 | Q1 V02/R03完整轨迹回放 | 三段完整EGL/ffprobe/九帧目视/浏览器播放通过，走路仍近原地，见[回放记录](README_SUPERMIMIC_Q1_VISUAL_R03_20261024.md) |
| 2026-10-24 | 每日研究简报与Q1研究记录 | PhysibleMDM/MPBRS/DiffDreamer三项；8摘要/3相关正文/4固定文本，111条去重，未复现；见[简报](research/2026-10-24/README.md)和[阶段](README_RESEARCH_CHECK_20261024.md)，main/Q1文档独立核验通过，见[回执](../data/manifests/research_watch_sync_20261024.json) |
| 2026-10-24 | Q1 D05/D05B根高支撑候选 | 三片/54导出/276统计通过，走路动力学变坏、保留失败对照不接入训练，见[阶段记录](README_SUPERMIMIC_Q1_SUPPORT_PROJECTION_20261024.md) |
| 2026-10-24 | Q1 D04参考动力学诊断 | 两端控制、3475帧分析、276项重算及三峰值帧复现完成，确认支撑缺口/出拳峰值自碰撞，见[诊断记录](README_SUPERMIMIC_Q1_REFERENCE_DYNAMICS_20261024.md) |
| 2026-10-24 | Q1 V02新协议渲染准备 | 两端各4控制及三片真实只读预检通过，小证据SHA一致；EGL/完整渲染待执行，见[准备记录](README_SUPERMIMIC_Q1_RENDER_PREP_20261024.md) |
| 2026-10-24 | Q1 R03固定追加预算完成 | 累计15000轮及八阶段完整配对评测完成，三片关节改善但仍未过0.15门/移动验收，见[R03记录](README_SUPERMIMIC_Q1_R03_CONTINUATION_20261023.md) |
| 2026-10-23 | Q1 D03相同前缀分析 | 两端3控制及A/B四份真实配对诊断完成、0新物理；腿部/位置与完整率分开报告，见[分析记录](README_SUPERMIMIC_Q1_PREFIX_DIAGNOSIS_20261023.md) |
| 2026-10-23 | Q1 E01/E03长预算入口准备 | 11+5项控制及61项release用于真实R03；实际训练/完整评测已完成，效果见R03，入口证据见[准备记录](README_SUPERMIMIC_Q1_EXTENSION_PREP_20261023.md) |
| 2026-10-23 | Q1 R02批量训练对照 | 两臂18阶段与匹配比较完成、均未过跟随门；选B作固定追加预算起点，首轮停止保留，见[运行README](README_SUPERMIMIC_Q1_BATCH_TRAINING_20261023.md) |
| 2026-10-23 | Q1 S01改进方案筛选 | P07/J01及审计/T01/B01/C01完成，匹配batch对照草案保存、未训练，见[方案记录](README_SUPERMIMIC_Q1_IMPROVEMENT_20261023.md) |
| 2026-10-23 | Q1 D02学习效果诊断 | 既有日志/六段轨迹奖励拆分完成；区分预算嫌疑、已确认摩擦缺口与未验证动态/模型因素，见[诊断README](README_SUPERMIMIC_Q1_LEARNING_DIAGNOSIS_20261023.md) |
| 2026-10-23 | Q1 V01动力学可视化 | 新闭环仿真3487步、独立重算、三段完整视频及浏览器播放完成；见[V01记录](README_SUPERMIMIC_Q1_VISUAL_20261023.md) |
| 2026-10-23 | SuperMimic for Q1适配与R01训练 | 训练/整片配对评测与独立重算完成；误差改善但未过0.15rad，P05/P06接触限制见[阶段README](README_SUPERMIMIC_FOR_Q1_20261023.md) |
| 2026-10-10 | Q1衣服/FGP导出SMPL位置复核 | hp3090 E003/E004及两份旧录制身体包实际读取，Mac原录制/小包存在；根轨迹、接触与标定可信度及姿态质量待验收，见[数据位置README](README_Q1_SMPL_INVENTORY_20261010.md) |
| 2026-10-09 | Q1 UMR 小样本接入 | E008/E009完成；膝范围75.20°→限速后41.71°、视觉穿地2.89cm，25项回归通过，整体保真/平衡未通过，见[UMR README](README_Q1_UMR_20261009.md) |
| 2026-10-09 | Q1 GMR 与新方法对照 | 官方固定版本 GMR 的 Q1 实验映射完成八组 IK / 四段支撑 PD；限速后前踢范围 40.49°，未通过整体保真，确认 NMR 推理仅 G1、UMR 可适配新机器人，见[GMR README](README_Q1_GMR_20261009.md) |
| 2026-10-09 | Q1 重力轴与站姿校准 | 14 组对照完成，根轨迹串轴修复通过；合成站姿前踢幅度增大但目标残差变大，姿态保真未验收，见[校准 README](README_Q1_CALIBRATION_20261009.md) |
| 2026-10-09 | Q1 数据集对照 | 八样本离线 IK / 支撑 PD 完成；发现腿幅度压缩和首帧对齐将水平位移混进高度，未通过高保真 / 移动验收，见[数据集 README](README_Q1_DATASETS_20261009.md) |
| 2026-10-09 | Q1 新数据与训练入口研究 | 核验 MOSAIC、HiPHI、BONES/SOMA、PHUMA、HumanTracker 与 SONIC 训练入口；未新增大下载或训练，见[研究 README](README_Q1_DATASET_RESEARCH_20261009.md) |
| 2026-10-08 | Q1衣服 IMU 重跑 FGP | 两段转换、重定向/支撑 PD 与抽帧检查完成，结果 40c8f95 已核验发布 Q1；人体重建误差偏大，无支撑平衡/主机稳定性未验收，仅测试，见[衣服转换 README](README_Q1_CLOTH_FGP_20261008.md) |
| 2026-10-08 | Q1真实动捕服 FGP→SMPL 测试 | 两段共 3892 帧与输入 6 项检查通过，完整重定向/支撑回放完成；位移全零，诊断校准有显著残差，仅测试，见[真实输入 README](README_Q1_MOCAP_TEST_20261008.md) |
| 2026-10-08 | PredActor 专题阅读与 Q1 适用性 | v3 方法/实验及公开代码静态核验；目前仅正式发行 G1 评测，完整训练/部署待发布；Q1 基线优先，见[阅读 README](README_PREDACTOR_20261008.md) |
| 2026-10-08 | Q1骨架重定向通路 | Mink/quadprog 离线接口与 8 项检查通过，24 秒合成 FK→IK 及支撑回放通过；真实 SMPL 与动力学移动尚未验证，见[重定向 README](README_Q1_RETARGET_20261008.md) |
| 2026-10-08 | Q1仿真关节控制 | 9 项检查通过；支撑下 22 关节小幅跟踪通过，无支撑静态保持 5 秒；动态平衡/SMPL/训练尚未验证，见[仿真 README](README_Q1_SIM_CONTROL_20261008.md) |
| 2026-10-08 | Q1模型迁移与仿真入口恢复 | 完整模型及 SDK 核心 103 文件已在 hp3090 核验；3 个 MJCF 各 100 步被动仿真通过，可继续开发，完整 SDK/工具链仍待迁移，见[迁移 README](README_Q1_MIGRATION_20261008.md) |
| 2026-10-08 | Q1分支与版本管理 | 已从核验的`d086898`建立并发布`Q1`，工作副本跟踪`origin/Q1`；分支约定、发布核验与重要文件列表见[版本管理README](README_Q1_VERSIONING_20261008.md) |
| 2026-10-08 | Q1资料与SMPL全身遥操评估 | SDK全身动作播放限制、22DOF资产和文档/模型参数冲突已核验；后续迁移阶段已完成模型/核心SDK与基础加载检查，完整工具链及训练仍待完成，见[阶段记录](README_Q1_SMPL_20261008.md) |
| 2026-10-05 | 每日研究静默检查 | 已查公告、框架发布与六项项目/代码入口无实质变化；记录检查，不重复简报；见[运行 README](README_RESEARCH_CHECK_20261005.md) |
| 2026-10-04 | 每日研究检查与恢复 | 三项候选：探索幅度、SONIC 人体数据、多动作手部重定向；接续 10/3 中断，未运行算法；见[运行 README](README_RESEARCH_CHECK_20261004.md) |
| 2026-10-03 | 研究检查中断 | 保存来源缓存，SSH 超时，未完成简报；10/4 已接续；见[中断记录](README_RESEARCH_CHECK_20261003.md) |
| 2026-10-02 | 第二课·倾斜惩罚与对照准备 | 已解释评分权重并开放单项配置；CPU 数值/参数检查通过，权重 3 训练尚未运行；见[奖励项 README](README_LESSON02_REWARD_20261002.md) |
| 2026-10-02 | 第二课·实际 PPO 训练与重载 | Agent 基线与用户 `lesson02_my_first_run` 复跑均通过；平均维持 0.645→4.983 秒，独立重载一致；已补观测/动作/奖励代码对应，理解待确认、X2 未训练；见[实验 README](README_LESSON02_CARTPOLE_20261002.md) |
| 2026-10-02 | 学习目标与数学深度调整 | 用户明确希望训练 policy 并形成论文，不必限定底层算法；改为实践先行、数学按需，尚未训练或确认选题；见[路线说明](README_LEARNING_ROUTE_20261002.md) |
| 2026-10-02 | 每日研究检查 | 三项新候选：动作节奏、地形感知遥操、视觉中断定位；已去重旧项目，未启动实验；见[运行 README](README_RESEARCH_CHECK_20261002.md) |
| 2026-10-01 | 教学第一课·训练流程图 | 最新补充结构体对照：优化器改网络权重，保留经历记录；Critic 通过预测参与 Actor 的训练依据，理解待确认；见[流程图与变量记录](README_LESSON01_PIPELINE_20261001.md) |
| 2026-10-01 | 教学第一课·阶段复习与算法路线 | 学员答对固定 Actor/只更新 Critic 的两项判断；一步回报估计出现卡点，终止情形题暂缓，转到流程图；见[复习记录](README_LESSON01_REVIEW_20261001.md) |
| 2026-10-01 | 教学第一课·完整训练循环 | 学员已复述用差经历更新网络以改善后续行为；最新补充平均回报与评测边界，未运行新实验；见[第八小节](README_LESSON01_TRAINING_LOOP_20261001.md) |
| 2026-10-01 | 每日研究检查与中断恢复 | 三项新候选及 DRAM v2 实质更新，补充 H001 先行工作；未启动实验；见[运行 README](README_RESEARCH_CHECK_20261001.md) |
| 2026-09-29 | 学习路线与头环资料分析 | 已完成规划及静态分析；见[训练工作包](训练工作包.md)、[EgoLocate 分析](EgoLocate与头环分析.md)；未启动训练 |
| 2026-09-29 | hp3090 项目初始化与存储规范 | 已完成，初始化提交 `c7b64a7`；见[初始化 README](README_SETUP_20260929.md) |
| 2026-09-29 | 重要阶段记录与上下文恢复约定 | 已写入规则、状态入口和模板；见[本阶段 README](README_RECORDING_POLICY_20260929.md) |
| 2026-09-29 | 每日研究检查与首份简报 | 每天北京时间 09:00 的当前聊天任务已创建，首次定时触发待验证；见[阶段 README](README_RESEARCH_WATCH_SETUP_20260929.md) |
| 2026-09-30 | 首次定时研究检查 | 已实际触发，完成来源核验与三条简报；见[运行 README](README_RESEARCH_CHECK_20260930.md) |
| 2026-09-30 | 教学第一课·控制与学习 | 学员已答对力矩变小和不能据此判断学习；见[第一小节](README_LESSON01_20260930.md) |
| 2026-09-30 | P0 资源静态核验 | 找到训练候选与具体资产路径问题，CUDA 小计算通过；未运行训练/重载，P0 未通过；见[核验 README](README_P0_INSPECTION_20260930.md) |
| 2026-09-30 | 教学第一课·奖励与参数更新 | 单步策略梯度实验通过；学员答对需要更新参数及补充运动信息；见[第二小节及跟进](README_LESSON01_REWARD_UPDATE_20260930.md) |
| 2026-09-30 | 教学第一课·角速度与累计回报 | 学员选择 B 正确；clip 抑制更新与纠正目标的区别仍需巩固；见[第三小节及反馈](README_LESSON01_RETURN_20260930.md) |
| 2026-09-30 | 教学第一课·奖励设计检查 | 已提供反例检查方法；学员“失败不能进入 clip”的误解转入第五小节纠正，未实施机器人实验；见[第四小节](README_LESSON01_REWARD_DESIGN_20260930.md) |
| 2026-09-30 | 教学第一课·价值网络与优势 | 学员答对 +4 的方向；最新释疑区分事后回报与事前期望，说明特权观测和预测用途，理解待确认；见[第五小节](README_LESSON01_ACTOR_CRITIC_20260930.md) |
| 2026-09-30 | 教学第一课·PPO 裁剪目标 | 学员答对 4.8 平台与同批分母 20%；Critic 更新问题转入第五小节释疑，并补充单样本拟合不等于期望必错；见[第六小节](README_LESSON01_PPO_CLIP_20260930.md) |
| 2026-09-30 | 教学第一课·第一次阶段回顾 | 已串联同一份经历训练 Actor/Critic，记录最新复述边界和定期总结要求；理解仍在确认；见[阶段总结](README_LESSON01_SUMMARY_20260930.md) |
| 2026-09-30 | 教学第一课·Actor/Critic 参数更新 | CPU 固定样本正、负、零优势对照与 18 项行为检查通过；解释梯度与优化器更新，学员理解待确认；见[第七小节](README_LESSON01_ACTOR_CRITIC_UPDATE_20260930.md) |

## 当前状态

- 已确认训练主机 `hp3090`、RTX 3090 24GB 和 `/media/yu/FAFF-E9771` 的可写存储。
- 主项目位置固定为 `/media/yu/FAFF-E9771/YUANQI`。
- 目录按 `data / logs / script / reports` 组织；关键文本与代码提交到用户指定的 GitHub 仓库。
- 已发现百度网盘 Linux 官方客户端 8.7.0；未验证登录或上传/下载，未上传模型。
- 已完成基础倒立摆 PPO 仿真训练/保存/重载；尚未完成 X2 站立或人体动作跟踪训练，见第二课实验 README。

## 阅读顺序

1. [最新学习安排](README_LEARNING_ROUTE_20261002.md)与[训练工作包](训练工作包.md)：实践先行、数学按需，近期目标是第一次完整策略训练。
2. 第一课当前结合[阶段总结](README_LESSON01_SUMMARY_20260930.md)和[参数更新代码](README_LESSON01_ACTOR_CRITIC_UPDATE_20260930.md)，再按需查看小节：[控制循环](README_LESSON01_20260930.md)、[奖励与更新](README_LESSON01_REWARD_UPDATE_20260930.md)、[累计回报与 clip](README_LESSON01_RETURN_20260930.md)、[奖励设计](README_LESSON01_REWARD_DESIGN_20260930.md)、[价值网络与优势](README_LESSON01_ACTOR_CRITIC_20260930.md)、[PPO 裁剪目标](README_LESSON01_PPO_CLIP_20260930.md)；[完整参考讲义](第一课_机器人控制与PPO.md)按需阅读。
3. [学习路线与 OKR](学习路线与OKR.md)：三平台交付与科研方向。
4. [EgoLocate 与头环分析](EgoLocate与头环分析.md)：手部、SLAM、双目及已有融合方案。
5. [实验记录模板](实验记录模板.md)：每次实验复制后填写。

## 项目运行与存储

- [训练服务器与存储](README_INFRASTRUCTURE.md)
- [Git 与百度网盘保存规范](README_STORAGE.md)
- [既有资料来源](README_SOURCES.md)
- [初始化结果](README_SETUP_20260929.md)
- [每日研究跟踪规则](README_RESEARCH_WATCH.md)、[简报索引](research/README.md)、[候选研究想法](research/README_IDEAS.md)

下一项工作：用户已按说明完成[倒立摆复跑](README_LESSON02_CARTPOLE_20261002.md)；现已准备[倾斜惩罚权重 1→3 的对照](README_LESSON02_REWARD_20261002.md)，由用户运行后比较失败比例、维持时间与倾角，不直接比较不同评分规则的回报。基础仿真训练入口已通过，人形 X2 的资产/配置与训练缺口仍按既有 P0 核验推进。
