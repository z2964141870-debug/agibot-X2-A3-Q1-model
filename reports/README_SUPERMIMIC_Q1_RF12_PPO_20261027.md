# RF12 Q1真实仿真PPO与分阶段训练

记录日期：2026-10-27/28，北京时间。状态：准备/发布/三端核验完成，RF12于2026-10-28 01:02:02启动，三重复RF11基线后已进入真实PPO训练；效果待约3500实训秒检查点。用户明确要求执行本计划；本次预算例外覆盖RF11结束后的旧限制，不恢复旧实验、不激活agent。

## 目标与范围

改善实际上肢幅度、骨盆相对腕轨迹与根部跟踪，并保留稳定性。唯一新实验从RF11最终策略初始化，用4096个MuJoCo Warp环境和现有RSL-RL PPO；不以loss、reward或存活单项触发加训。

仅操作用户25386、`/opt/data/private/ZHY`四目录。权威Git现为`/Users/yu/Documents/ChatGPT/元启/data/repositories/q1-rf12-20261028`的Q1；Mac只收小记录、图和视频，权重及大轨迹留服务器LOCAL_ONLY。新代码在`script/supermimic_q1_ppo/`，保留RF11所有冻结来源。

## 阶段与预算

- A：原300训练/30验证、obs80/goal590、history10、RF11控制及原reward，改用真实物理PPO。累计约3500实训秒评测。
- B：若稳定但未达标，继续至约7100/10700实训秒。原reward上增加`alpha*(0.02*Rq+0.01*Rw)`，600实训秒渐进；上肢逐关节指数尺度0.8rad，骨盆heading相对腕位置尺度0.15m，原新增分项各自记录。
- C：首次达标后自动追加最多18000实训秒；3600实训秒内渐进达到50%原300/50%AP03新增片。三小时未达标停止，总累计不超过28800秒。

初始化、评测、绘图、等待不计实训。新critic前10个PPO迭代只更新critic，计入实训；同一learner进程暂停更新评测，保留优化器、归一化、RNG和训练环境。故障不重启、预算不重置。

固定PPO参数：std0.1不学习、24步采集、5epoch/4minibatch、lr1e-4、clip0.2、gamma0.99/lambda0.95、梯度上限1。策略目标编码器和归一化复制RF11；正态概率、GAE及critic保持FP32。环境数不是actor batch，实际每轮98304转移。

## 自动判据

固定30验证、两条普通走和双手抬起片，三重复，同协议重评RF11。以共同前缀比较跟踪，完整段另报完成率/失败；逐次与相应RF11重复对照。

| 条件 | 阈值 |
|---|---|
| 上肢RMSE、骨盆相对腕RMSE | 各降低至少15% |
| 固定双腕峰高误差 | 降低至少25% |
| 根XY、腿RMSE | 各恶化不超过10% |
| 验证 | 完成至少98%、提前终止最多1/30 |
| 固定普通走/抬手 | 三次均完整执行 |
| 指令振荡、承重滑速 | 各恶化不超过20% |

至少2/3重复满足整组条件且固定片三次完整才触发C。至少2/3出现验证失败>=6/30或根/上肢误差增加>=25%视为明显退步；A/B立即结束，C每小时检查且连续两次明显退步提前结束。任何缺失或非有限指标均不冒充达标。

## 数据、存储与研究

先建立资源索引和AMASS引用/软链接/使用核查，保存删除清单后仅删不用的原始部分，保留转换结果、元数据与必要模型。删除不等于备份。AP03最多1000训练，原300与30验证身份/演员隔离不变，新增优先普通走/上肢；沿用既有重定向，无自定性能门。

研究依据来自SuperMimic、BeyondMimic、RSL-RL官方项目/已安装代码，以及Diffusion Policy、LoRA原论文摘要。当前首先核优化信号与Q1适配；扩散需要相应动作示范，LoRA需要适合的预训练底座，不能由名称推断适用。作者讨论和视频说明只作辅助，X未获得可直接核验原帖，不引用转述作证据。见[来源核验收据](../data/manifests/supermimic_q1_RF12_research_20261028.json)；未观看完整视频或完成各论文复现，不将网页作者性能声明套到Q1。

## 运行条件与交付

开训前策略移植、动作单位、f+1时轴、自然终止/人工截断、reward分解、16与4096环境零optimizer契约必须通过。余盘100GiB、单实验12GiB、共享cgroup30GiB、86度、900秒无进展保护，单训练，不降4096。

逐阶段真实误差、抬手幅度/时序、原生接触/滑移和预算；六格PNG/PDF/CSV保留原值与500iteration均值，PPO无WM/想象指标明确N/A并另附PPO诊断；固定54秒8段6类型MP4交付Mac，保留跨片/失败标签。

## 实际现场与恢复

2026-10-27 23:49:58已按清单删除17265个不用的AMASS原始motion NPZ，逻辑179476251444字节；663个静态形体/未分类文件保留。活动AP02无raw依赖、目标树无指向raw软链接、无打开文件/工作目录使用。首次严格纳秒mtime预检因迁移截秒中止，零删除；核实大小和秒级时间一致后对每文件登记当前inode/device/mtime并在实际unlink前再核。原始元数据、历史12.20MB逐文件清单、所有转换结果与模型保留，删除不是备份。可用空间1378787356672→1558248005632字节，底层共享盘仍可能独立变化，不称逻辑删除字节为固定可用空间增量。

完整清单/结果/资源索引在服务器`data/supermimic_for_q1/RF12_PREPARATION_20261027/`；Git仅小收据`data/manifests/supermimic_q1_RF12_resources_20261027.json`及来源代码。AP02的330片身份、演员、坐标/帧率和SHA已建立统一索引，PHUMA/BONES/AMASS登记原料与历史转换入口；跨原数据集动作等价尚未建立，当前不盲目混库。

2026-10-27 23:39现场：SSH成功，boot`4d859008-4c83-4a6d-814d-cbb9dfe885c9`，GPU`GPU-26028785-9fb9-45af-e614-4cc53af15d08`、15MiB/0%/27度，无HP/PPO训练进程；余盘1430285090816字节。此为时点快照，开训前重核。

RF11最终SHA`6c4de3171bd45359691a63ffe1712729c14fdb2080f8066e4c25632f4bbc5146`，AP02清单SHA`ba9ac284a018cdd40eea5200d2c157b7da7cfdae89e6c820eed2c9748cc83337`。当前尚无RF12正式训练效果。

## AP03与PPO实测准备

AP03流式读取既有45.19GB SOMA原包一次，沿用`retarget_soma.process`、4workers/4iterations/root-relative-rest，新增583片无转换错误；原300训练/30验证路径与SHA保持，训练883片/236演员/4714.6秒，验证30片/11演员/159.333秒，演员不相交。类别为walking250/jogging150/turning95/gesture259/dancing79/jumping50。最多1000是上限；原HP选择规则、演员隔离及每演员配额下缺117片，不追加自定质量门或阻止本轮。原料SHA/name与转换SHA去重已核，跨PHUMA/BONES/AMASS的语义动作等价仍未知。[AP03清单](../data/manifests/supermimic_q1_AP03_soma_library_20261028.json)。

16env V01在零optimizer阶段发现BF16不同矩阵批量使不变策略的logprob最大差0.44027；保留失败。独立修为固定4096前向块（尾块补齐），保留同一网络/权重/归一化，概率计算FP32；V02的384转移/96样本反向通过，策略和logprob最大差0。正式4096预检98304转移/24576完整小批量反向通过，4025个人工截断、自然终止、动作单位、f+1奖励、重置历史和真实s1 bootstrap已实际覆盖，0optimizer，CUDA峰分配3.442GB/保留3.626GB，worker正常reap、GPU释放。[16收据](../data/manifests/supermimic_q1_RF12_preflight16_20261028.json)、[4096收据](../data/manifests/supermimic_q1_RF12_preflight4096_20261028.json)。

原生协议V01/V02均通过。V02另外保留16env活跃训练状态，完成384次零学习转移，再独立4env运行固定两走/抬手/一验证片，核训练backend的qpos/qvel/qacc_warmstart/ctrl/time、历史/回合累计和CPU/CUDA RNG未变；接触观察不改仿真状态，f+1连续。4片均完成，抬手腕峰0.253/0.128m仍远低于参考0.534/0.560m、上肢约1.299rad，仅复现瓶颈和核协议，不能当正式30验证基线。[协议收据](../data/manifests/supermimic_q1_RF12_native_protocol_20261028.json)。

9项自动门/预算单元测试通过，缺失/非有限指标不触发加训。RSL-RL使用当前安装实现（元数据3.0.1、实际split actor/critic API），不升级软件；实际Warp加载1.16.0，不按旧包元数据冒称1.14.0。前10个迭代仅critic，随后同一learner开放actor；保存优化器、冻结归一化、RNG、当前配方/累计实训和所有物理三重复轨迹。每阶段保存判据/PNG/PDF/CSV与固定初末MP4，新增媒体失败独立留记录，物理证据先落盘；终后统一交付。无WM/想象更新，标准图两项N/A。

交付smoke用明确标注的两行人工fixture检查六格/CSV的回合归一化与完成回合分母（0.6/0.5），加载真实330参考/8片固定suite，真实EGL渲染240x320非空画面；0新积分/0optimizer。人工fixture不作训练或效果证据，正式54秒媒体尚未制作。汇总见[准备回执](../data/manifests/supermimic_q1_RF12_preparation_20261028.json)。

旧临时权威副本出现数百历史文件及Git对象/config缺失，原因未确认。本阶段未在旧目录覆盖修复，已从指定origin Q1克隆至持久路径，独立核`c3404210b63e103987bd2e907d5ab8aeeb427b1b`并接回本轮新文件；旧来源/无关改动不动。此后提交只在持久副本进行。

## 可复现入口

在服务器唯一根内，现有Python记为`PY`，经`script/supermimic_q1_hp/activate.py`运行；必要时先核`--help`。已通过的预检输出目录不可覆盖重做。正式入口：

```sh
PY=/opt/data/private/ZHY/data/environments/supermimic_q1_hp_20261026/bin/python
"$PY" script/supermimic_q1_ppo/supervise.py --help
```

正式命令使用`--mode train --runtime data/references/supermimichp/TRACKING_C02_sampler_runtime_20261027 --library data/supermimic_for_q1/AP03_SOMA_Q1_20261027/q1_candidates/pilot_manifest.json --evaluation-library data/supermimic_for_q1/AP02_SOMA_Q1_20261027/q1_candidates/pilot_manifest.json --checkpoint data/supermimic_for_q1/BONES_RF11_SOMA300_SAMPLERONLY_E4096_MAX01H_20261027/run/final.pt --video-suite data/supermimic_for_q1/AP02_SOMA_Q1_20261027/demonstration_suite_V02.json`，`--output`用独立RF12新目录，并传已发布40位`--code-commit`、现场expected boot/GPU/OOM及`--detach`。启动前必须再次核现场与输出不存在，禁止覆盖或自动重试。日志在`logs/<RF12根名>_train.log`，controller及delivery另记同名前缀。

## 同步与下一步

资源准备初次发布`c3404210b63e103987bd2e907d5ab8aeeb427b1b`，正式PPO/AP03准备发布`684548c8e912802960a837924e5af7fd4c03248d`，均已push origin Q1并独立核40位远端。22文件865265字节在持久checkout/Mac主根/25386逐大小SHA相同，[代码同步回执](../data/manifests/supermimic_q1_RF12_code_sync_20261028.json)。后续本报告/小记录提交不改变冻结训练来源。

2026-10-28 01:02:02控制器49990/start_ticks3416720启动，worker49995/start_ticks3416816，boot/GPU与预检一致。启动前三端代码一致、无其他训练、共享6.73GB/OOM0、GPU15MiB/26度、盘余1.193TB；实际cgroup为v1，v2默认文件不存在不代表内存异常。全源1123项冻结，包括HP/PPO/RSL代码、kinematics、模型/参考/固定suite，01:07:38独立重核SHA全相同。[启动现场收据](../data/manifests/supermimic_q1_RF12_launch_observed_20261028.json)。

约01:04开始实际采集/学习。01:07:38三状态RUNNING，已完成42轮/175.025409实训秒/4128768真实转移，critic840/actor640次优化器步；前10轮actor未变断言已通过，std固定、alpha0/新增采样0。GPU9811MiB/55度、共享10.29GB/OOM0、余盘1.154TB、实验约359MB，均在保护内。正式学习已开始，准备阶段的0optimizer不再用于描述当前训练。

RF11重新同协议评测三重复完成：验证均0/30提前终止/100%完成，根XY0.12391至0.12929m、上肢0.93565至0.93774rad、相对腕0.12216至0.12257m。固定两走/抬手均完整，抬手上肢约1.2983rad、腕峰约0.2523至0.2529/0.1285至0.1287m，仍远低于参考0.53425/0.55992m；这些是RF11继承基线，不是RF12改善。物理不同重复允许浮点/接触差异；逐重复同共同前缀对照。

既有`q1-rf08-4096`已通过automation_update更新为RF12并实际回读ACTIVE、每10分钟；正常进度静默，异常/实质结果/交付完成通知。独立`supermimichp-x2`与`yuanqi`未改。下一步约3500实训秒检查`run/stage_decisions.json`和`stages/stage_01/`真实三重复/六格/固定54秒MP4，按冻结规则停止或连续进入B/C；仅故障外的获准阶段连续，不自动重启。
