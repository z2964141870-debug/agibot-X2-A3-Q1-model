# A3 humanplus zzx 专项专家与统一学生

记录日期：2026-10-18。D04/HP20/R26已单次登记，但19:25第一阶段isaac-baseline发生CUDA设备断言和新NVRM Xid43，STOPPED；独立审计verified=false、无自有组残留，zhy-a3已PAUSED。035已真实加载，专项训练/optimizer/最终效果均未开始。下面实施与命令保留为历史，禁止重跑HP20或把计划当完成。

## 目标与范围

遵循用户最新方向：先在完整失败动作zzx上证明一个专项跟踪专家优于035，再考虑统一学生与学生访问状态的纠正标签。专家是中间工具，不替换035，不按动作身份切换部署。HP19六衣服对035均RMSE恶化9.4723%、仍5/6不倒，四官方关节全部超门，原teacher通过不足以保证闭环保留。D01/D02/D03均已关闭，旧阶段及旧审计不复跑。

本轮是新的学习制度包，不宣称单一退化因果：训练集合缩至zzx完整300帧，恢复原完整g1_dyn decoder可训练，保留编码器/动作std冻结与原架构，去除仅为统一策略保留设计的teacher正则/绝对动作硬门及有限候选缩步，使用原生PPO+原辅助目标。取消约束仅限新专家；HP12等旧保护与终态不变，最终统一策略原验收门不改。

## 完成内容与设计依据

已核对真实HP19保存config：adaptive_sampling.enable=true、bin_size50、uniform_sampling_rate0.1、pre_failure_sample_window200，原已启用失败自适应采样，保持不动，不把“加课程/失败采样”作为新方法。原训练奖励包括anchor位置/朝向各0.5、相对身体位置/朝向及速度各1、局部VR五点3及动作平滑/接触/关节限制等项；这些不等价于关节RMSE/跌倒/世界根腕最终门。首轮保持奖励配方，保存原分项与完整轨迹，避免同时扫奖励权重。

保留fresh035 Actor+Critic/空Adam、seed0、16环境24步、FP32、原物理/reset、原KL自适应LR名义2e-6及范围1e-6至2e-5、原clip和原生a3_fast/G1配对[1,0,0,1]。更新时复用实际采样token，不重新采样token计算动作概率。专家可训练范围为原完整decoder，不加模块/改结构，不继承已淘汰PT。

独立代码humanplus_specialist_protocol/trainer/entry/trial/verification，独立freeze/audit。实现了原生空Adam精确所有权、Actor冻结与buffer检查、首batch概率与采样token存证、0/最终保存ID/矩/计数核验、原生Kit路径/实际settings/PID/清回调/退出0证据。旧trainer和HP19等行为保持。

## 固定预算与验收

固定256更新、1024原生Adam提交、98,304交互（256×16×24），不是收敛宣称。预算理由是从8更新筛查转为单动作可学习性验证，保留小并行规模和900秒硬期限；不根据中途效果追加次数/扫LR或权重。训练外层Actor/decoder上限8192=6144 rollout Actor+1024原训练Actor+1024实际token decoder，无teacher/候选额外forward。只保存0和256，只最终256评测；达到900秒或科学STOPPED不续跑/扩预算/评早期点。

新合同内顺序：Isaac035的zzx499步完整基线 → 训练 → 保存验证 → 同配置Isaac最终499步 → MuJoCo最终499步 → 独立汇总。HP11已有MuJoCo035基线不重放。Isaac评测保留终止标志但抑制自动物理重置，499步连续记录，不能用跌倒后的重置片段拼成绩；poststep物理状态与prestep参考相配，两个Isaac模型同口径，不能把其数值与MuJoCo直接等同。固定浮点/原生策略mean，记录根/身体/动作/关节及参考，原生正常关闭也须独立验收。

专家中间门：两模拟器均整段不倒、各自匹配035的整段关节RMSE至少改善1%；MuJoCo世界根/腕不超过035的+5%。这些门仅决定是否值得研究学生纠正，专家通过不是统一策略门通过，也不是未见泛化。学生阶段须另登记：在学生实际访问状态验证专家能恢复，分支不能偷偷改变学生物理或随机状态；可靠才采纠正标签、保留035原能力示范，最终同一策略完整十项原门。当前不启动蒸馏。

参考根平移输入限制保持：仅改根平移可使Actor1570输入不变，不能因此推断现有策略不能走或这就是跌倒原因。首轮不擅自加root输入/换架构；若专项仍失败，再按双模拟器证据检查目标、参考可行性、训练执行或输入局限，不称一种方法已不可能收敛。

## 论文依据与边界

2026-10-18只读核验BeyondMimic arXiv:2508.08241v4原文HTML及Atom元数据，首发2025-08-11、v4更新2025-11-13；原文描述多样跟踪策略、VAE用DAgger压缩、再用state-latent diffusion训练。这里只借鉴跟踪能力先获得、再整合的组织方式，未复现该论文，也不换成扩散模型/框架。原文入口https://arxiv.org/html/2508.08241v4。用户提到第三/六节；本次按v4实际方法内容核对，不假称旧版本节号相同。

DAgger核对PMLR原始摘要页面，AISTATS2011，出版元数据2011-06-14、627–635页；说明观察分布依赖已有动作及其在线学习归约，不保证本项目专家在学生偏离状态可靠。本次未复现理论/实验，入口https://proceedings.mlr.press/v15/ross11a.html。既有PredActor/MimicX等审查保留，不重筛模型或下载框架。

## 版本、控制与执行入口

唯一计算根humanplus@100.101.30.68:/home/humanplus/YUANQI_A3，固定vendor fe6868ba37034f89b912f0fb851bce19f120266d与只读x2-sonic-isaaclab环境。Mac仅小代码/README/清单。初始六项标准库边界控制：Mac五通过/真实vendor源不在本机一项跳过；humanplus六通过含固定vendor AST。登记前静态审查发现base_eval没有headless/seed/num_envs键，须用Hydra ++先构造再合并训练配置；增加第七项CLI组成控制。D04原始v1保留为已失效来源，未登记/运行；仅v2可用于后续新登记。控制不是模型或仿真验证。

完整实施审查后固定data/manifests/a3_humanplus_specialist_design_20261018_D04_v2.json，包含实际存在输入大小SHA、执行代码闭包、共享环境源码、HP11合同及原奖励源配置。只用035和完整zzx，旧必要E11/E12保留但不用于专家teacher，不继承被删PT。之后独立单次register；每阶段只有前阶段及独立审计成功才能执行。

```bash
# 仅当完整实施/控制/冻结已记录，且HP20/R26尚未占用
/usr/bin/python3 -B -m script.project_tools.freeze_humanplus_specialist
/usr/bin/python3 -B -m script.a3.humanplus_specialist_trial register
# 依序isaac-baseline/train/verify-training/isaac-final/mujoco/effect，各仅一次
/usr/bin/python3 -B -m script.a3.humanplus_specialist_trial run --stage isaac-baseline
# 每一阶段完成后单次独立审计，再进入下一阶段
/home/humanplus/anaconda3/envs/x2-sonic-isaaclab/bin/python -B -m script.project_tools.humanplus_specialist_audit --stage isaac-baseline
```

所有科学阶段保持humanplus_health+supervisor三次5秒启动门、可用>=8GiB/无其他计算/GPU空闲，运行>=4GiB、CPU90/GPU85连续两次5秒、900秒/阶段。cgroup v2/全boot journal保护不放松；首真实机器异常留证并暂停heartbeat，只终止自有组，不自动复跑。旧快照不是启动证据。

## 保存与下一步

S02清理已完成并审计，含S01累计释放4.01GB，HP19权重删除不再加载/恢复/重建。全部旧数组/配置/日志/指标保持；035仍baseline，无新改进模型。新模型和大数据仅LOCAL_ONLY，未网盘独立校验不称备份。

重要阶段README/索引/current state/清单审diff与audit_git_payload后scoped本地提交，GitHub已知SSHpublickey失败不重复push，LOCAL_COMMITTED_UNPUSHED。下一完成独立实施边界复核、冻结与登记；真实模型形状/梯度/模拟器及效果只能在登记后留证。

## HP20/R26独立登记完成

实施提交d63556a后单次register已完成，data/manifests/a3_humanplus_specialist_20261018_HP20_plan.json固定329源码/111实际输入/11环境源码、v2设计及256更新预算。0新模型/仿真/optimizer。下一supervisor实时三次5秒门后单次isaac-baseline；登记资源快照不能代替启动门。HP20/R26此前无占用，旧账本/STOPPED/清理均保持。

## 19:25 计算异常停止与恢复入口

登记提交7e4141c后单次isaac-baseline于2026-10-18T11:25:16.184033Z开始，supervisor先通过三次5秒启动门，native PID2970870、worker组2970867。Isaac创建了1环境，真实加载035，motion库加载单zzx/499帧/9.960秒。随后IndexKernel.cu:113出现index out of bounds/device-side assert，异步堆栈在policy.rollout的create_encoder_masks处报错，PhysX也报告CUDA错误。11:25:40左右内核新增NVRM Xid43，明确pid2970870/namepython，与本次native进程一致。

监督器在25.677323531秒以CRITICAL_KERNEL_EVENT终止自有组，HP20_result及state均STOPPED、stages空。未完成499步基线，没有完整轨迹或正常关闭收据；实际仿真步数未知，不填成0。没有执行专项训练、optimizer、训练保存验证或最终效果。不能把这一工程故障当作专家学不会或方法失败，也没有035改善证据。

单次独立审计已完成，verified=false、6健康同boot afb6796b-3883-4df4-99ea-bc381c52ea13、CPU最高76/GPU44°C、最低可用19732905984字节、0新增OOM、1条新增Xid43；after_issues仍CRITICAL_KERNEL_EVENT、自有组残留空。温度正常/组已清理不能证明GPU恢复或硬件损坏，未推断重启。审计SHA b8b9271f83b01539c4627f2084b56604b8b72f03bb0af0747e1d0e65a16750b6；原日志/内核收据/Kit/注册源码保持，不读删既有crash dump，不触碰共享环境/驱动权限。

恢复先读data/manifests/a3_humanplus_specialist_20261018_HP20_{plan,result,isaac-baseline_audit}.json与logs/a3_humanplus_specialist_20261018_HP20/isaac-baseline/{native_console.log,state.json,health.jsonl}及isaac-baseline_after_health.json。独立审计不再复跑，STOPPED后不加载PT或调用旧checked全产物链。zhy-a3应用已实际PAUSED，原30分钟设置保留；旧hp3090/yuanqi-a3及每日研究不变。

只读源码发现明确适配缺陷：motion_lib_base在1878行已用body_indexes筛选body_pos_w，get_body_pos_w直接返回筛选后数据；commands.body_pos_w沿用该数据。本次replay又用完整机器人body_indexes_data索引一次，属于重复索引，具有越界风险。CUDA报错异步且无同步定位，本轮不把它断言为唯一触发原因。后续独立桥接应直接用筛选后的参考并先按CPU名字/shape验证，动作29关节则按原EvalDebugDumper的action_term关节名/索引处理，不能直接假定motion全关节与动作29顺序相同。

下一仅保存及静态修复论证；原HP20源码/输入闭包不改。处理新内核事件的恢复门并固定独立新合同前不再科学计算，不换编号直接重跑、不豁免Xid/重建健康基准。机器首故障暂停要求已执行。用户路线仍为专项专家到统一学生，没有新增专家成果。

## 19:34 CPU桥接准备完成，科学计算仍暂停

新增独立humanplus_specialist_bridge.correspondence，仅用标准库按名称和shape生成各自源数组的索引，不导入torch/Isaac或加载权重。已选参考身体使用自身0..N-1索引，完整机器人身体另按名称映射；29个动作关节分别映射到显式参考关节名和机器人关节名，排除被动关节并拒绝名称缺失、歧义或shape不符。三项测试覆盖3个已选参考身体对9个完整身体、参考31关节含被动/反向顺序、无效元数据在索引前拒绝。Mac与humanplus各3项通过；没有将这些合成数组控制当作真实模型或数值验收。

准备清单data/manifests/a3_humanplus_specialist_bridge_20261018_B01.json固定两新增源码的大小SHA及HP20三份既有JSON收据SHA。已核对HP20五个注册模块对登记提交7e4141c无diff，未改写旧行为或重跑旧审计。新桥接尚未接入任何执行入口，未登记新科学阶段；下一独立入口必须以原生command/robot/action_term元数据验证参考与机器人两套顺序，再取值。当前测试不能证明真实关节名来源正确、消除全部CUDA越界或GPU恢复，也不证明本次重复索引是唯一触发点。

本阶段0新张量反序列化/模型/仿真/optimizer，无实时GPU恢复验收。zhy-a3保持PAUSED/原30分钟设置，HP20 STOPPED和未知实际步数保持。恢复需先明确本次Xid43的处理及可核验恢复条件，再在独立合同中接入桥接与必要同步定位；不能仅因bridge控制通过就重启。后续路线仍是专项完整能力先验证、可靠后才统一学生，不复跑已拒绝配方。

复现CPU控制：`/usr/bin/python3 -B -m unittest script.a3.test_humanplus_specialist_bridge -v`。代码/记录经payload审查后scoped本地提交，GitHub仍LOCAL_COMMITTED_UNPUSHED，不重复已知失败push。没有新增权重或存储清理，S01/S02累计4.01GB释放与官方035/必要输入保护保持。
