# A3 humanplus zzx 专项专家与统一学生

记录日期：2026-10-18。D04独立制度，拟HP20/R26（登记前核对账本）。当前为实施与必要控制阶段，未登记、未启动新模型/仿真/optimizer；下文命令只是有前置条件的下一步骤，不能当作已执行。

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
