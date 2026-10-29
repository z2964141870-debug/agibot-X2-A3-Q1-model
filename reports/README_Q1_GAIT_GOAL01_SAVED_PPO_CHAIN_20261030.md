# Q1保存PPO检查点与学习链路

日期：2026-10-30，北京时间。阶段：GAIT_GOAL01_SAVED_PPO_CHAIN。状态：保存张量和日志核查完成，普通步态目标未达。

## 目标与范围

接续[保存PD请求核查](README_Q1_GAIT_GOAL01_SAVED_PD_DEMAND_20261030.md)，检查PPO的小补偿是否真的经过策略更新，以及现有文件能否说明抬脚信号在哪个学习环节失效。只读取原initial、301秒、600秒三个检查点和113轮训练日志；不构造策略或环境，不调用forward、采样、积分、backward或optimizer.step。

本轮0新积分、0策略推理、0随机采样、0优化。底座301.1074845249823秒、PPO600.0709074319457秒，合计901.178391956928秒不变；两线physical_regression和失败证据保留。唯一V02授权已用，用户采纳null、goal ACTIVE；原监督、缓存维护和其他任务保持。

## 结果：actor确实更新，不能归因一直冻结

原RSL优化器按actor.parameters()再critic.parameters()构造参数序列。核对冻结的模型/分布源码、全部13个序列位置、张量名称/形状和Adam一二阶矩后，得到以下实际保存计数：

| 检查点 | 实训秒 | 轮数 | 六个head参数各自Adam步数 | 六个critic参数各自Adam步数 |
| --- | ---: | ---: | ---: | ---: |
| initial | 0 | 0 | 0 | 0 |
| stage_01 | 301.135292 | 58 | 960 | 1160 |
| stage_02 | 600.070907 | 113 | 2060 | 2260 |

每轮5个epoch、4个minibatch，共20次优化；前10轮只更新critic，随后actor解冻，恰好解释200步差额。三份检查点的全部张量和已有Adam矩均有限；训练后的398478个head参数及418049个critic参数都相对初始变化。固定分布参数std的14个元素保持0.1且没有Adam状态，是learn_std=False的预期行为。

因此现有证据排除了本轮“actor一直没解冻”或“只有日志计数增加、head根本未改”的解释。参数变化和优化器矩本身不能证明学到有效卸载/抬脚，也不能由actor/critic参数变化比例判断哪一个学坏。

## 结果：预热期也在注入补偿噪声

实际策略在训练时从latent高斯采样，再以0.15*tanh(latent)转换为14维腿腰位置残差；std全程固定0.1，entropy_coef=0。entropy_coef为0不关闭采样，也不会让固定std自动衰减。

最初10轮head的均值输出和更新KL均为0，但日志中的采样残差逐元素RMS为0.0148396至0.0148626rad。只对零均值固定高斯做80节点Gauss-Hermite积分得到0.0148534rad，与该量级一致；没有随机抽样或策略前向。最终一轮采样残差RMS为0.0158780rad。

这是训练动作分布的证据。此前固定普通走评测的确定性12腿补偿约0.0055rad，是另一组状态、不同通道汇总；不能用两者直接建立噪声导致失败的因果关系。采样残差也不是实际关节运动量。确定性评测与随机训练的差别本身不是PPO实现错误，但后续若研究探索强度，应把它作为独立变量。

## 结果：训练混合上肢条件，固定评测使用alpha.5

继承的reset_gate每回合约25%选alpha0，其余alpha.5；这里名为keep_baseline的分支只取消上肢参考混合，**腿腰残差仍然开启**，不能称它是训练期间的无head对照。

113轮alpha_sum在冻结的二值cap契约下对应alpha0转移占比23.73%-26.26%，平均alpha为0.36868至0.38136；固定评测始终alpha.5。PPO面对的是两种上肢控制条件的混合训练分布。该差异可能影响适配，但没有相同预算、仅改变混合比例的对照，尚不能判定为步态退步原因。

## 奖励、优势与更新的证据边界

所有日志的原奖励、站脚/摆脚/站稳bonus和两个cost能重建总奖励，差异仅为浮点归约误差。下面仅列训练首轮和末轮的每转移描述值，它们不是同状态、同相位或同回合条件的比较：

| 日志项 | 第1轮 | 第113轮 |
| --- | ---: | ---: |
| 总奖励 | 0.072272 | 0.081126 |
| 摆脚bonus | 0.006308 | 0.005554 |
| 站脚bonus | 0.007278 | 0.010761 |
| 平滑cost | 0.002713 | 0.001264 |

总奖励升高不能据此称换脚改善；摆脚bonus降低也不能单独称策略在放弃抬脚。采样状态、动作类别、终止和上肢条件都在变化，应以原固定物理评测的负面结论为准。

实际学习设置是gamma.99、lambda.95、整份rollout优势归一化、ratio clip.2、clipped value loss；actor与critic梯度各自按1裁剪。自适应KL目标.01，优化器前置hook把学习率限制在1e-5至5e-5。113个轮末记录和检查点均为5e-5；**没有保存全部minibatch学习率，不能声称每个更新内部都固定5e-5。** 下层构造器的fixed/1e-4被上层gait_adapter覆盖，不是本轮实际设置。

原日志KL是每轮第一时刻4096个观测上、整轮更新前后均值的比较；最终mean.00731009/p99.02155497，不是自适应调率使用的每minibatch旧/新策略KL。不能据此还原clip fraction、梯度被裁比例或更新是否过强。

24步采集末尾有critic bootstrap，不能把0.48秒叫作硬规划上限。自然片段结束和硬终止不bootstrap，人工时限用实际终态下一观测的V处理；全日志人工时限计数为0，故不能据本轮判断该分支对物理结果的影响。

检查点实际仅保存actor、critic、Adam和少量元数据；原训练rollout没有持久化。现有文件缺少逐转移的随机动作/奖励分量/接触相位、V/return/原始与归一化advantage、各minibatch ratio/clip fraction/KL和梯度范数。因此当前不能测出：

- 正确卸载后抬脚的转移是否得到稳定正优势。
- critic误差是否改变这些转移的优势符号或大小。
- 抬脚与tracking/平滑等项在actor梯度上如何相加，以及多少被裁剪。

保存Adam矩不能反推历史逐步优势或梯度；value loss从约.297到.499也不足以判critic失败。此前局部状态奖励梯度较强，不等于PPO实际采样得到了足够有用的动作优势。

## 验证、文件与发布

5项针对性测试通过，覆盖参数编号/预热计数、缺失或错计actor状态、非有限/负二阶矩、奖励重建/alpha混合/预热噪声及分布积分边界。真实检查对应39张量行、113训练轮、11108352条**原历史**采集转移；本轮新增采集0。

三个检查点与原checkpoint_selection历史SHA一致；plan、selection、training_evidence和receipt与原交付收据的大小/SHA一致。14份实际检查的学习链路源码在读取前后匹配原plan冻结身份；这只覆盖本轮选取的学习源码，不扩展此前未冻结仿真后端的历史身份结论。

独立工具script/supermimic_q1_gait_goal_analysis/saved_ppo_chain.py，SHA为053545b01a64e8c52df637be52184fa6265c72d9d1ca8392d4aeacc694883626；分析基础提交39eddb376574010c188d1555303c2f5c11cf02e9。原训练、奖励和控制代码未改。

输出data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_PPO_CHAIN_20261030内3工件72249字节：checkpoint_parameters.csv、training_iterations.csv、summary.json。权威工作树、Mac主根和25386逐大小/SHA已一致，两份工具代码也已三端核验。原权重/大NPZ留服务器，不重复媒体或保存轨迹审计。

[检查点逐张量表](</Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_PPO_CHAIN_20261030/checkpoint_parameters.csv>) / [113轮训练链路表](</Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_PPO_CHAIN_20261030/training_iterations.csv>) / [来源与边界](</Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_PPO_CHAIN_20261030/summary.json>)。

执行日志logs/session_records/q1_gait_goal_saved_ppo_chain_20261030.log保留成功结果和前置只读命令的解释器/loader/引号错误。使用已有activate.py修正入口，无安装、训练重启或原证据变更。以下为已执行命令；输出拒绝覆盖，无需重做：

```sh
chain_root=/opt/data/private/ZHY
env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=10s 120s \
  "$chain_root/data/environments/supermimic_q1_hp_20261026/bin/python" \
  "$chain_root/script/supermimic_q1_hp/activate.py" -u \
  "$chain_root/script/supermimic_q1_gait_goal_analysis/saved_ppo_chain.py" \
  --root "$chain_root" \
  --ppo "$chain_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_PPO_SOLE_Z_E4096_MAX20M_20261029" \
  --output "$chain_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_PPO_CHAIN_20261030" \
  --source-base-commit 39eddb376574010c188d1555303c2f5c11cf02e9
```

[本轮清单](../data/manifests/supermimic_q1_gait_goal_saved_ppo_chain_20261030.json)。Git提交、独立40位远端与记录镜像见后续publication旁收据；本报告不提前声称未核验的提交发布完成。

## 下一步与未解条件

两线都继续留在SuperMimic框架内。底座已显示跟踪/局部滑移改善与姿态代价并存；PPO已确认更新但未形成两走一致补偿。现在需要连接的是实际卸载/抬脚动作、对应优势和策略更新，不能仅由“再加奖励”或“半径没用满”跳到加训。

已有[逐子步观测设计](README_Q1_GAIT_GOAL01_SAVED_PD_DEMAND_20261030.md)可区分执行层和承重转换，仍是未授权、未执行的设计。若后续另获学习诊断授权，实际rollout还应记录相位/承重/alpha、奖励分量、V/return/原始和归一化advantage、latent均值/采样和minibatch裁剪与梯度信息，才能测策略梯度是否支持正确动作；本轮没有实施这类新训练。

原有限训练授权已消耗并按门结束。新物理、策略推理和训练均需用户新的明确决定；本阶段离线核查完成不结算整体目标，也不启动agent或改变监督任务。
