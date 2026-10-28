# Q1上肢参考控制与平衡耦合诊断

记录日期：2026-10-28，北京时间。阶段UR01：三条件、三重复真实物理诊断及本地图/视频交付完成。上肢局部修正有效，全身物理门未通过。用户指出连续多轮仍无解决方案，本次把“奖励/算法可能有问题”收窄到可操作的控制对照。0 optimizer，没有恢复RF12或启动下一训练，监督保持暂停。

## 已落实的修正

新增独立 `script/project_tools/q1_upper_reference.py`，保留RF11策略、归一化、80×10/590输入、噪声关闭的评测协议、PD、力矩限幅和原reward。只改8个上肢关节最终PD目标，身体/腿部仍使用冻结策略在实际闭环观测下的输出：

```text
q_policy = q_default + xi_joint_offset + action_scale * a_policy
q_target_upper = q_reference[f+1] + clip(q_policy_upper - q_reference[f+1], -radius, radius)
a_upper = (q_target_upper - q_default_upper - xi_joint_offset_upper) / action_scale_upper
```

两个事先固定的候选为radius=0和0.15rad；没有看验证结果后调参，也没有替换展示片。动作层全程正常物理积分，未把机器人关节或根直接写成参考。新模块位于独立诊断工具，不改RF11/RF12冻结runtime、训练脚本或旧权重。

radius=0表示参考前馈PD，radius=0.15表示保留被限制的小幅旧策略补偿。它不是已训练的新残差网络；“残差”仅指物理角度目标的剩余偏置。默认角、随机偏置、各关节动作尺度和下一参考帧均参与换算；简单把rad写进网络action是不正确的。

## 真实物理结果

同一个33环境批量：原30条演员隔离验证，加此前固定的两条普通走和一条抬手训练展示片。每个条件三次重复，按原生求解接触读取受力/滑移；不是4096训练，也不是重跑全部330条benchmark。三条件每次重新从各片首帧初始化；不声称浮点求解状态或随机张量逐位相同。

下表为三重复验证逐片完整有效前缀RMSE的均值；独立CSV保留各候选对应的RF11共同前缀，不能把较早终止造成的短前缀当作改善。共同前缀比值另见后表。

| 条件 | 上肢rad | 骨盆heading相对腕m | 根XY m | 腿rad | 验证失败/30 |
| --- | ---: | ---: | ---: | ---: | --- |
| 冻结RF11 | 0.93721 | 0.12238 | 0.12747 | 0.26549 | 0/0/0 |
| 上肢参考前馈 | 0.06971 | 0.05455 | 0.23162 | 0.28692 | 2/2/2 |
| 参考+限幅0.15rad补偿 | 0.16165 | 0.05386 | 0.19661 | 0.28958 | 1/1/1 |

| 共同前缀相对各自RF11 | 参考前馈（三次范围） | 限幅补偿（三次范围） |
| --- | ---: | ---: |
| 上肢RMSE | 0.07383–0.07432倍，约降92.6% | 0.17189–0.17233倍，约降82.8% |
| 相对腕RMSE | 0.44162–0.44269倍，约降55.8% | 0.43615–0.44227倍，约降56% |
| 根XY | 1.73592–1.89884倍 | 1.51457–1.56996倍 |
| 腿 | 1.07776–1.08371倍 | 1.08132–1.10638倍 |
| 指令变化 | 1.45590–1.49121倍 | 1.37452–1.45339倍 |
| 承重滑速 | 1.22519–1.23870倍 | 1.12964–1.22606倍 |

固定抬手参考双腕峰高为骨盆相对0.53425/0.55992m。RF11约0.252/0.129m；参考前馈约0.5596/0.5780m；限幅补偿约0.5379/0.5542m。**幅度接近并不代表动作成功**：参考前馈抬手只完成152/271/271步，限幅补偿179/267/128步（目标271）；RF11均完成271步。右腕峰值时序仍有偏差，根部明显滑移，不只看峰高数值。

两条固定普通走：参考前馈三次均完整320/210步；限幅补偿的stop片236/320/241步，start片均210步。不能说限幅方案已保住普通走，也不能挑参考前馈的成功抬手重复替换repeat0失败媒体。

## 收窄后的结论与解法

1. **上肢存在已实测的局部修法**：显式参考前馈能在原Q1模型/PD/力矩配置下把大部分关节和腕轨迹误差消掉。旧策略给错位置目标是可以直接干预的重要瓶颈；不能再把主要问题笼统归为Q1没有能力抬手或仅需放大PD。
2. **不能把局部修正直接部署成全身方案**：改变手臂后，继承的全身反馈控制出现根漂移、滑移和终止，两个候选均未通过原先的完整物理门。实际身体动力学扰动与网络的上肢输入分布变化同时存在，本次未隔离谁占主导；不能以此宣布唯一根因。
3. **后续候选应把上肢目标跟踪和下肢平衡适配分开**：上肢采用参考前馈，仅容许受限补偿；保留RF11为冻结教师与旧普通走对照，单独适配下肢/腰平衡分支，使它看到实际手臂运动并训练补偿。训练时混入外部上肢动作干预，保留旧任务行为约束；先检验手臂变化能否被下肢消化，再考虑扩库。此候选尚未训练，也没有新的训练预算。
4. RF12的PPO配置确实关闭了自适应KL（`adapter.py: schedule=fixed, desired_kl=None`），且未记录逐minibatch KL/clip比例。它说明缺乏策略变化的直接诊断，**不证明这是RF12唯一致崩因素**。后续任何策略微调应先记录策略偏移并设置更新约束，不能仅用finite-gradient检查代替保留已学控制能力。
5. 这次没有用新的reward口径冒充物理改善，没有再加扩散/LoRA网络，也未测新的reward方案。用明确动作目标绕过上肢偏置，比继续让整个网络自行发现目标映射更值得作为下一候选验证，但不能保证全身训练必成功。

作者官方HugWBC论文采用上肢动作干预和课程来训练下肢适应上肢运动，可为第3项提供方法依据：[作者PDF](https://hugwbc.github.io/resources/HugWBC.pdf)。这里只核验了论文中的机制；未引入其代码、未复现其机器人结果，也不把论文结论当作Q1成功证据。BeyondMimic等全身跟踪器仍是可参考基线，不能把RF12仅替换优化器称为完整复现它们。

## 实现与核验

- `q1_upper_reference.py`：独立参考目标修正。
- `test_q1_upper_reference.py`：3测试，覆盖非零默认/随机偏置、非均匀动作尺度、下肢输出保留、正负残差截断和非法参数。
- `diagnose_q1_upper_reference.py`：事先固定两候选/三重复；保存真实状态、目标、参考、原生接触、JSON和NPZ；运行前后策略及归一化指纹相同，0 optimizer。
- `deliver_q1_upper_reference.py`：只读保存轨迹，检查真正执行的目标残差界限，再生成图/CSV/视频；0额外物理积分/optimizer。

25386唯一根 `/opt/data/private/ZHY`，boot仍为4d859008-4c83-4a6d-814d-cbb9dfe885c9；GPU-26028785-9fb9-45af-e614-4cc53af15d08。使用原Python经activate加载CUDA/EGL。冻结RF11权重SHA为6c4de3171bd45359691a63ffe1712729c14fdb2080f8066e4c25632f4bbc5146，RF12冻结684548c保持历史不变。

物理诊断实际11:40:58至11:44:19，北京时间；controller82623/start_ticks7250354，worker82626/start_ticks7250376；全诊断COMPLETED，守卫reap返回0。独立30GiB共享cgroup、OOM基线0、100GiB盘余、12GiB实验、86°C、900秒无进展与25分钟总限保护保留。上述PID仅记录当时身份，不作实时存活断言。

首个目录Q1_UR01_REFERENCE_CTRL_DIAG_20261028因漏建run目录在创建环境前退出，日志/失败状态保留；补创建目录后在独立V02执行。不是训练失败重启或重置训练预算。

```sh
# 仅复现非训练诊断；output必须为新目录，不能覆盖已保存V02证据。
data/environments/supermimic_q1_hp_20261026/bin/python \
  script/supermimic_q1_hp/activate.py script/project_tools/diagnose_q1_upper_reference.py \
  --runtime /opt/data/private/ZHY/data/references/supermimichp/TRACKING_C02_sampler_runtime_20261027 \
  --library /opt/data/private/ZHY/data/supermimic_for_q1/AP02_SOMA_Q1_20261027/q1_candidates/pilot_manifest.json \
  --checkpoint /opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF11_SOMA300_SAMPLERONLY_E4096_MAX01H_20261027/run/final.pt \
  --output /opt/data/private/ZHY/data/supermimic_for_q1/NEW_NONTRAINING_DIAGNOSTIC_DIRECTORY
```

## 产物与同步

服务器根：`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_UR01_REFERENCE_CTRL_DIAG_V02_20261028`。
Mac小文件根：`/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_UR01_REFERENCE_CTRL_DIAG_V02_20261028`。

三重复9个JSON、总results、六格真实物理图PNG/PDF、逐重复共同前缀CSV、16.12秒四列对照MP4等18个小文件共6086423字节，Mac与25386逐大小/SHA一致。403帧Mac ffmpeg全帧解码返回0；图和抬手代表帧已目视。视频固定repeat0，包含两条普通走和抬手3个原速独立片段，保留提前终止hold标签；四列为RF11/参考前馈/限幅补偿/运动学参考，各相机分别跟随自己的根，不能从这种镜头代替量化根漂移。它是16.12秒诊断视频，不是新的训练54秒全类型交付。render仅保存qpos的FK，0额外物理积分。

轨迹/权重仍SERVER_LOCAL_ONLY，不宣称网盘备份。保存的实际PD目标残差最大值：参考前馈2.3842e-7rad、限幅补偿0.150000155rad，三重复均满足设置；运行前后策略及归一化指纹相同。交付后GPU15MiB/27°C、无计算进程。完整量值、判据边界、18文件SHA与403帧解码记录见[UR01小记录](../data/manifests/supermimic_q1_UR01_reference_control_20261028.json)。

本阶段在持久Q1副本审核/显式暂存/payload audit/commit/push，独立核远端后镜像Mac主根及25386；具体发布身份以Git历史和同步收据为准。RF12监督保持暂停，其他两个研究任务不动。没有新训练、扩库、删除或agent。
