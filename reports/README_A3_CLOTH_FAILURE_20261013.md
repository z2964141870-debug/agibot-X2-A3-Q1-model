# A3已知失败行走加入训练的受限对照

记录日期：2026-10-13。实验E45 / R19。状态：STOPPED；21:08:16–21:09:31训练，第19次optimizer触发原教师RMSE保护，结束语料分支。官方035仍为基线，尚无匹配评测改善。

## 目标与范围

验证将已知失败的zzx行走完整参考加入训练，能否在原能力保留约束内改善六段衣服开发动作。依据是[E43](README_A3_CLOTH_EFFECT_AUDIT_20261013.md)的匹配轨迹和[论文研究决策](README_A3_OUTCOME_RESEARCH_20261013.md)：该失败动作原本不在训练三段中。本轮不重复文献筛选。

仅在新ZHY工作根执行模型/张量/仿真；原hp3090仅代码、Git、标准库传输，Mac仅小型代码/记录。没有真机操作或驱动/功率/BIOS变更。

## 完成内容与关键决定

- 新增独立控制入口、开发角色/效果协议及保存回调，旧E23/E42/E44代码与账本保留。
- 主变量为训练语料3→4：保留zj walk、zj warmUp、fjw walk，加入`G_20251229_zzx/zzx_111505_walk`。新增动作明确`DEVELOPMENT_TRAIN`，其他六段开发效果比较仍完整保留；旧E23分区不改，不称未见验证。
- 原训练器`ClothCurveTrainer`不变：16环境、24步、seed0，FP32、缓存采样token、仅dyn decoder和Critic更新，原学习率/自适应规则、奖励、路由及原生重置分布不变。教师384帧、系数1、RMSE≤0.05/max≤0.5不放宽。
- 原三段MotionLib只校验SHA后复用，不重复转换或旧18项控制。新增一段300帧/30Hz，经原转换器生成后独立核验关节重排、根位置/旋转、保存再读及目录恰好4段。
- 新回调在通过保护的完整更新2/4/8/16/32保存，预算32更新/128optimizer/12288交互，最长900秒/阶段；不续训或自动恢复。首次STOPPED不执行其后verify/evaluation。
- 训练干净结束才独立核验并按2→4→8→16→32每点六衣服、四官方全程评测，最多50回放。复用E26/E02基线，不拼接各动作最优点。衣服零跌倒、均RMSE至少改善1%、原不倒衣服及各官方关节退化≤2%、世界根/腕≤5%全部保留。
- 若再次教师早停，结束语料分支，下一单独研究能力保留与优化冲突；不追加数据/轮数或扫权重。即使通过也只证明开发适配，仍需独立seed和真正未见会话。

## 版本、命令与证据

实现起点`486dc54abbbc96712e930d5708b38ec7d19b5738`。源码SHA、固定预算和输入参考SHA见[准备清单](../data/manifests/a3_zhy_cloth_failure_preparation_20261013_E45.json)。实际登记后以`registration.json`固定源码/输入SHA与启动版本；之后只能更新说明和小清单。

入口为`script/a3/cloth_failure_trial.py`，协议/回调为`cloth_failure_protocol.py`与`cloth_failure_callback.py`。在ZHY根目录使用系统Python控制，不在Mac或原机运行模型：

```bash
/usr/bin/python3 -m script.a3.cloth_failure_trial plan
/usr/bin/python3 -m script.a3.cloth_failure_trial register
/usr/bin/python3 -m script.a3.cloth_failure_trial run --stage prepare
/usr/bin/python3 -m script.a3.cloth_failure_trial run --stage train
```

只有前一步实际成功且状态匹配才能执行下一条；register仅在OUT不存在时一次执行。训练成功后才`verify-training`，随后依次`curve-2-cloth`、`curve-2-official`、`curve-2-verify`，按相同次序处理4/8/16/32。最后状态COMPLETE不自动代表效果通过，必须读`effect_verification.json`。

路径（ZHY根`/opt/data/private/hechengzhu/ZHY`）：

- `data/experiments/a3_zhy_cloth_failure_20261013_E45/`：登记/state/准备/效果；当前STOPPED。
- `data/training/a3_zhy_cloth_failure_20261013_E45/R19_001_s0/`：配置及分里程碑保存点。
- `logs/a3_zhy_cloth_failure_20261013_E45/`：各阶段console、5秒健康及实际训练合同。
- `data/manifests/a3_zhy_cloth_failure_20261013_E45.json`：运行后累计小清单。

## 实际验证、结果与边界

本地标准库5项控制通过，覆盖源分区不可变/新增角色/缺失或重复动作/不可拼点/原效果门；5个新文件静态编译和无张量plan通过。复核E44准备回执的原三段SHA及字段存在。随后ZHY准备阶段2项回调控制通过：2/4/8/16/32保存且只在32停，扩展预算被拒绝。新增zzx300帧/30Hz，关节最大误差5.94e-8rad、根1.13e-7m、旋转7.35e-8rad，全部≤1e-6；保存再读逐值相同，目录恰好4段。新增PKL 90,299bytes、SHA `82de6e17d9fc8466078e16069f239109dd2a8e9ef1c0878f32d9399532190a73`。随后一次训练结果见下。

2026-10-13晚读取源机与ZHY Git均干净且同起点，ZHY boot为`d994e71f-bae0-46ba-a5f7-6c2163771c6c`，E44仍STOPPED，登记前E45目录不存在。登记三次相隔约5.2秒可用均9.783GiB；可见最大RSS0.137GiB，GPU无计算任务。进程smaps读取遗漏经/proc/status复核为173僵尸及3休眠基础进程（RSS分别1736/1084/300KiB），没有因此遗漏>1GiB活跃计算；未改变任何记账方式。准备累计5健康样本同boot/OOM0，CPU最高41/GPU26°C，最低可用9.286GiB，准备进程采样RSS约0.845GiB；登记到准备有63.78秒采样间隔，不称全程连续监测。这不代替训练启动预检。原E44第19optimizer教师RMSE0.050075越门、E42全程衣服+9.63%/5of6不倒及官方退化均保留拒绝；本轮尚不能改变这些结论。

## 保存与同步状态

实现已以`e3156ce87f7d6697a79cde60dfefc2e512a88230`推送main并核对GitHub及ZHY同HEAD。实际登记源码即该提交，全部源码/输入SHA固定。准备结果见[累计清单](../data/manifests/a3_zhy_cloth_failure_20261013_E45.json)；本段结果保存后再scoped审计提交同步，提交号由Git历史及执行回执记录。原始数据/模型/日志保持ZHY `LOCAL_ONLY`，网盘未核验。

## 本次训练停止与对照结论

E45于北京时间21:08:16开始，21:09:31确认停止。完成4整次更新，加第5次的3个minibatch：19optimizer、1920交互。第18步teacher RMSE0.0491308533，第19步0.0507731512>原0.05；max abs0.23896718未超过0.5。console的明确异常为`Official action anchor drift budget exceeded; stop without retry`；控制state的reason=null仅表示监测器没有另加停止原因，不能解释为无异常。

| 原条件 | E44 / 三段 | E45 / 四段 |
| --- | --- | --- |
| 实际停止optimizer | 19 | 19 |
| 第8步teacher RMSE | 0.02823073 | 0.02633119 |
| 第16步teacher RMSE | 0.04571941 | 0.04605539 |
| 第19步teacher RMSE | 0.05007507 | 0.05077315 |
| 完整更新/交互 | 4 / 1920 | 4 / 1920 |
| 新闭环效果回放 | 0 | 0 |

首概率门action RMSE3.7312e-7、max2.3246e-6、ratio偏差1.1444e-5通过；19步冻结参数摘要不变。原生采样实际g1=1200/a3_fast=720，配置未改但有限样本计数不必与E44相同。训练日志明确MotionLib四key含zzx；没有逐动作采样计数，不能声称每段曝光相同。

保存step0/2/4，已只用标准库核验字节和SHA，**没有加载checkpoint进行独立训练核验或效果评测**。step2 SHA `0203565f1a1b916a154a0f0343e870be882b58e4e13c8ddab2cacf7a8ef4d0f5`；step4 SHA `3e679581fb1386e1fe718a1497af62615d8555a4bc03853e092a3b3b09edd645`。未到8/16/32；这些早期文件仅保留证据，按预定首STOPPED规则不继续本分支评测。

累计20健康样本同boot/OOM0、CPU最高41/GPU35°C、采样GPU功率最高117.02W，最低可用5,069,934,592bytes（4.72GiB），未触发4GiB保护。准备与训练阶段间最大间隔314.80秒，不称连续覆盖；停止后GPU无compute app、可见最大RSS约0.137GiB。没有确认新机器故障。

原始配置文本比较中algo不相等，已单独保存[差异解释](../data/manifests/a3_zhy_cloth_failure_config_20261013_E45.json)：仅trl.output_dir和logging_dir随E45/R19目录变化；扣除这两个存储字段后algo逐字相同，manager_env仅motion_file目录变化。旧回执的raw_algo=false和已登记核验代码原样保留，未为通过而改门；未来独立核验应事先显式区分存储路径与训练配方。

证据支持：将已知失败zzx加入语料，没有避开相同的原教师早停。**不能据此声称新增动作无益、必然存在相反梯度、锚点越门等同实际摔倒，或A3不能改善。**当前系数1的teacher MSE软惩罚不能保证所有更新都留在原约束内；其实现是在optimizer更新后测384帧再停。梯度相对方向/大小与Adam实际参数步尚未分解。

[停止小清单](../data/manifests/a3_zhy_cloth_failure_stop_20261013_E45.json)记录实际步数/路由/合同、配置和日志SHA、保存点身份。归档命令`/usr/bin/python3 -m script.a3.archive_cloth_failure`已执行一次，只读模型文件字节，写独立停止回执，不改原账本；已有回执拒绝重写。

## 下一步与恢复入口

E45语料分支关闭；禁止再次register/prepare/train、恢复step2/4或执行verify/evaluation。E44也继续STOPPED。原baseline与E42拒绝结果不变。本工程heartbeat保持ACTIVE，因为这是预定质量门停止，没有确认机器首故障。

下一只研究能力保留与优化冲突：先对照实际源码的PPO/辅助/teacher损失、梯度裁剪、Adam与自适应LR，以及已保存证据；明确一个可证伪的局部假设再登记新的独立计算，不增加数据、训练轮数或扫teacher权重。若需计算，仍三次5秒≥8GiB、无其他大进程/计算、GPU空闲、可靠遥测；CPU90/GPU85两次、4GiB、900秒原保护不变。旧实验文件不改，机器首故障才保存入口并暂停。
