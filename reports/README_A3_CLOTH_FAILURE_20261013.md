# A3已知失败行走加入训练的受限对照

记录日期：2026-10-13。实验E45 / R19。状态：实现完成、尚未登记或启动。官方035仍为基线，尚无匹配评测改善。

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

- `data/experiments/a3_zhy_cloth_failure_20261013_E45/`：登记/state/准备/效果；当前尚未创建。
- `data/training/a3_zhy_cloth_failure_20261013_E45/R19_001_s0/`：配置及分里程碑保存点。
- `logs/a3_zhy_cloth_failure_20261013_E45/`：各阶段console、5秒健康及实际训练合同。
- `data/manifests/a3_zhy_cloth_failure_20261013_E45.json`：运行后累计小清单。

## 实际验证、结果与边界

本地标准库5项控制通过，覆盖源分区不可变/新增角色/缺失或重复动作/不可拼点/原效果门；5个新文件静态编译和无张量plan通过。复核E44准备回执的原三段SHA及字段存在。尚未运行新机2项回调运行时控制、新增MotionLib转换或训练。

2026-10-13晚读取源机与ZHY Git均干净且同起点，ZHY boot为`d994e71f-bae0-46ba-a5f7-6c2163771c6c`，E44仍STOPPED，E45目录不存在。这不代替下一次启动预检。原E44第19optimizer教师RMSE0.050075越门、E42全程衣服+9.63%/5of6不倒及官方退化均保留拒绝；本轮尚不能改变这些结论。

## 保存与同步状态

小型代码/README/清单已在Mac保存，待审diff/暂存、payload审计、scoped提交推送main并核对远端，再同步新机。实际提交号由Git历史及执行回执记录。原始数据/模型/日志保持ZHY `LOCAL_ONLY`，网盘未核验。

## 下一步与恢复入口

先发布固定代码/记录。登记和每阶段启动都要求原三次5秒间隔快照：可用估计≥8GiB、无其他RSS>1GiB/计算、GPU空闲、可靠遥测、boot/OOM不变；不改记账、不清缓存、不终止他人进程。资源不足保持待启动并静默。运行保护CPU90/GPU85连续两次5秒、4GiB保留、900秒原线不变。

资源满足后一次登记/prepare，保存准备结果再一次train；任何STOPPED先归档真实原因和未执行分支，不复跑。首真实机器故障保存证据并暂停本工程heartbeat；教师质量保护不是机器故障。恢复先读本README、current state及实际state，不凭旧PID推断运行。
