# A3 完整精度修正后的两更新与效果诊断

记录日期：2026-10-12。独立实验 A3_ZHY_FP32_20261012_E10 / R12。状态：两更新/8optimizer/3保存点与4完整回放独立核验完成；计算一致性修正通过，策略效果门槛失败。

## 目标与依据

E09真实四batch概率合同通过，证明显式关闭默认BF16解决本次采样/训练精度偏差，见[修正验收](README_A3_ZHY_CONSISTENCY_20261012.md)。本阶段验证修正入口能否完成真实反向/优化器/保存，再区分计算修复和策略效果。

新机 `/opt/data/private/hechengzhu/ZHY`，原hp3090不计算；用户已授权继续ZHY探索与微调。只开独立R12，不改旧R11或停止账本，无开机恢复、重复启动或长训练。

## 固定方案与实现

入口 `script/a3/zhy_fp32_trial.py`，训练器 `script.a3.consistent_precision.ConsistentPrecisionTrainer`，复用既有官方辅助loss、VerifiedFineTuneTrainer优化器审计和DurableModelSaveCallback；既有CPU核验器仅配置到新目录，不修改历史实验。

官方035初始化，seed0、16环境、4minibatches、epochs1、LR尺度0.1（actor2e-6、adaptive1e-6–2e-5）、2次更新，总8个优化器步；初始/更新1/更新2分别保存。明确algo.trl.bf16/fp16/tf32=false、ACCELERATE_MIXED_PRECISION=no，真实Actor持续检查FP32。独立CPU重载验证大小/SHA、网络和优化器有限性、初始权重、更新计数/真实LR及辅助loss日志。

保护沿用CPU90/GPU85摄氏度连续两次5秒、内存保留4GiB、单阶段900秒、同boot、OOM/关键内核错误/遥测缺失停止；需GPU空闲、有效E09和官方模型SHA，任一计算故障不自动重复。注册固定代码、模型、基线及动作SHA。

两更新核验后，只回放固定split中4个既有diagnostic动作，全长度/参考30fps/frame_stride4/oracle参考/no delay，与E02在同新机同boot核验的官方035对应结果比较；不复跑官方20动作基线。沿用平均关节误差至少改善1%、逐动作不变差、无新增跌倒、root/body/action delta退化不超过5%门槛。指标从原始timeseries独立复算，计算完成不等于策略通过。

## 版本、命令与产物

基于main `4c3fb2eb4f6013406584a97c8a107fb512d548cf`，固定vendor `fe6868ba37034f89b912f0fb851bce19f120266d`；代码/合同在启动前scoped审计推送并核验远端。

```bash
cd /opt/data/private/hechengzhu/ZHY
data/environments/a3-sonic/bin/python -m unittest script.a3.test_zhy_trial -v
data/environments/a3-sonic/bin/python -m script.a3.zhy_fp32_trial register
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_fp32_trial run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_fp32_trial verify
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_fp32_trial evaluate
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_fp32_trial verify_evaluation
```

训练 `data/training/a3_zhy_fp32_20261012_E10/R12_001_s0/`；注册/状态/诊断输出 `data/experiments/a3_zhy_fp32_20261012_E10/`；日志 `logs/a3_zhy_fp32_20261012_E10/`；小型清单 `data/manifests/a3_zhy_fp32_20261012_E10.json`。张量/模型/原始日志留新机LOCAL_ONLY，Mac/Git只留小型代码与README/清单，不保存凭据。

## 验证、边界与下一步

新执行器静态编译与6项CPU保护检查通过（0.004秒），包括容器内存估计、旧路径拒绝、无重启恢复及遥测失败终止子进程。实际北京时间17:07:03–17:08:06完成，独立CPU重载/大小SHA/有限性/初始权重/优化器计数/LR与官方辅助loss核验通过，状态TRAINING_VERIFIED。0/1/2保存点优化器计数分别0/4/8，真实优化器审计恰8步；初始/末次LR2e-6，实际8步为3e-6、3e-6、3e-6、2e-6、3e-6、4.5e-6、3e-6、2e-6，属于已审计的KL自适应变化（E11复核更正此前“实际LR2e-6”概括，原日志不变）。精度保护确认48次采样和8次训练Actor调用均完整FP32。

| 保存点 | 大小（字节） | SHA-256 |
| --- | --- | --- |
| 初始 | 134051287 | `172e622dbdc1667657ef020260e448017da54ddd39a4bf195dfd92d256527ced` |
| 更新1 | 402076373 | `71d2f3e0c5efbe441a1b754ad93f3b24058fdc6a533d5fe6be04d8d62c5405a2` |
| 更新2 | 402076437 | `62e7db387e465abdb758d704f1ae0346a2ffb14304db4d3767cc46b2719f65cd` |

13健康样本同boot `d994e71f-bae0-46ba-a5f7-6c2163771c6c`，CPU最高44/GPU34摄氏度、采样功率最高132.38W、可用估计最低5.74GiB、OOM0，无新关键内核错误/遥测缺失；85.402秒最大间隔位于注册至启动之间。运行版本 `4d6913a39c740fbf5099e498c093f761296e056f`，模型/张量/原始日志LOCAL_ONLY。

上述训练核验阶段策略效果曾记NOT_EVALUATED；随后完成下述4动作全长诊断。

## 效果结果与结论

北京时间17:15:20–17:17:20完成4动作全长度回放，共5775策略步，4/4无跌倒；原始关节、root/body轨迹和动作SHA/逐步覆盖由CPU独立复算，状态VERIFIED。官方基线复用已核验E02，同新机同boot、同动作SHA/参考帧率/stride/全长度/无延迟，未重复计算官方20动作。

| 既有诊断动作 | 官方关节RMSE（rad） | R12关节RMSE（rad） | 相对误差变化 |
| --- | --- | --- | --- |
| 001 前向慢走 | 0.065687 | 0.069632 | +6.01% |
| 029 原地慢转 | 0.076654 | 0.078635 | +2.59% |
| 048 右单腿平衡 | 0.045770 | 0.044923 | -1.85% |
| 050 原地慢踏步 | 0.043070 | 0.048683 | +13.03% |

四动作等权平均关节RMSE从0.057795变为0.060468rad，误差增加4.6255%；3/4动作关节退化。001 root位置误差+26.87%，048/050根朝向误差+18.82%/+29.84%，部分anchor-local身体误差超过5%上限。**预登记效果门槛失败，不升级R12或替换官方模型。** 零跌倒和优化器成功不作为效果通过依据。

全阶段39健康样本同boot、CPU最高46/GPU35摄氏度、采样功率最高132.38W、可用估计最低5.74GiB、OOM0，无新关键内核错误/遥测缺失；最长441.431秒间隔位于训练核验/记录与评测之间，不能称为全时连续覆盖。计算阶段使用5秒保护，任务已结束，随后只读GPU查询无计算进程。完整证据与保存点/指标SHA见[小型清单](../data/manifests/a3_zhy_fp32_20261012_E10.json)，原始工件新机LOCAL_ONLY。

E08/E09解释并修复本次初始概率偏差：未提前优化Actor主体权重，真实训练隐式BF16造成采样FP32/训练BF16两条计算路径。R12证明修正后仍可执行反向与优化器；但本次效果结果表明，修正精度本身不足以解释或消除全部微调退化。不可将R12这4个动作与旧R11的20动作平均值直接比较，也不能从一次短训推出每次更新必改善。

下一项应在新FP32入口上诊断真实损失/梯度、更新哪些模块及与期望跟随指标的关系，再选单一变量试验；优先保护官方已有能力，明确微调要改善的真实缺陷。当前不扩大训练、不恢复原机，官方仍为基线。

四动作已反复用于诊断，且oracle使用未来参考，不是新鲜最终测试、真实遥操或机器人验收。局部门槛本次未通过，官方基线保留；零更新ratio一致不保证PPO每轮效果单调提高或长训稳定。结果已更新本README/索引/current state/清单，按scoped审计提交推送main核验，保留失败结论；本轮不追加长训练或策略升级。
