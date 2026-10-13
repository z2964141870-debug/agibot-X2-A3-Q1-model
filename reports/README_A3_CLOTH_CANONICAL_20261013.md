# A3官方std边界明确后的衣服微调E37

记录日期：2026-10-13，北京时间。实验A3_ZHY_CLOTH_CANONICAL_20261013_E37 / R14；15:30:17准备因MEMORY_RESERVE_PROTECTION停止，0优化/回放。

## 目标与修改范围

接续[E36](README_A3_CLOTH_CONSTRAINED_20261013.md)准备前置边界拒绝，仍执行同一受约束衣服解码器小微调。官方Actor.get_std会先将std限定到[.001,.5]，冻结摘要必须在该已知运行边界后建立；序列化权重原件保持。原暖启动载入时仍精确匹配原checkpoint；保存step0只允许std等于官方运行clamp，其他Actor/Critic字段精确相等。随后所有非g1_dyn Actor参数（包括规范化后的std）与buffer冻结、逐步核验。

这项边界不改变确定性mean，MuJoCo官方策略基线使用mean，复用E26/E02现有同条件完整基线；不重跑旧基线，不将该规则当A3零更新概率失配的根因。额外两个控制验证只改std/不改源张量、非有限std拒绝；与原三个控制合为5项保存回执。

## 固定预算与判据

沿用E36三段原train/30Hz/900总源帧，其他分区不进训练；2PPO更新/8optimizer、16环境、4mini-batch、1epoch、seed0、LR2e-6、FP32。冻结encoder/kin/std，只更新g1_dyn和Critic；官方aux与384帧固定锚点MSE系数1保留，逐步动作漂移RMSE<=.05/max<=.5。第一真实optimizer前ratio偏差<=.001、动作RMSE<=1e-5，独立CPU复算；全部原动态效果门和未来独立会话验证要求不变。完整参数/效果门详见E36固定协议。

每阶段900秒与CPU90/GPU85摄氏度两次5秒、4GiB保留、首异常停止、无自动恢复/复跑。只ZHY计算，旧E36保持STOPPED，官方仍基线。代码共享cloth_constrained_trial.py的参数化入口，独立cloth_canonical_trial.py绑定新目录/模块/清单，cloth_std_contract.py固定边界；E36旧源与旧SHA保存在97b3856提交。

## 版本、命令与产物

```bash
cd /opt/data/private/hechengzhu/ZHY
PYTHONPATH=script/vendor/sonic_for_a3:. data/environments/a3-sonic/bin/python -m script.a3.cloth_canonical_trial register
data/environments/a3-sonic/bin/python -m script.a3.cloth_canonical_trial run --stage prepare
data/environments/a3-sonic/bin/python -m script.a3.cloth_canonical_trial run --stage train
data/environments/a3-sonic/bin/python -m script.a3.cloth_canonical_trial run --stage verify-training
data/environments/a3-sonic/bin/python -m script.a3.cloth_canonical_trial run --stage evaluate-cloth
data/environments/a3-sonic/bin/python -m script.a3.cloth_canonical_trial run --stage evaluate-official
data/environments/a3-sonic/bin/python -m script.a3.cloth_canonical_trial run --stage verify-effect
```

登记后不可再register。data/experiments/a3_zhy_cloth_canonical_20261013_E37/、logs/a3_zhy_cloth_canonical_20261013_E37/、data/training/a3_zhy_cloth_canonical_20261013_E37/R14_001_s0/与data/manifests/a3_zhy_cloth_canonical_20261013_E37.json完全独立。原始工件LOCAL_ONLY，Mac只有小型代码/报告/清单。

## 实际结果与限制

登记后准备stage启动0.21秒即保护停止，returncode=-15。两个健康样本：登记时可用11,446,599,680字节，准备启动采样降至4,257,181,696字节(3.964GiB)，低于原4GiB线；同一采样容器RSS8,556,961,792字节、limit17,179,869,184字节、usage17,166,340,096字节、inactive-file可回收4,243,652,608字节。保护按既有容器估计触发，没有放宽。

两个样本同boot d994e71f-bae0-46ba-a5f7-6c2163771c6c，CPU最高40/GPU26摄氏度、GPU最高33.57W，OOM计数0，无新关键内核/遥测错误。45.46秒登记到启动间隔不是连续监测。准备console为0字节、MotionLib目录0文件、训练目录不存在；0训练/optimizer/闭环。未形成prepare验收回执，不能称5项新控制或真实std边界已经通过，也不据空console断言全部导入都未发生。

15:35:53另做一次只读恢复快照：可用13,028,638,720字节(12.13GiB)、RSS364,158,976字节，boot相同，CPU40/GPU26摄氏度、GPU利用率0，未见工程计算进程。官方checkpoint384MiB、锚点rollout7.5MiB、锚点输出844KiB、登记1.1MiB；文件大小不能直接解释或排除瞬时RSS峰值。缺故障瞬间按PID的RSS/PSS/阶段覆盖，不能归因于某个库、其他任务、训练算法或硬件重启；这次不是已确认boot变化/OOM/热停。

小清单data/manifests/a3_zhy_cloth_canonical_20261013_E37.json，SHA256 ace6aece55e8493ca5aa50cf1196018b76435be76776bf2c878d8512ea75781e；附加只读证据[资源停止清单](../data/manifests/a3_cloth_resource_stop_20261013_E37.json)。入口runner SHA256 a1ca578c6c942c6c0ef694103d1c3e62833beda36b94abf268d1cc1dd691b3a7；登记代码来源20005ed。原E36/E37登记/状态/健康日志不改、不复跑，未启动R14。

## 下一步与持续工作

保留受约束衣服微调代码和原概率/动态效果门。下一项先只读拆准备controller/worker的内存来源、导入依赖与容器记账，给新的入口增加按PID RSS/PSS和阶段证据，避免把资源门误当模型效果。新的计算必须另编号/固定预算，并在可靠遥测与资源余量证据齐全后才启动；不能因为停载后内存回升就自动复跑E37。

本次没有真实机器重启/OOM/热保护，长期工程heartbeat保持ACTIVE但下一步为只读资源定位，旧故障巡检/每日研究任务不改。本阶段README/索引/current state及小清单立即scoped审计提交/推送并核验远端；原始工件LOCAL_ONLY。官方仍基线，既未完成微调也未证明改善。
