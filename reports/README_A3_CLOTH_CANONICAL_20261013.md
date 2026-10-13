# A3官方std边界明确后的衣服微调E37

记录日期：2026-10-13，北京时间。实验A3_ZHY_CLOTH_CANONICAL_20261013_E37 / R14；固定协议，待登记/执行。

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

## 当前结果与下一步

旧E36失败已记录；E37尚未准备/训练，不能声称任何优化效果。先scoped审计/推送，再新登记与一次准备；合格后两轮训练、六衣服/四官方开发对照和独立结果验收。失败不改账本/门槛；通过开发门仍不部署、须真正未见会话验证。
