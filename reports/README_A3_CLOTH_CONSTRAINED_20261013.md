# A3衣服动作受约束解码器微调E36

记录日期：2026-10-13，北京时间。实验A3_ZHY_CLOTH_CONSTRAINED_20261013_E36 / R13；已登记，尚未执行准备/训练。

## 目标与选择依据

继续衣服/SMPL完整跟随、取得官方035同条件可验证提升。E23已有三段训练分区合格参考；E26衣服开发闭环5/6不倒，zzx从4.66秒有腿部持续偏离；E35补全被动初始化仍倒且均误差+5.93%，结束该改善分支。E12固定状态证明encoder和decoder均可独立引起动作漂移，旧R12全模型短微调退化，不能继续用保存点/更新次数当改善。

本轮唯一主方案为受约束的衣服解码器适配：只允许actor_module.decoders.g1_dyn参数更新，Actor其他参数/所有buffer（含编码器、运动学解码器、std）冻结；价值网络可学习新分布。保留现有官方PPO和辅助目标，加入固定官方动作锚点MSE，限制每次更新的漂移。数据、更新范围与约束共同构成本轮方案，不声称单独隔离了冻结的因果效果。

## 固定输入、预算与判据

训练只使用E23原train分区三段300帧/30Hz：G_0113_zj_train/183926_zj_walk、同会话192333_zj_warmUp、G_20251226_fjw/fjw_140008_walk。通过官方CSV-to-MotionLib转换并以名称核对29关节/root/quaternion，原质量门不放宽。仅运动学合格，不等于动力学平衡或实时衣服输入合格。其他三段合格参考只参与开发评测；原七段均已反复开发，不能冒充未见测试。

冻结锚点是E11官方采样384步输入/E12独立核验的初始均动作，不新采集旧条件。暖启动动作最大差<=1e-5；第一真实小批在任何optimizer前检查RMSE<=1e-5、ratio偏差<=0.001，保存完整字段供CPU独立核验。冻结参数及buffer逐步检查；锚点MSE系数1，每次实际更新检查384帧动作RMSE<=0.05、最大差<=0.5。这只是固定开发状态的能力约束，不保证全分布保留。

一次两轮PPO/8optimizer，16环境、1epoch、4mini-batch、seed0，LR初始2e-6、adaptive范围1e-6至2e-5，bf16/fp16/TF32全关。原已验证完整精度、官方aux和实际LR审计入口复用，不修改vendor。三个有针对性的CPU控制验证冻结优化器所有权、实际非零步仍冻结、坏/全mask概率拒绝。每阶段900秒；CPU90/GPU85摄氏度连续两次5秒、保留4GiB内存、首机器/数值异常停该分支，无开机恢复/自动重试。

登记文件固定所有代码/input SHA、boot和OOM基线。计算仅/opt/data/private/hechengzhu/ZHY既有a3-sonic环境；原hp3090只Git/标准库传输，Mac只小型代码/README/清单。

训练独立核验三保存点0/1/2、8真实optimizer步、官方step0精确相等、后续仅dyn Actor变化、FP32与概率合同。合格后候选六衣服开发动作逐段499步，匹配E26原初始化与原门，不采用被拒的E34初始化；另四旧官方开发诊断动作匹配E02基线。无重跑官方基线。效果门：六衣服0跌倒/均关节误差至少改善1%，五段原不倒各允许至多2%关节退化；四官方各不新增跌倒/至多2%关节退化；各动作根/手腕世界误差至多5%退化。通过也需另选从未进入开发的独立会话再验收，不部署、不真机。

## 版本与可复现入口

代码script/a3/cloth_constrained_trainer.py、cloth_constrained_trial.py、test_cloth_constrained.py。从官方035精确暖启动，不接旧R12/R13账本。严格状态机，不覆盖既有目录/重置编号。

```bash
cd /opt/data/private/hechengzhu/ZHY
PYTHONPATH=script/vendor/sonic_for_a3:. data/environments/a3-sonic/bin/python -m script.a3.cloth_constrained_trial register
data/environments/a3-sonic/bin/python -m script.a3.cloth_constrained_trial run --stage prepare
data/environments/a3-sonic/bin/python -m script.a3.cloth_constrained_trial run --stage train
data/environments/a3-sonic/bin/python -m script.a3.cloth_constrained_trial run --stage verify-training
data/environments/a3-sonic/bin/python -m script.a3.cloth_constrained_trial run --stage evaluate-cloth
data/environments/a3-sonic/bin/python -m script.a3.cloth_constrained_trial run --stage evaluate-official
data/environments/a3-sonic/bin/python -m script.a3.cloth_constrained_trial run --stage verify-effect
```

register已完成，恢复时不得重复。仅按state.json允许的下一阶段执行。data/experiments/a3_zhy_cloth_constrained_20261013_E36/保存协议/分区/验收，data/training/a3_zhy_cloth_constrained_20261013_E36/R13_001_s0/保存训练，logs/a3_zhy_cloth_constrained_20261013_E36/保存分阶段控制台与health；小清单data/manifests/a3_zhy_cloth_constrained_20261013_E36.json由各阶段生成。

## 当前结果与保存

语法检查通过；登记成功。准备控制、真实优化、闭环效果及未见泛化尚未执行，不能称改善。README/索引/current state与代码先scoped审计/发布，同步证据由Git历史和远端核验提供。原始模型/张量/日志LOCAL_ONLY，未验证网盘。

## 下一步

运行一次prepare；数据/冻结控制通过后立即两轮训练及同条件效果评测。若合同/效果失败，保存失败与具体限制，保留官方基线；不延长同一次试验或改质量门。
