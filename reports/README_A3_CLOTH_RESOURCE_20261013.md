# A3衣服微调资源入口E38/R15

记录日期：2026-10-13，北京时间。状态：代码已修改，尚未登记或运行E38。

## 目标与范围

继续衣服/SMPL完整动作跟随和官方035同条件改善。E37已因4GiB保留保护停止，0优化；不能复跑或归因其瞬时8.56GB容器RSS。新E38只修改准备/监督资源入口，沿用E37的官方std运行边界及E36受约束衣服配方。旧账本保留，原hp3090只Git/传输，计算仅ZHY。

## 修改及依据

静态源码确认checkpoint_store的原顶层torch导入进入controller依赖链；train包装worker只为三项FP32配置先导入完整Trainer，再启动训练子进程。前者改为仅checkpoint数值操作局部导入，后者改为相同字面配置。模型、优化器和动态质量门不变；移除额外依赖不构成E37峰值根因证明。

stage_resources记录可见进程PID/PPID/comm/RSS/PSS/VmHWM和容器cache/rss/active-file/inactive-file，不读取或保存argv/环境。5秒温度/内存/内核监督及准备导入、控制测试、官方checkpoint载入、CSV转换边界均fsync。controller用系统Python且拒绝已导入torch；实际worker使用既有a3-sonic环境。

## 固定预算与判据

登记前3次间隔5秒健康快照、至少8GiB可用起始余量、无>1GiB已存活进程、GPU空闲、boot/OOM与E37一致。准备进程RSS额外上限3GiB；原4GiB保留线、CPU90/GPU85两次5秒、阶段900秒、首故障停止不放宽。准备验收5项衣服/std控制、8项checkpoint故障控制、三段train的900帧MotionLib ABI与重载，其他分区不进训练。

合格后仅2PPO更新/8optimizer、16env、4mini-batch/1epoch、seed0、LR2e-6/FP32。只g1_dyn及Critic可更新，encoder/kin/std/Actor buffer冻结，384帧官方teacher约束与第一真实batch概率门保留。六衣服/四官方开发对照复用E26/E02基线：衣服0跌倒/均误差改善>=1%、原不倒单段和官方误差<=+2%、根/手腕世界误差<=+5%。通过开发仍须真正未见会话，禁止部署。完整协议见[E36](README_A3_CLOTH_CONSTRAINED_20261013.md)、[E37](README_A3_CLOTH_CANONICAL_20261013.md)。

## 版本、命令和工件

```bash
cd /opt/data/private/hechengzhu/ZHY
/usr/bin/python3 -m script.a3.cloth_resource_trial register
/usr/bin/python3 -m script.a3.cloth_resource_trial run --stage prepare
/usr/bin/python3 -m script.a3.cloth_resource_trial run --stage train
/usr/bin/python3 -m script.a3.cloth_resource_trial run --stage verify-training
/usr/bin/python3 -m script.a3.cloth_resource_trial run --stage evaluate-cloth
/usr/bin/python3 -m script.a3.cloth_resource_trial run --stage evaluate-official
/usr/bin/python3 -m script.a3.cloth_resource_trial run --stage verify-effect
```

data/experiments/a3_zhy_cloth_resource_20261013_E38/及logs/a3_zhy_cloth_resource_20261013_E38/独立，模型data/training/a3_zhy_cloth_resource_20261013_E38/R15_001_s0/，小清单data/manifests/a3_zhy_cloth_resource_20261013_E38.json。禁止重置或恢复。原始工件LOCAL_ONLY，Mac仅小型代码/README/清单。源码/输入SHA登记后固定；先审查diff与payload审计、提交推送核验再启动。

## 结果与下一步

当前只有只读现状和源码依据，E38尚未测试，未取得新权重或提升。先验系统Python入口无tensor导入，再固定登记和一次准备；通过后顺序推进训练、独立校验和匹配效果评测。任何拒绝及时记录，不以降门槛换通过。E37峰值缺按PID证据，保持原因未知。
