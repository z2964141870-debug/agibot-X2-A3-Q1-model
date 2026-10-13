# A3衣服微调资源入口E38/R15

记录日期：2026-10-13，北京时间。状态：E38/R15已登记，16:27:09准备通过，实际训练尚未启动。

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

代码提交a69be1a已推送核验并同步ZHY。系统Python入口实际未导入tensor/Trainer；单次只读进程RSS17,387,520字节。登记前3次被动快照通过，登记后一次准备21.29秒成功，5项冻结/概率/std与8项checkpoint故障控制全部通过。三段各300帧/30Hz，关节最大误差9.191e-8rad、根1.748e-7m、旋转7.485e-8rad，PKL重载完全相同；官方29维std中4项按官方运行clamp规范化，最大0.5000176429748535到0.5，其他模型数值未优化。

准备6个监督健康样本、11个阶段资源事件，同boot d994e71f-bae0-46ba-a5f7-6c2163771c6c、OOM0、CPU40/GPU26摄氏度，GPU最高33.54W。阶段事件最低容器可用10,976,686,080字节(10.22GiB)，prepare工作进程checkpoint载入后RSS最高1,302,716,416字节、PSS1,297,114,112字节，controller未导入torch。5秒采样不能保证捕获每个瞬时峰值，VmHWM另存；不能据本次未重现推定E37根因。全容器进程扫描记录PermissionError/ProcessLookupError，已抽查一条后者为仍留PID的已退出omni.telemetry僵尸；其他进程PSS缺失保留，不能称全进程PSS完整。自己的controller/worker阶段PSS及容器/温度/GPU/内核遥测有效。

实际optimizer与新闭环仍0，没有新权重或提升。[准备小清单](../data/manifests/a3_zhy_cloth_resource_20261013_E38.json)保存测试回执/输入代码SHA/健康和采样边界，原始日志在ZHY。下一依固定协议启动一次两更新训练，独立核验冻结与第一实际batch概率，再进行六衣服/四官方匹配对照。拒绝不复跑、不降门槛；E37峰值原因仍未知。
