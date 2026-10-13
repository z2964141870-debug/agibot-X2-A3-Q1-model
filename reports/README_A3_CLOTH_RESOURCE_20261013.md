# A3衣服微调资源入口E38/R15

记录日期：2026-10-13，北京时间。状态：准备通过；16:45:18训练因第一实际batch概率合同拒绝STOPPED，0optimizer，不复跑。

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

16:44:24–16:45:18的一次train运行53.53秒，实际完成环境/模型初始化、官方权重暖启动、step0保存和24步采样，但第一实际PPO batch在optimizer前拒绝。有效帧动作RMSE0.0006889391806907952、最大差0.017822980880737305、1帧动作差>1e-4；ratio0.78307044506073–1.0000152587890625、最大偏差0.21692955493927002，原RMSE1e-5/ratio0.001门保持。不存在model_step_1/2、actual_optimizer_steps日志，cloth_contract.optimizer_steps空；实际0optimizer/新效果回放，不能把step0称微调模型或宣称冻结解决了一致性。

暖启动回执Actor/Critic与官方完全相同、optimizer空；384帧teacher动作最大差0。step0为134,048,471字节、SHA594b33a7fde0ca47cf135c5421f37c8c581ae905b8e372e47386cb516aaa1394，仍待独立数值核验。保存的precision回执覆盖24次rollout/0次training记录；实际training hook源码会检查FP32，但失败后尚未flush其记录，不能称已保存第一training精度回执。当前代码在概率断言成功之后才保存first batch，故此次拒绝没有原始batch文件，这是诊断证据缺口。

prepare/train累计17健康样本同boot/OOM0、CPU最高41/GPU34摄氏度、GPU最高114.85W，可用最低7,333,621,760字节(6.83GiB)，无关键内核/遥测错误。阶段间1035.20秒非连续覆盖；实际train进程采样RSS最高5,875,990,528字节，VmHWM5,954,699,264字节，未触发资源保护。PSS扫描权限缺失当前抽查为pod/ssh-agent/gpg-agent，RSS均<2MiB；已退出僵尸另记，不宣称全进程PSS完整。不能据此认定硬件故障或E37峰值来源。

[E38停止小清单](../data/manifests/a3_zhy_cloth_resource_20261013_E38.json)保存输入代码SHA、准备回执及健康/资源，原始console与step0在ZHY、LOCAL_ONLY。原E38保留STOPPED，不复跑或放宽门。下一独立零优化诊断：在任何概率断言之前持久保存采样输入/mean/token/latent和第一个训练batch，比较输入写入、编码器离散边界及实际执行路径。尚未定位单帧差异，官方仍基线，未来动态效果与未见会话仍待。
