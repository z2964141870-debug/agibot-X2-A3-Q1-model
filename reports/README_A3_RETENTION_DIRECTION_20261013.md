# A3保存状态的能力保留梯度方向诊断 E46

记录日期：2026-10-13。状态：实现和标准库控制完成，尚未登记/运行。官方035仍baseline，没有新效果提升。本实验独立于已STOPPED的E44/E45；禁止恢复其训练或效果评测。

## 目标与范围

继[保留机制审查](README_A3_RETENTION_CONFLICT_20261013.md)，回答固定问题：E45 step4状态在**保存首衣服batch及实际采样token**上，PPO下降方向是否一阶增加完整384帧teacher MSE？step0只做初始合同控制。本实验是固定输入局部反事实，不是第17–19次optimizer历史重演，不是效果选点。

仅在ZHY既有环境计算；Mac只小型代码/报告，hp3090只Git与标准库传输。不创建环境/仿真/Critic/optimizer；不写可部署checkpoint。

## 完成内容与关键决定

- 新增独立控制器、离线probe、预算/方向协议和3个标准库控制测试；原E44/E45代码、账本不修改。
- 固定step0/4、E45保存首batch、原E11 teacher输入与E12 targets，注册时逐个SHA核验。E39运行metadata只提供已解析观测维度，已静态核验其完整algo_config与解析后的E45配置一致，运行时再次断言。
- 每状态1次原Actor辅助forward、1次固定token decoder forward、4次teacher forward，计划共12次，注册硬上限24次；最多2状态/12次autograd.grad，0optimizer/rollout，900秒。
- 沿用实际vendor PPO和aux方法。**值预测使用保存mb_values作为无梯度常数**，不实例化Critic；它保持对独立Actor的偏导，但数值总loss仅是所声明常数值项下的目标，不宣称复现实际Critic损失或全训练loss。std冻结，entropy也应无Actor梯度，运行时断言。
- KL只记录，不调整学习率；没有梯度裁剪、Adam更新或有限步反事实。PPO、aux、轮转前96teacher、全384teacher和总Actor梯度均按FP32原式求导；点积/范数用float64归约。
- 总目标标量及Actor梯度加和验收atol1e-7/rtol1e-5；原step0概率RMSE<=1e-5、ratio偏差<=.001；step0teacher最大差<=1e-5。step4teacher RMSE须与原第16optimizer存证.0460553877055645相差<=1e-5，作为保存状态/接口控制，非闭环效果评测。
- Actor全部参数/buffer、保存optimizer tensor-tree和输入前后摘要一致，所有参数.grad保持None；零梯度余弦为null。输入模型文件运行前后SHA不变。
- 原三次间隔5秒可用估计>=8GiB、无其他RSS>1GiB/计算、GPU空闲、可靠遥测；CPU90/GPU85连续两次5秒、4GiB保留、900秒保护不放宽。不丢缓存、不终止其他进程。异常STOPPED，不复跑。

## 版本、命令与证据

实现前已发布main为3bc298bb1f6bf069f13306056d72b84e3de6918b，源码SHA见[实现清单](../data/manifests/a3_retention_direction_implementation_20261013.json)，固定旧输入见[原计划](../data/manifests/a3_retention_conflict_plan_20261013.json)。新代码先审计提交同步，再登记。运行前实际Git/作业/boot/资源必须重查。

本地只执行标准库检查：
```bash
python3 -m unittest script.a3.test_retention_direction_protocol -v
python3 -m py_compile script/a3/retention_direction_protocol.py script/a3/retention_direction_trial.py script/a3/retention_direction_probe.py
python3 -m script.a3.retention_direction_trial plan
```
三项测试已通过；compile和plan通过，不加载张量或模型。

仅ZHY `/opt/data/private/hechengzhu/ZHY`，确认OUT/LOG不存在、资源连续通过后各执行一次：
```bash
/usr/bin/python3 -m script.a3.retention_direction_trial register
/usr/bin/python3 -m script.a3.retention_direction_trial run --stage probe
```
任何已有状态按实际账本处理，COMPLETE/STOPPED禁止重跑。登记或启动资源不足不创建worker，记录WAIT，保持工程ACTIVE。

产物：`data/experiments/a3_zhy_retention_direction_20261013_E46/{registration,state,progress,result,step0_direction,step4_direction}.json`；日志`logs/a3_zhy_retention_direction_20261013_E46/`；累计小清单`data/manifests/a3_zhy_retention_direction_20261013_E46.json`。没有新训练目录/模型。

## 结果与边界

当前只有实现和静态/标准库测试结果，0新模型加载或forward。未测量梯度冲突，不能预判aux竞争或宣称性能改善。即使局部PPO下降方向提高teacher MSE，也只支持下一独立实际步保留约束设计，不能证明19步停止的唯一原因。若不支持冲突，则单独检查Adam/非线性；不扫teacher权重。

## 保存与同步

小型代码/README/清单将scoped审计、提交推送main并核对；同次提交号从Git历史取。旧数据/模型/日志保留ZHY LOCAL_ONLY，网盘未核验。实现登记以前推送状态以实际Git为准。

## 下一步与恢复

先完成实现发布/同步，再按原门槛登记并单次运行E46。结果及健康核验后立即更新本README、索引/current state和小清单；不将梯度诊断当作超越baseline。只有后续独立训练和六衣服/四官方匹配开发效果通过原门，随后独立seed及真正未见会话验收，才可主张提升。
