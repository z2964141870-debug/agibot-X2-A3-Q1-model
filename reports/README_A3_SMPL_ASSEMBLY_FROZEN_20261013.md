# A3补全被动状态的冻结闭环E34

记录日期：2026-10-13，北京时间。实验A3_ZHY_SMPL_ASSEMBLY_20261013_E34；准备完成，待登记/启动。

## 目标与依据

E33六段2994帧/17964连接装配验收通过，最大距离3.447e-7m，原root/29关节/驱动/碰撞geometry精确不变；闭环因入口回执目录不存在在策略步前退出，0步。旧E33封存；本分支复用已通过SHA核验的六个NPZ，不重复装配或旧E26基线，只完成被动状态补全的冻结官方闭环对照。

## 固定协议

物理主变量仍仅为原参考的六被动rod姿态/派生速度；新执行入口先创建OUT/entry并用实际atomic_json检查同类路径可写，保留原三个E33模块。新代码script/a3/smpl_assembly_recovery.py，继承E33固定代码/原输入/门槛，增加其STOPPED状态、失败归档、assembly_contract及六个NPZ的SHA。新目录通过只读用途软链复用旧NPZ，实际入口再次核验原始qpos及SHA。

冻结官方035、seed0、六段各499步、原30Hz参考/50Hz策略/无延迟，初始完整1570输入与E26必须相等。独立动态门仍根高<0.45m或倾斜>60度判跌倒；新六段均不倒，旧五段各误差不得恶化>2%，等权平均误差下降才通过开发对照。0optimizer，开发集不是未见验收，不能宣称训练权重提升。

仅ZHY既有环境单CPU线程/串行计算，一次900秒，5秒遥测，CPU90/GPU85摄氏度连续2次、4GiB保留、首异常停止，无开机恢复/复跑。原hp3090不计算；Mac仅小代码/报告/清单。无真机、驱动/功率/BIOS操作。

## 命令与产物

```bash
cd /opt/data/private/hechengzhu/ZHY
# 已执行且不得重复：record-prior封存E33策略步前失败。
data/environments/a3-sonic/bin/python -m script.a3.smpl_assembly_recovery register
data/environments/a3-sonic/bin/python -m script.a3.smpl_assembly_recovery run
```

实验data/experiments/a3_zhy_smpl_assembly_20261013_E34/，日志logs/a3_zhy_smpl_assembly_20261013_E34/，小清单data/manifests/a3_zhy_smpl_assembly_20261013_E34.json。旧原始装配工件继续保留，所有大型工件LOCAL_ONLY。

## 实际验证与保存

新模块语法检查通过，E33失败已独立归档、0策略步且健康正常；新E34尚未登记/执行，不记闭环通过。准备协议与E33失败先scoped审计/提交/push核验，再登记一次。不存在机器新故障，无用户待补资源，长期工程ACTIVE。

## 后续方向

新闭环结果直接决定被动初始化修正是否有用。若仍跌倒或退化，保存反例，下一以合格参考的平滑/接触根运动或官方行为约束的小微调作为独立主变量，保留能力与未见验证必须同行。不继续把分析次数当效果。最新论文以当前可访问原文/代码和日期核验，相关思想登记为候选，不自动换框架。
