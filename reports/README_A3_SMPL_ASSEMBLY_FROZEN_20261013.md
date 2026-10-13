# A3补全被动状态的冻结闭环E34

记录日期：2026-10-13，北京时间。实验A3_ZHY_SMPL_ASSEMBLY_20261013_E34；14:23:43停止，六段回放均执行499步，zzx仍跌倒；动作捕获验收断言失败，旧分支不复跑。

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

登记14:21:50，执行14:21:54–14:23:43，共109.03秒，0装配复算。新状态输入已实际加载；六个metrics均499步，共2994步，前五段记录无跌倒，zzx记录首次跌倒标签6.06秒（旧E26为5.66秒）。仅延后0.40秒，不是解决跌倒或达到原零跌倒门；仍不能宣称基线提升。

独立验收在captured actions与timeseries raw_action_29精确比较时失败：某段4/14471元素不等，最大差1.913624。官方控制链有ACTION_CLIP，捕获与记录可能分别来自裁剪前/后；尚需独立语义核验，不为通过而放宽数值门或改写E34结果。旧E34 STOPPED保留，6完整回放不重复。下一独立E35只对保存数组核验裁剪协议、目标与初始输入，并完成动态/误差对照；0新回放/optimizer，不重复装配。

23健康样本同boot d994e71f-bae0-46ba-a5f7-6c2163771c6c，CPU40/GPU35摄氏度，最大GPU功率113.57W、最低可用9,197,752,320字节、OOM0，无新关键内核/遥测异常，最大采样间隔5.256秒。不是机器故障；计算已退出。代码基线310683c2e07712026c43d10977296dca0874ef2b，原始6组回放/输入/接触/日志LOCAL_ONLY。

本结果已写README，原STOPPED小清单、索引/current state随本阶段scoped audit/提交/push核验。无用户待补资源，长期工程ACTIVE；还没有新策略微调或未见验收。E33结果与失败提交310683c已核对GitHub/源机/ZHY，保留生成文件scoped stash。

## 后续方向

被动初始化修正没有消除本开发步行跌倒，暂不能升级。下一E35用已保存结果结清动作/目标/对照语义，再以具体跟踪短板选择受约束的小微调或接触根运动参考修正，不扩大无关诊断，不用分析次数当效果。新训练同时保留官方动作能力及未见会话验证。相关最新论文核验另存当前主线查新记录，不自动换框架。
