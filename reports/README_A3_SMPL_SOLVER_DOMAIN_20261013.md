# A3右踝solver输入有效域E30

记录日期：2026-10-13，北京时间。实验A3_ZHY_SMPL_SOLVER_DOMAIN_20261013_E30，准备，未登记/执行。

## 目标与固定决定

E29两例zzx右踝FK/DFK失配发生于明显倾斜后，up电机实际1.62398/1.57500rad超过固定MJCF上限1.55rad。只检查solver FK边界输入行为，解释“一个导数列为0而DFK非零”的局部现象，不重复2994步审计，不重跑策略或将它当初始跌倒根因。唯一主变量为right ankle up motor输入；两例down电机上下文分别固定为E29已保存值。

每个上下文在上限1.55rad附近固定偏移[-.02,-.005,-.001,-.0001,0,.0001,.001,.005,.02,.08]，另加入原始up角，共22个输入；保存FK、IK(FK)、域内/域外、相对边界输出差。MuJoCo编译模型确认两电机限位均[-1.55,1.55]，不依赖XML字符串假定。域外FK输出与上限输出差≤1e-9rad、上限以下0.02rad输出响应>1e-4rad、相同输入回访差≤1e-8rad，才认定“在这些上下文中与模型上限匹配的平台”。域内IK(FK)电机重建门0.002rad另列，不为平台发现放宽。局部输入扫描不覆盖整个机构有效域，不证明源码怎样夹紧或物理承载能力。

## 版本、预算与命令

代码script/a3/smpl_solver_domain.py；固定E29验收、继承输入/源码/模型/完整solver头文件/已编译bridge SHA。仅ZHY既有环境单CPU线程、一次900秒、5秒遥测、CPU90/GPU85度连续2次、保留4GiB、首异常停止。无策略权重/前向/训练/CUDA/mj_step/solver重编，不自动复跑或开机恢复。

```bash
cd /opt/data/private/hechengzhu/ZHY
data/environments/a3-sonic/bin/python -m script.a3.smpl_solver_domain register
data/environments/a3-sonic/bin/python -m script.a3.smpl_solver_domain run
```

实验data/experiments/a3_zhy_smpl_solver_domain_20261013_E30/，日志logs/a3_zhy_smpl_solver_domain_20261013_E30/，清单data/manifests/a3_zhy_smpl_solver_domain_20261013_E30.json；登记后不重复入口。Mac只保留小型代码/报告/清单，原始工件LOCAL_ONLY。

## 结果、记录与恢复

待实际核验，不把实现当完成。完成或失败立即更新README/索引/current state和小型SHA清单，检查diff/暂存/audit后scoped提交推送main并核对ZHY与远端，不覆盖并行工作。随后优先独立检查更早4.4–4.9秒参考腿部时序/动态支撑可行性；未定位初始失稳、未改善官方，七段仍是开发集，工程heartbeat ACTIVE。
