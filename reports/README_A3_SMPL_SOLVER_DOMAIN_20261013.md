# A3右踝solver输入有效域E30

记录日期：2026-10-13，北京时间。实验A3_ZHY_SMPL_SOLVER_DOMAIN_20261013_E30，13:16:17完成，VERIFIED_SOLVER_BOUNDARY_SCAN。

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

## 实际验证与结果

登记13:16:07，运行13:16:11–13:16:17，共5.38秒。模型编译后确认两个右踝motor限位均[-1.55,1.55]rad；两上下文共22输入完整扫描。平台判据通过，boundary_behavior=CONFIRMED_MODEL_LIMIT_FK_PLATFORM，域内roundtrip合同PASS。

| 检查 | tick272上下文 | tick281上下文 |
| --- | ---: | ---: |
| FK在全部域外输入相对上限输出的最大差rad | 0 | 0 |
| 上限以下0.02rad的FK输出变化rad | 0.00493945 | 0.00501198 |
| 相同上限输入回访FK差rad | 0 | 0 |
| 域内IK(FK)最大电机重建误差rad | 8.4119e-6 | 4.6687e-6 |

1.5501/1.551/1.555/1.57/1.63rad和原始1.62398/1.57500rad，在各自固定down电机上下文下都给出与1.55rad完全相同的FK输出。1.53/1.545/1.549/1.5499rad则有连续非零响应，排除整条FK恒定/检查路径不工作的解释；IK(FK)在域外返回约1.550008/1.550005rad，接近上限而不恢复原始越界角。支持在这些上下文内FK对超过up上限的输入形成平台，与E29该电机列的数值导数0一致。E29 DFK该列仍非零，故两条API路径在这些域外状态并不一致；这是局部可复现接口行为，不是已取得内部solver源码或全域正确性的证明。

这两例都晚于E26关节误差4.68秒及倾斜30度5.40秒。证据能解释晚期转换失配，不能证明它触发了最初失稳，也不能把所有并联速度差都归因此平台。尚未检查整条轨迹全部六个电机首次进入无效域的时刻，其他机侧/腰/负限位同样未覆盖，不能笼统宣称“首次越界一定只在5.44秒”。暂不修改PD、FK/DFK、观察速度或夹紧策略输出；后续诊断应显式标注输入有效域，并先定位最早越界，避免把域外回算当正常反馈。

3健康样本同boot d994e71f-bae0-46ba-a5f7-6c2163771c6c，CPU最高41/GPU26摄氏度，最低容器可用8,963,358,720字节，OOM0，无新关键内核/遥测异常，最大采样间隔5.173秒。0策略权重加载/前向/新回放/optimizer/mj_step/无CUDA/无bridge重编，任务正常退出。代码SHA d10c4496a60b9a32a35fbcdc4983601110af36cc78c236172b30ef7539f3a11e，清单SHA c7f9403e9364a4d6e83f96bb79f29665d764bab5a2d89594f02178ee3aeb861c，10注册代码SHA与Mac镜像一致；基线HEAD a31029b4956343952a9195ef8175f4dce978da42。

## 保存与恢复

本README、索引/current state和小型SHA清单已保存，结果scoped提交推送/ZHY同步待完成，原始工件LOCAL_ONLY。下一新E31优先逐帧对照六个并联电机实际qpos与各自有效域，记录最早越界/持续区间并与膝误差和倾斜排序；不重跑E26/E29/E30，不能从末端50Hz样本推断完整1ms子步覆盖。之后聚焦更早参考腿部时序/动态支撑可行性。未定位初始失稳、未改善官方，七段仍是开发集，工程heartbeat ACTIVE；没有新机器故障，无需用户补充资源。
