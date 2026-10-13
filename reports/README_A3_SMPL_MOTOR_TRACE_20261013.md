# A3保存电机越界与事件排序E31

记录日期：2026-10-13，北京时间。实验A3_ZHY_SMPL_MOTOR_TRACE_20261013_E31，准备，未登记/执行。

## 目标与固定协议

E30在两右踝上下文确认FK在up角超过1.55rad后平台，但尚未扫描整个轨迹的电机越界。只对E26保存qpos逐帧比较六个并联电机与编译模型限位，记录最早/持续5帧越界、越界帧数/超量，排序对照原关节逐帧RMSE>0.2rad、左膝误差>0.35rad连续5帧、倾斜15/30/60度和既有跌倒时刻。唯一主变量为保存电机角的有效范围事件分解，不更改或重跑原参考/策略/solver。

六段各499帧；越界容差1e-6rad，严格跌倒前不含跌倒帧，4.4–4.9秒另列。模型电机range不等于已核验全部solver有效域，更不是动力学硬限位；E30仅核验指定右踝上限。五个独立上下边界值正对照验证比较方向与等号。策略标签为tick/50，实际步进后状态=(tick+1)/50；末端50Hz样本不能推出1ms子步从未越界。时序先后不能代替因果证明。

## 版本、预算与命令

代码script/a3/smpl_motor_domain_trace.py，固定E30验收及继承的输入/源码/模型/头文件/bridge SHA；不调用solver或编译bridge。仅ZHY既有环境单CPU线程，一次900秒、5秒遥测、CPU90/GPU85度连续2次、4GiB保留、首异常停止，无策略权重/前向/回放/optimizer/CUDA/mj_step，不自动重复或恢复。

```bash
cd /opt/data/private/hechengzhu/ZHY
data/environments/a3-sonic/bin/python -m script.a3.smpl_motor_domain_trace register
data/environments/a3-sonic/bin/python -m script.a3.smpl_motor_domain_trace run
```

实验data/experiments/a3_zhy_smpl_motor_trace_20261013_E31/，日志logs/a3_zhy_smpl_motor_trace_20261013_E31/，清单data/manifests/a3_zhy_smpl_motor_trace_20261013_E31.json，登记执行后不重复入口。原始保存状态仍LOCAL_ONLY，Mac只留小代码/README/SHA清单。

## 结果与恢复

待实际执行。完成/失败立即保存README、索引/current state及小清单，audit/scoped提交推送main核对ZHY/GitHub、保留并行修改。越界若晚于最早偏离则先查参考腿部时序/动态支撑；若较早出现也须新单变量验证，不自动把先发生的事件当根因。官方仍基线、七段仍开发参考、长期heartbeat ACTIVE。
