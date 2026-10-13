# A3候选与官方保存轨迹分解E43

记录日期：2026-10-13，北京时间。状态：PRECHECK_NOT_STARTED，8GiB额外启动余量预检拒绝，尚未创建登记/账本或运行worker；不是已确认机器故障。

## 目标与单变量

E42/R17训练合同通过，六衣服/四官方完整开发效果拒绝，衣服全程均误差+9.63%且zzx仍倒，三官方动作超过保留预算。下一先分析保存结果，避免将跌倒后的大误差当跌倒前退化，或仅凭两更新决定增加训练。

E43唯一主变量是已保存同条件轨迹的共同跌倒前窗口与关节分组分解。固定E42与E26/E02各十轨迹、输入捕获、指标及E28已验29关节映射SHA；不加载模型、不forward、不生成新仿真步、不做optimizer或CUDA。此实验是描述性比较，不是跌倒原因干预或下个方案收益证明。

## 方法、预算与判据

逐动作核对499或官方各自固定步数、连续tick、有限值，重建跌倒/全程RMSE、捕获preclip到记录postclip语义；双方首1570输入逐值相同、五类参考数组逐值相同，否则拒绝分析。共同跌倒前区间为[0,min(两方首次跌倒tick,全程长度))，双方共享同一窗口，跌倒tick不计入。腿/腰/臂和各关节分别计算RMSE及对总MSE变化的贡献，贡献须求和一致；固定zzx4.4–4.9秒局部窗口和膝0.35rad连续5帧起点仅供描述，不改跌倒判据或当充分条件。

两个最小控制覆盖连续事件起点及共同跌倒前截断，真实输入/初态/参考合同与贡献求和检查同跑。仅ZHY既有环境/单线程numpy，标准库controller，三次5秒被动预检/至少8GiB余量；CPU90/GPU85两次5秒保护、4GiB保留、900秒预算、首异常停止、无重启恢复/重复。原hp3090只Git/传输，Mac小型代码/记录/清单。

## 版本、入口与工件

代码script/a3/cloth_effect_audit.py，复用既有轻量只读资源监督；登记冻结精确代码/输入SHA，不重置E42和旧失败账本。

```bash
cd /opt/data/private/hechengzhu/ZHY
/usr/bin/python3 -m script.a3.cloth_effect_audit register
/usr/bin/python3 -m script.a3.cloth_effect_audit run --stage verify-training
```

data/experiments/a3_zhy_cloth_effect_audit_20261013_E43/及logs/a3_zhy_cloth_effect_audit_20261013_E43/独立；输出training_verification.json只是复用状态机文件名，语义为保存轨迹分解、0训练。小清单data/manifests/a3_zhy_cloth_effect_audit_20261013_E43.json，raw LOCAL_ONLY，未核验网盘。

## 结果与下一步

代码/方案0c4e41b已推送核对GitHub并同步ZHY，标准库py_compile通过。登记入口在额外8GiB资源预检阶段拒绝，OUT/registration/state均不存在，未启动numpy/worker，0模型/forward/新仿真/optimizer。18:51:21只读复核可用估计8,543,608,832字节（约7.957GiB），低于固定8,589,934,592字节；容器RSS370,860,032字节，file cache9,009,848,320字节，其中inactive1,392,312,320/active7,604,383,744。计数不能确认归属或证明缓存是异常；未丢缓存、终止其他进程或改变估算方式/阈值。

同boot d994e71f-bae0-46ba-a5f7-6c2163771c6c/OOM0，GPU26摄氏度/0%/13MiB，最大可见进程RSS147,488,768字节、controller17,858,560字节；无新boot/OOM证据，不把额外启动余量不足当成4GiB运行保护触发或机器首故障。小型[预检清单](../data/manifests/a3_zhy_cloth_effect_preflight_20261013_E43.json)保留准确计数与未启动证据。

恢复先做三次相隔5秒的被动预检；原8GiB余量、无大进程/其他计算及遥测都满足后，检查OUT仍不存在，才能登记/一次运行。若发现已有新账本，以真实状态顺序继续，不重置。根据共同跌倒前与官方局部退化决定更新强度/保留约束或参考层单变量，不先追加训练、不放宽原效果门。官方仍基线，十动作都是开发数据，最终未见验证未开始；工程heartbeat仍ACTIVE，恢复入口与阻塞记录同步仓库。
