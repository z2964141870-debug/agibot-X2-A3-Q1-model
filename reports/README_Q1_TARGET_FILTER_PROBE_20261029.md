# Q1腿部目标时序滤波对照

2026-10-29 02:57，北京时间。独立quick33已02:56:08实际启动，controller145547/start_ticks12741303、worker145550/start_ticks12741645，原boot/GPU一致。冻结494f45a6fed67488c2af81c0bc1791b371d7b474独立远端/10文件258758字节三端SHA已核，02:57:34为baseline repeat2/frame200、0optimizer，尚无候选效果。7CPU反例/CLI help/解析及27行已存轨迹集成均通过；正式运行来源冻结。07:00停新优化/物理探索、09:00交付，不恢复/重试旧训练。

## 证据与边界

RF15与参考相位腿部前馈都没有可重复改善普通走。仅读取已存9份NPZ，比较phase baseline、RF15静态和学习head，各三次、固定两走/抬手，数据在data/supermimic_for_q1/Q1_TARGET_DYNAMICS_DIAG_20261029/summary.json。PSD用去均值Hann窗50Hz位置、12腿合并，不是力/扭矩频谱；PD误差使用20ms末状态，不是5ms内部瞬时误差。

RF15静态普通走目标变化RMS约.0893至.1142rad/20ms，99分位约.3205至.3884rad；18%至28%的逐关节变化大于.1rad。10Hz以上位置频谱只约.32%至.44%，不支持高频占主导的断言。学习head在停走repeat0/2把变化RMS提高至.1301/.1293rad、目标对末状态误差约.3704/.3585rad；repeat1相反，仍不能由闭环相关性认定因果。轻微滤波可能削弱突变，也可能增加滞后并破坏平衡，必须物理对照。

当前runtime控制JSON实际统一kp15.791367/kd1.005310，来源为Q1 armature.004与既有公式；腿effort36Nm、踝22Nm。它们在本对照保持不变，不把统一PD推断成已确认错误，也不伪造实测力矩。MuJoCo官方XMLreference的position/velocity/dampratio定义只提供控制接口说明，不证明Q1参数优劣；本轮网页请求未返回可读内容，不引其作为新研究结论。

## 方法与预登记候选

固定RF14两小时stage_02.pt教师/归一化、手臂alpha.5、腰和原PD/限矩/噪声/合法采样不变。仅对12腿物理弧度目标做时序处理，随后除以原action_scale还原动作。首次目标直接初始化，环境重置清该环境的历史；不会用0位置产生启动瞬变。以下是50Hz策略输出后的滤波，不是MuJoCo原生5ms actuator dynamics。

| 候选 | 处理 |
| --- | --- |
| baseline | 完全不变，逐步assert action位等价 |
| ema010 | 一阶时间常数10ms，系数1-exp(-.02/.01) |
| ema020 | 时间常数20ms |
| ema040 | 时间常数40ms |
| rate010 | 每20ms每腿关节目标最多变化.10rad |
| rate015 | 每20ms最多变化.15rad |

EMA和限速都在上一输出与当前请求之间，不增加目标幅度或力矩上限；可能产生延迟，不宣称是安全保证。3项源码位于script/supermimic_q1_target_probe，独立复制物理评测路径以保留旧1166来源不变，未改旧phase/HP/PPO/balance/gait源码或资产。

原phase probe在准备环境前设置32000+repeat，但既有prepare函数随后把cfg.env.seed和cfg.algo.seed重置0，故旧种子标签不能证明三个独立随机种子。其原JSON不改，报告补正为同一配置重复。新prepare_seeded明确在create_env前设置环境/算法/全局seed32000+repeat，teacher构造后、rollout前再次设置全局随机状态；逐case落seed_contract.json。原生GPU求解器仍不保证位级一致，三次不是独立训练seed。

## 验证与执行

7项CPU反例：baseline位等价、大初值无启动瞬变、物理rad限速、EMA衰减/滞后、跨环境重置隔离、不超出前值与请求范围、错误维度/非有限拒绝。远端真实Python经activate.py测试和help通过；首个频谱工具换行写法在运行前修正，27行实际集成/JSON可读完成，不把这些当物理改善。

正式发布/独立40位远端及三端SHA后才启动唯一quick根data/supermimic_for_q1/Q1_LEG_TARGET_FILTER_PROBE_20261029。原30验证+固定两普通走/抬手=33环境，六候选各三重复，新建环境并匹配正确种子。教师/参考/来源冻结，0optimizer，单次壁钟<=2400秒/900秒真实无进展；100GiB余盘/12GiB实验/30GiB共享/OOM0/86度，07:00截止。失败不自动重试。

分别报告两走滑移、摆脚误差/抬脚、真实前进、倾斜/指令变化，以及上肢/腕、抬手幅度/时序/漂移/完成和验证失败。根XY仅诊断；不因滤波后目标变化下降就判物理成功，不以两走均值掩盖一片退步。只有可重复且没有明显物理代价的候选才值得独立full330复核和固定六类型视频；旧继续门仅诊断，不自动授权新训练或用户采纳。若皆无收益则保留负面结果，另找控制/目标适配原因，不把未用预算用于盲重试。

## 版本与交付

RF15交付/腿部相位结果已e55e3bd74a11e921271d31a2dd306e6570b0d081发布，独立核远端和9记录362890字节三端SHA一致；RF15四份54秒本地视频/六格及71工件、phase26小工件已齐备。原审计不重复。新对照仍先准备，实际启动身份/冻结版本/结果随现场补记；user_accepted_checkpoint=null。原q1-rf08-4096保持ACTIVE每10分钟，其他任务不变。

02:57实际启动收据见[data/manifests](../data/manifests/supermimic_q1_target_filter_launch_20261029.json)。新根data/supermimic_for_q1/Q1_LEG_TARGET_FILTER_PROBE_20261029已存在，只监督原controller/worker，不能重复启动。首查GPU725MiB/37度、共享16124006400字节/OOM0，盘余1643644420096字节；正常计算推进不是改善结论。正式冻结含继承1166加3新源码，终后再核实际来源数及guard/进程退出。
