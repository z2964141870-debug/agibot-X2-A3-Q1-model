# Q1 CPU状态转换等价加速

2026-11-08，北京时间。独立实验CPU_STATE01；原共享训练继续人工暂停，32k仅研究候选，采纳为null。

## 用户摘要

保存的同状态诊断显示，CPU状态转换中位约10至12ms，GPU前向约1.3至1.5ms。先验证用隔离MuJoCo运动学代替逐连杆Torch循环是否能保持策略输入并减少耗时。此项只改善软件处理，不能减少人体未来依赖，也不能修复已观察到的f0闭环失败。

当前已实现独立转换器、五个坐标/不可变性合同及234状态配对检查；尚未执行服务器合同或真实配对比较。没有新的策略前向、物理积分、渲染、优化器更新或GPU预算扣费。

## 方案与边界

- 原函数`script/q1_five_stage/q1_native_review.py:native_state`、原199执行/200冻结来源和旧实验输出均不修改。
- 新`KinematicState`只接受CPU float32 `[1,79]`且不求梯度。复用只读模型，持有独立`MjData`；调用`mj_kinematics`、`mj_comPos`、`mj_comVel`及世界朝向的`mj_objectVelocity(mjOBJ_XBODY)`，读取连杆原点。不调用动力学、积分或`mj_forward`。
- 原输入的根角速度在世界坐标，MuJoCo自由关节qvel使用根局部角速度；明确转换。保留关节参考位置、WXYZ符号、稀疏29槽映射及上一动作尾部。
- 真实策略运行时为Torch2.11.0+cu128/MuJoCo3.11.0；共同CPU物理回放另为Torch2.7/MuJoCo3.6。本次使用前者且隐藏CUDA，不能以本地3.9环境或物理运行时冒充策略运行时。
- 本地相同版本MuJoCo不可用，五合同只登记待服务器执行；本地不编造通过数。

## 预登记协议

只复用已完成936调用的两条实际状态前缀：A022前走128状态、A023起走106状态，共234。每个状态原函数和新函数各调用一次，逐帧交替先后，不增设重复微基准或物理回放。

逐帧检查269状态字段、93维解析观测及930维连续历史，`atol=rtol=2e-5`；连杆原点绝对误差另限3e-5。缺失关节槽必须精确零，上一动作必须精确保留，输入和共享模型不变、仿真时间不动。检查过程中禁止`mj_step/step1/step2/forward/inverse`。坐标合同另覆盖非零根旋转、非零关节ref和anchor、质心与原点的区别、四元数负号与缩放、拒绝梯度/批量/非有限输入和结果不别名。

每片报告全部调用与去前两调用的p50/p95/p99、最大值和超过20ms数；配对顺序逐帧交替。计时只覆盖CPU转换及独立解析观测/历史，不包含完整worker、RPC、GPU、感知、网络或执行，不外推硬件端到端。

总时限300秒、输出64MiB、自身RSS2GiB。准入22GiB/退出24GiB、GPU18GiB/80度、磁盘余100GiB、private小于500GB、原非阻塞锁保留。累计GPU账本2433/14400秒且无未结算预留，CPU实验必须保持账本原字节不变；不清缓存、不重启原训练。

来源为已有SAME_STATE保存分析889文件、原失败根及两trace，加本次三新源码、显式原Primebot源码SHA和原物理机器人checkpoint。运行前后核同，旧FAILED不改写。

## 入口与恢复

新代码：`script/supermimic_teleop/native_state_cpu.py`、`native_state_review.py`、`test_native_state_cpu.py`。输出固定为服务器`data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_CPU_STATE01_20261108`，首次运行前检查输出、日志、收据与进程防重。

先发布三文件清单并独立核远端，再只部署新文件；运行时执行：

```sh
CUDA_VISIBLE_DEVICES='' data/environments/morphmimic_original_20261029/bin/python -m unittest discover -s script/supermimic_teleop -p test_native_state_cpu.py -v
```

通过后使用已发布的清单SHA与源码40位提交运行新`native_state_review.py`，外层`timeout --kill-after=5 300`。优先复用45014；必要恢复仍原样`source /Users/yu/.zshrc`后`ssh -tt hp_han`，不让用户代测、不读取认证文件。

等价失败就保留原输出及差异并定位，不放宽容差或直接替换旧worker。等价通过也先不接入真实策略；后续独立少量GPU动作等价门通过后，才考虑用于新H4时间mask/闭环实验。整体低延迟目标未完成，旧七失败不重跑。
