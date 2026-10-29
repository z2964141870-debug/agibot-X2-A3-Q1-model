# Q1 H32起点差异的离线诊断准备

2026-10-29后续更新：Windows/25386连接及本页待镜像已恢复，见[恢复发布收据](../data/manifests/q1_windows_ssh_hp_han_restore_publication_20261029.json)。已读实际策略/归一化/重置源码，尚未证明随机性、BF16或后端是差异原因；安装MuJoCo-Warp3.11.0的types没有DeterminismType定义，仅是诊断接口边界。新增[SOLE02保存状态奖励分析](README_SUPERMIMIC_Q1_REWARD_SIGNAL_20261029.md)，没有第三次H32物理运行、PPO16/4096或新训练。下方UNKNOWN/待镜像为当时历史，不代表当前连接。

记录日期：2026-10-29，北京时间15:56，16:04更新发布状态。阶段：SOLE01_INIT。状态：逐步只读记录实现及8项离线测试通过；代码1f224fd已独立核GitHub，10项308960字节Git/Mac逐大小SHA一致，服务器镜像待。原生验证未运行，连接UNKNOWN。底座与PPO两线继续，尚未查明起点差异的原因。

## 目标与范围

用户要求在SuperMimic内同时改进底座和PPO适配，并在连接恢复后继续。本轮延续[SOLE01](README_SUPERMIMIC_Q1_SOLE_CANDIDATE_20261029.md)，没有沿用过期夜间预算，也没有第三次H32物理运行、PPO16/4096预检或正式训练。原V01/V02失败收据保留，候选采纳null，原监督/其他自动任务不变，不启用agent或旧goal。

V02仅给出warm-up末端状态最大差.07455075、上一动作差.015625及warmstart差13.735565；原收据没有逐步状态，不能倒推出从哪一步开始不同。两次原运行合计576控制转移/2304子步、0optimizer，没有有效物理方向结论。

## 实现与源码线索

`script/supermimic_q1_sole/initialization_audit.py`提供纯NumPy差异记录，`h32_direction.py`在原10个状态/9次动作的warm-up里接入。每步记录状态、历史观测、动作、goal和可见的qpos/qvel/qacc/warmstart/ctrl/time/外力/act数组；同步Torch/Warp后只读复制。没有增加env.step或优化器调用，原末端状态/历史断言及atol2e-6/rtol2e-5保留。

收据分别给出最早状态/动作差异和最早随机状态差异，避免把随机数状态不同直接当作物理因果。记录Torch CPU/CUDA、Python和NumPy随机状态，另记录策略参数SHA、training模式、autocast/cache、TF32以及MDP对象归属。BF16数组精确转成FP32供NumPy比较；这不改变网络实际推理精度。

本地源代码给出两条需要核查的初始化路径差异：初始环境prepare之后还构造了三个网络，而fresh分支重新prepare、保留既有网络；prepare会设置全局seed0。另一个差异是fresh分支替换了env，但仍持有初始mdp/motion对象。新增记录暴露这些路径，当前没有证明任一条导致数值差异，未据此改重置语义、放宽保护或归责BF16/世界模型。

这些只是选定数组的观察，不能称完整可恢复的MuJoCo Warp求解器快照。原生learner/networks在Mac没有缓存，本轮固定提交的HTTP读取遇连接重置，未取得或换用其他版本源码，不能声称重新核过上游实现。

## 实际验证

在持久权威Git的Q1分支执行：

```sh
python3 -m unittest discover -s script/supermimic_q1_sole -p 'test_initialization_audit.py' -v
python3 -m py_compile script/supermimic_q1_sole/initialization_audit.py script/supermimic_q1_sole/h32_direction.py script/supermimic_q1_sole/test_initialization_audit.py
python3 script/supermimic_q1_sole/h32_direction.py --help
git diff --check
```

8/8测试通过，0跳过；覆盖原浮点容差与字节变化、整数精确比较、shape/dtype/非有限/空数组、动作先于状态偏离、RNG独立分类、参考副本、求解器/缺字段、阶段缺失及JSON序列化。首轮离线测试发现NumPy int64索引不能直接JSON序列化，已转Python int并增加标量/非有限收据反例，最终通过。该离线修正不是训练重试。

三文件编译、CLI help和diff检查通过。日志位于`logs/session_records/sole01_h32_init_audit_20261029.log`，995字节。没有Torch/MuJoCo实际导入、策略推理、积分或optimizer更新，不能将这组测试记为原生起点问题已修复。代码与范围见[清单](../data/manifests/supermimic_q1_h32_init_audit_20261029.json)。

## 连接与保存

用户本次“应该好了，继续”之后，仅有的一次Windows桥恢复在非回显认证交互中Broken pipe，没有建立18873转发，没有执行容器命令。它不是Permission denied，不能据此判密码不正确。凭证未写入文件/命令参数/Git。本轮不重复桥探测；远端身份、GPU和作业当前均UNKNOWN，旧退出/GPU15MiB/OOM0仅历史。

GitHub初次SSH只读核查在连接22端口时超时；随后显式push origin Q1成功，独立ls-remote得到完整`1f224fd95f8df0d1c2c5efc98300a81cfa5f4931`。该提交10项308960字节与Git blob、工作树和Mac主根逐项大小/SHA一致。[本轮发布及Mac核验收据](../data/manifests/supermimic_q1_h32_init_local_publication_20261029.json)记录该固定来源；本次收据/文档的后续提交以Git历史为准。25386镜像仍未完成，不称三端一致。

旧待补范围仍是`supermimic_q1_sole_native_publication_20261029.json`及6772cf5的五个小状态文件；本轮新增来源/测试/日志/README/清单也待服务器同步。权重和大NPZ仍SERVER_LOCAL_ONLY，未搬运。

## 下一步

连接成功后先核身份、GPU占用和来源，再补小记录镜像、读取固定原生实现，检查新增记录在实际Torch/Warp数据类型上的兼容性。V02失败后的停止规则仍生效：本轮没有安排第三次物理运行或依赖的PPO16/4096，更没有新正式训练。先通过离线与只读证据缩小差异，再据证据确定下一次独立验证范围。

底座脚底Z候选仍有原尺度信号弱的限制；PPO候选仍只有CPU契约。两者的学习信号、更新稳定性和净物理收益分别核查，不能用诊断工具通过或单一奖励定义替换宣称自然步态改善。
