# 当前帧参考路线的公开代码核对

2026-11-06约08:42完成，只读静态审计。沿HuMiT作者项目页的Code入口核Track默认main，固定ec9890bbcef3511229f0c2a0bb9ffc54c4ccc7fd；不把论文结果直接算作本项目效果，也没有安装或执行作者训练器。

## 核到的代码与发布边界

完整Git树19项、未截断。默认main最后提交日期为2026-03-17T10:15:39Z，消息archive；本次独立ls-remote核同。另有arxiv-project-page分支，未发现tag，GitHub releases返回空。八份关键文件73768B均核Git blob SHA1和本地SHA256，[逐文件实据](../logs/session_records/q1_humit_source_audit_20261106.json)。

公开main的cfg/task/motion.yaml使用G1、50Hz策略时钟；policy包含机器人状态/前动作和ref_qpos、ref_qvel、ref_linvel_b、ref_angvel_b。observation.py:58/72/86/99的四类reference观察均用episode_start+episode_length对应的当前索引，没有在这几个观察类中取t+10。

ppo.py:138的actor读取归一化policy组，critic另读priv组。这给“当前参考和机器人状态进入actor、额外训练信息留在critic”的结构提供可检查实例。但当前公开树没有论文完整的稀疏人体目标配置、实时传感/通信部署入口、检查点或ONNX；其policy组目前是完整重定向关节/速度参考。不能仅据此树声称已复现论文的紧凑在线输入或低延迟结果。README要求另取动作数据与外部Isaac Sim/Isaac Lab/active_adaptation依赖，不是本项目可立即替换的Q1模型。

## 隐含未来依赖

部署输入的索引名为current，并不自动证明数据前处理因果：

| 路径 | 此快照的实现 | 对Q1新接口的要求 |
| --- | --- | --- |
| 关节与根线速度 | load_npz.py:156/157及load_pkl.py:155/156用torch.gradient | 中间样本中心差分读取t+1；实时输入改用到达历史的后向/回归估计，并与训练匹配 |
| 根角速度 | 两转换器的_so3_derivative用q[t-1]和q[t+1] | 不可把离线中心SO(3)导数当作当前可得传感速度 |
| 重采样 | _compute_frame_blend用左右端点，姿态Slerp | 如实记录右支持等待或改用到达历史，位置与姿态共用时轴 |
| actor取速度 | command.py:271后直接读保存joint_vel/body速度，观察再取current索引 | 只改最后索引无法消除上游已有未来依赖 |

例如输出50Hz时，中间速度的中心差分需要额外一个20ms输出帧；输入插值还可能有右端支持。这是上述离线转换路径的源码依赖推导，不是对作者在线前端的测量；本次没有取得论文完整在线实现，不能将这个数字用于修正作者报告。

原Q1/MorphMimic的侧车与上游捕获也尚未获实时因果认证，先前已核120→50Hz插值的右支持。不同仓库的转换路径不能混加成Q1总延迟。新的因果学生要从捕获、重采样、重定向到速度特征逐段核来源与可用时刻。

## 对本项目的取舍

近期继续同human接口H10/H5/H4/H2/H1和H4/H5尾部对照，以准备好的相同权重/物理/资源门隔离预览代价。实际结果待原桥接，不将本次静态审计当作物理比较。

长线优先新增独立的当前参考actor：输入当前可得人体姿态、因果人体/机器人历史、每个输入的年龄与可用性；历史速度与姿态使用同一时间轴。世界模型可以继续服务训练，未来目标只进入教师、训练损失或特权critic；部署actor接口应通过“改不可见未来不改动作”的实际运行时检查。当前full权重的H1仍对应f+1，不能靠改名冒充f0零预览学生。教师本身低离地/滑移问题仍需解决。

本轮没有实施新的因果学生训练、修改当前PD或替换依赖。原V04和final/完整五阶段门保持。先完成真实短预览失败证据，再定独立训练的输入、预算和保存准入。

## 原连接与交付

约08:46按原授权做一次有界恢复：Windows100.65.183.91:22的TCP成功，原生SSH出现交互认证提示后Broken pipe退出255，未取得Windows/Linux shell；未上传、未下发、不循环认证或换入口。认证只交互使用，不入本记录。实时训练继续UNKNOWN，不能据此停止/重启原训练。

作者源码和API响应仅保留Mac主根data/supermimic_for_q1/Q1_HUMIT_CAUSAL_SOURCE_AUDIT_20261106；未把第三方源码导入本项目执行或推Git。load_pkl.py一次raw请求25秒超时，随后同固定Git blob API读取并核同；不是更换服务器登录入口。

本轮只读来源与代码身份核验，不新增低影响检查测试或重复物理。原抗噪结果f20498399ea4e8a549ec43b0b0dd6ad389bd29af已显式push、独立40位远端与Mac七份606603B大小SHA核同，[收据](../logs/session_records/q1_upper_noise_robust_publication_20261106.json)。goal保持active，最低今日10:00，用户采纳null；本次发布及服务器镜像以后续实据为准。

## 一手来源

- 作者项目页：https://xiaohu-art.github.io/Track/ 。
- Code入口/固定Git：https://github.com/xiaohu-art/Track/tree/ec9890bbcef3511229f0c2a0bb9ffc54c4ccc7fd 。
- HuMiT论文：https://arxiv.org/abs/2610.29903 ，论文级结果已在首轮研究记录，本轮不重复当实验。
