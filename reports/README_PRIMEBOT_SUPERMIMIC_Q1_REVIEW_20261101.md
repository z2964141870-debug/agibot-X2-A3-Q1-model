# PrimebotTech SuperMimic Q1核查

2026-11-01，用户提供PrimebotTech/SuperMimic，并明确让现有MorphMimic G1训练继续。结论：已经公开Q1专用SuperMimic实现，值得作为Q1控制与足量训练对照；当前公开材料不能证明已有可直接使用的成熟Q1权重。我此前没有核到这个仓库，已补入项目依据。

## 固定来源

仓库https://github.com/PrimebotTech/SuperMimic ，固定main为118338b19dcd14f768d1e7de5a48d04b78987688；独立git ls-remote与GitHub API相同。API created_at为2026-10-24T12:18:48Z，最新提交为2026-10-24T12:20:27Z，共两条提交，当前releases为空。13份源码124660B逐Git blob SHA1及大小相同，并登记SHA256；均只存忽略的data/references/primebot_supermimic_review_20261101。没有运行第三方代码或安装环境。

固定[README](https://github.com/PrimebotTech/SuperMimic/blob/118338b19dcd14f768d1e7de5a48d04b78987688/README.md)自述Q1独立实现，仍在研究验证阶段，动作跟踪质量仍待验证；列出不随仓库提供Q1 XML/meshes、重定向动作、训练权重和本地结果。THIRD_PARTY同样声明外部资产不随库分发。这里不将组织名当作论文作者官方关系证明，也不把未公开的私有结果判为不存在；本次核验范围是这个公开版本。

## 可复用内容和差异

| 项目 | 固定源码/示例中实际内容 | 对当前工作的意义 |
| --- | --- | --- |
| Q1结构 | Q1Robot校验pelvis自由根、22执行器/nq29/nv28，从XML建立关节/FK/限位/资产身份 | 是Q1专用实现，不能直接加载为MorphMimic 29关节canonical策略 |
| 单卡新训练 | q1-fresh-single示例1024环境、WM512/策略512、250000轮、replay1000、WM每轮10批、LR2e-4 | 原几分钟Q1短试验不能代表这套完整配方的结果；示例轮数也不证明已完成训练 |
| 控制 | 50Hz控制、物理2ms、action_scale0.5、按关节分组PD | 与旧Q1场景的统一增益/不同尺度需要逐项对照，不能只复制网络 |
| PD表 | 髋膝40/1、踝pitch20/1、踝roll10/1、腰和上肢80/2 | 源码称来自walk/run控制标定；实际资产和控制器匹配尚未在本项目验证 |
| 转速限矩 | 大关节34Nm/拐点7.5/截止17rad/s，小关节21Nm/8/16，另受XML控制范围约束 | 不能与旧固定36/22Nm上限视为同一个执行器模型，更不能据此认定旧步态单因 |
| 延迟和WM | 示例启用modeled_action_delay、heading状态、SiLU、策略/模型目标归一化，关闭额外action_noise/lag | 是独立实现的具体选择，未经验证不改正在运行的原版G1 |
| 奖励 | 示例启用Q1按尺寸比例缩放及arm_joint_weight0.05 | 可作为后续独立Q1对照，不能把改过配方称为原论文原样复现 |

主要依据为固定configs/q1-fresh-single.example.toml、src/supermimic/robot.py、simulator.py、models.py、train.py、state_representation.py和THIRD_PARTY.md。完整文件身份及公开范围见[核查清单](../data/manifests/primebot_supermimic_q1_review_20261101.json)。这是源码阅读结论，尚未做本地XML匹配、数值测试、物理运行或效果对照。

## 当前决定和后续

保持原G1训练：10:50:12核50376/160000轮，原PID/start/boot不变，五optimizer更新/非有限0/OOM0；40000和48000检查点已存，仅stat。新仓库不构成停止或热改现训的依据，不新增GPU并发、agent或修改其他自动任务，MORPHMIMIC_FAITHFUL01保持ACTIVE/采纳null。

后续Q1接入先核这个公开版本与已有Q1 XML/动作库/关节顺序/控制契约的差异，再评估能否原样运行或复用；权重与结果只有取得并实测后才称可用。MorphMimic共享多本体训练仍要完成X2/T2同片数据、共享梯度和latent对齐，不能以Q1专用仓库替代该验收。
