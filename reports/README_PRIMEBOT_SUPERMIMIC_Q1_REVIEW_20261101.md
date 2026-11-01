# PrimebotTech SuperMimic Q1核查

21:36范围补充：[组织整体核查](README_PRIMEBOT_Q1_ECOSYSTEM_REVIEW_20261101.md)已在primebot_rl_mjlab发现公开起身PPO的model_30000.pt/ONNX、配置与q1_getUp参考动作，独立资产库也有XML/meshes。下文“不随库提供权重/资产”仅指SuperMimic这个固定仓库，不适用于整个组织；PPO起身候选也不能证明通用走路或已训好的SuperMimic WM。各库执行器配置不同，回放必须按模型/配置/权重成套核验。

2026-11-01，用户提供PrimebotTech/SuperMimic，并明确让现有MorphMimic G1训练继续。结论：已经公开Q1专用SuperMimic实现，值得作为Q1控制与足量训练对照；当前公开材料不能证明已有可直接使用的成熟Q1权重。我此前没有核到这个仓库，已补入项目依据。

20:03北京时间进展与见解更新：原G1已101621/160000轮，原controller/worker启动tick、boot/GPU根不变，五optimizer持续更新/非有限0/OOM0/GPU8779MiB。旧20962观察会话超时后按既有授权先记新独立故障，再仅一次6378原生Windows恢复，训练未重启。GitHub API本次重读main仍118338b、releases仍空；组织API名称“启元机器人”、主页primebot.com。不能把README的“独立实现”理解成否定厂商关联；它表示这不是SuperMimic论文作者原代码的原样发布。正式组织归属仍应以厂商渠道为准，不仅凭名字或GitHub认证标志判断。

## 我的判断与优先级

该库应成为后续Q1适配的优先对照实现。它公开了机器人控制和完整训练流程，能减少继续自造Q1底层契约的风险；仍需用同一版XML、关节顺序、动作和足量评测核验效果。此前主要围绕几百轮底座/PPO短试与额外奖励诊断，不足以否定MBRL方法，也不能证明已经公平复刻这套Q1配方。

最值得先核的不是换一个网络名称，而是以下具体差异。左列旧Q1来自已冻结GAIT_GOAL01场景，不能误当当前G1控制配置；右列为此Q1源码，均非真实硬件参数实测。

| 关键项 | 旧Q1实验冻结/已核设置 | Primebot Q1公开实现 | 影响与下一步 |
| --- | --- | --- | --- |
| 执行器与PD | 位置执行器，22关节同KP15.791/KV1.005 | XML要求力矩motor，软件按关节设PD；髋膝40/1、踝20或10/1、腰臂80/2 | 这是不同控制对象和闭环，不能拿同一策略动作数字视为等价；逐关节核模型/gear/默认姿态/力矩通道 |
| 时间与限矩 | 控制20ms/物理5ms，固定36/22Nm | 控制20ms/物理2ms、转速方向限矩34/21Nm及XML界限 | 应完整复制控制周期和执行器曲线后独立比较；不是把旧力矩上限直接调大 |
| 训练量 | GOAL01底座123轮/PPO113轮即按旧物理门正常结束 | 单卡1024环境、WM与policy各512、250000轮 | 短试能说明当时轨迹退化，不能当完整配方效果；示例轮数本身也不证明作者已跑完 |
| 状态与梯度尺度 | 旧原生Q1/MorphMimic各有已冻结定义 | heading动力学、延迟context推进、WM除K/策略目标除H、SiLU | 会改变优化数值及clip前梯度尺度，需整套版本对照，不只抄网络或只改日志显示 |
| 奖励量纲 | 旧实验原定义及独立奖励修订 | 身高比例缩放位置/速度/终止高度、29/22关节和系数、arm权重0.05 | 说明跨机器人不能只改DOF数量；须保留原定义、尺度和门槛并列，不能假定其单独解决抬脚 |

已核[旧控制证据](README_Q1_GAIT_GOAL01_SAVED_PD_DEMAND_20261030.md)显示多数贴地窗口并无控制起点饱和，后续子步亦不支持持续限矩单因。因此，上述控制契约是高优先级差异，仍不是旧失败已定位的唯一原因。

源码main训练每轮采集2/3控制步交替、WM10批、K8、想象H32、critic5x4；单卡1024与八卡8192的全局采样量不同，两配置同为250000轮并不自动构成严格采样等价。该实现的主体是SuperMimic式WM/actor/critic，不能与当前MorphMimic的motion/KP双接口五optimizer、跨本体共享和latent对齐验收混称同一条路线。

公开证据边界也更明确：README称动作质量仍验证；CITATION.cff明确论文级跟踪和真机部署尚未展示；源码不附资产、数据、权重，当前releases空。它的测试在未给真实XML时还会生成合成22关节资产，故工程测试通过也不能替代真实Q1走路视频/评测。这不否定其可能存在未公开的内部训练结果。

执行决定：保持当前G1 160000轮原计划和冻结执行源码；等待原GPU释放后做原版官方评测和本地原速视频。Q1阶段先把现存资产/关节顺序/执行器/时轴/状态/延迟与此实现逐项对齐，若资产满足则优先复现其独立基线，再把旧PPO残差作为后续单独路线。此判断不是本轮另启Q1训练、安装依赖、替换现训或采纳检查点的动作。

新过程快照、API刷新与推断边界见[见解清单](../data/manifests/primebot_supermimic_q1_interpretation_20261101.json)。第一次HTTPS git刷新失败，随后GitHub API ref成功；未把网络错误当仓库消失。资料快照仍固定118338b，不改运行中的71859ca。

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
