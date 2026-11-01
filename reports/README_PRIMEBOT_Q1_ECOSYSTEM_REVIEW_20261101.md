# PrimebotTech Q1公开资料整体核查

记录日期：2026-11-01，北京时间。阶段：PRIMEBOT_Q1_ECOSYSTEM_REVIEW。状态：公开源码与静态XML核查完成，权重效果和部署尚未验证。

## 结论与范围

用户要求利用现有训练时间查看整个PrimebotTech组织。本次列出12个公开仓库，重点固定Q1资产、GMR、PPO、mjlab和SDK五库版本，核25份小源码/配置256320B；不运行第三方代码、安装环境、另启GPU任务或更改正在训练的G1。

最重要的新发现：SuperMimic仓库本身不随库提供权重，但同组织的primebot_rl_mjlab已公开model_30000.pt、两个ONNX、训练配置以及q1_getUp.npz参考动作。它们构成可继续核对和回放的Q1起身PPO基线，不能由此推断已提供通用走路策略或SuperMimic世界模型权重。此前“无随库权重”的判断仅适用于SuperMimic这个仓库，不能扩展到整个组织。

## 仓库分工与固定版本

| 仓库 | 本次固定commit | 可用于当前Q1工作的内容 |
| --- | --- | --- |
| primebot_rl_mjlab | 2153a6c561b3b755ae5bcca89683cb66d5ad5852 | BeyondMimic式单动作跟踪/PPO、公开起身检查点、ONNX和配置 |
| primebot-q1-urdf | 9657817ba375f41b0e5d2b62d8d6b72b757cdf4f | Q1 XML/URDF/meshes与力矩motor资产候选 |
| GMR | a7d0682f1bdd10bdb0b11910cf786d86e1423522 | Q1 SMPL-X/BVH重定向与比例/足部映射 |
| mjlab | 4a19cb64fcc2c25f372d75adf998f8a79f55b727 | PPO库明确要求的后端分支及执行器实现 |
| primebot-q1-sdk | 7310314eee058323a2c15ab1f4f947df4112573e | ROS2/FastDDS关节与高层运动接口 |
| SuperMimic | 118338b19dcd14f768d1e7de5a48d04b78987688 | 此前已独立核查的Q1 WM方法实现与完整训练配方 |

其余公开仓库为T1 SDK/资产、三项toolchain和Q1 DIY外壳，当前仅登记用途，未把构建工具或壳体当训练模型。组织名和厂商关联判断沿用[SuperMimic核查](README_PRIMEBOT_SUPERMIMIC_Q1_REVIEW_20261101.md)，与论文作者原代码的关系另行区分。

## 公开PPO检查点实际是什么

固定目录为primebot_rl_mjlab/logs/rsl_rl/q1_tracking/2026-10-10_17-46-17。Git树中：

| 工件 | 字节 | 本次验证层次 |
| --- | ---: | --- |
| model_30000.pt | 6290695 | 树中大小/Git blob身份；未下载或反序列化 |
| 2026-10-10_17-46-17.onnx | 1452870 | 同上 |
| q1_getUp_os.onnx | 916378 | 同上 |
| params/agent.yaml | 1029 | 内容下载并逐blob核对 |
| params/env.yaml | 34504 | 内容下载并逐blob核对 |
| src/assets/motions/q1/q1_getUp.npz | 536594 | 树中大小/Git blob身份；未下载或读取 |

保存env明确motion_file=q1_getUp.npz、4096环境、物理2ms/decimation10即控制20ms，copy_motion_end=true、结束复制4.8秒。保存agent是seed42、24步/环境、30001轮、actor/critic均512-256-128/ELU/观测归一化，PPO为LR.001/自适应KL.01/clip.2/entropy.005/5epochs/4minibatches/gamma.99/lambda.95/gradclip1。文件名30000不等于本项目已核内部optimizer或成功率。

回放有实际版本缺口：保存env的spec_fn仍指向src.assets.robots.q1.q1_constants.get_spec，当前main则是q1_v3模块；保存目录另有367623B项目diff与5819B rsl_rl.diff，本次只登记其Git树身份，尚未读取。需恢复或解释训练时代码/配置/导出metadata，不能只拿当前main和pt就称原样复现。两个ONNX是否完全对应同一checkpoint也未核实。

## 不能混用的控制契约

| 项目 | SuperMimic固定实现 | primebot_rl_mjlab当前Q1配置 |
| --- | --- | --- |
| 控制对象 | 力矩motor，软件PD | DelayedActuatorCfg包裹BuiltinPositionActuatorCfg |
| 髋/膝KP/KD | 40/1 | 30/1.5 |
| 踝KP/KD | pitch20/1、roll10/1 | 20/.8 |
| 腰与臂 | 腰臂80/2 | 腰roll80/3.5、腰yaw30/1.5、臂20/.8 |
| action scale | .5 | .25 |
| 延迟/限矩 | modeled_action_delay与转速限矩34/21Nm | lag0..10、配置限矩36/21Nm |
| 数据与方法 | WM驱动的跟踪学习，单卡示例250000轮 | BeyondMimic/RSL-RL PPO单动作跟踪，保存配方30001轮 |

PPO库sim2sim脚本读取ONNX的joint_names/default_joint_pos/joint_stiffness/joint_damping/action_scale，但实际软件PD又将全部tau限为±18Nm。这与训练配置36/21Nm存在差异，须在回放前核对。该差异尚不是已证明的效果故障，更不能据此解释我们旧Q1失败的单因。

优先复制完整的模型、执行器、观测、默认姿态、时轴、动作、权重及训练配方；不从各库挑不同增益拼成一个未经测试的控制器。

## 模型与重定向资料补齐到哪里

Q1独立资产库30个XML/URDF/mesh条目的Git blob，与GMR/assets/q1及PPO/src/assets/robots/q1_v3内对应条目全部相同。这是固定树的身份对照；除三份小XML外，网格/URDF本次未下载，不能当物理兼容验证。

REXML结构解析排除了注释里的旧motor：q1_v3.xml及full_range均pelvis自由根/22hinge但0执行器；q1_v3_mc.xml则有22个一关节一motor、标量默认gear1、髋膝腰36Nm/踝臂22Nm，无mocap/equality。后者可作为SuperMimic所需力矩资产的候选。其XML声明物理1ms，不能忘记实际SuperMimic会配置2ms；MuJoCo资产加载、默认参数继承和Q1Robot完整校验仍未运行。PPO还使用额外的q1_v3_mc_without_scene.xml，不能将两个变体视为相同。

GMR真实params路由有smplx、bvh_lafan1、bvh_qingtong到Q1，未见bvh_soma路由。README虽列soma解析选项，也明确是否可用取决于目标IK映射。我们SOMA559不能直接据此宣称已适配。README记载批量重定向走cuda:0、target_fps未实际重采样、输出根四元数xyzw；与当前WM的wxyz和50Hz输入必须显式对齐，本次未重定向。

SDK提供高层速度与/aima/hal/joint/command关节接口；Python关节示例发送位置及stiffness/damping，不是运动策略，也不证明ONNX已真机验证。

## 原训练现场

21:36:05北京时间，通过原生hp_han Windows到同一25386，核原boot/GPU/唯一根与controller362416/start38298223、worker362421/start38298276均相同。原G1为RUNNING、110338/160000轮，五optimizer分别WM1103380/motion108338/critic2166760/KP108338/KPcritic2166760；非有限0/OOM0、GPU8779MiB，容器总19308711936B。40000/80000/96000检查点均274169803B，只stat未加载。

原观察句柄6378在上下文续接后返回Unknown process id；这不证明网络或服务器故障。已按原授权先登记新独立观察故障，再一次原生48707恢复，之前故障次数不重置。首个只读观察使用cgroup-v2路径失败，改读实际cgroup-v1后取得上述数值，未重启或改变训练。

冻结训练代码71859ca8abe238477aa18245a31d81204da9197f保持，尚未做正式物理评测，不以optimizer继续更新宣称步态改善。目标未达、user_accepted_checkpoint=null，本次不改变应用goal状态或其他自动任务，不启动agent。

## 证据、复查与下一步

小参考放主根被忽略的data/references/primebot_org_review_20261101；25份内容共256320B逐大小/Git blob SHA1一致并登记SHA256。五库完整Git树均truncated=false。初始raw GitHub TLS reset保留为网络失败，后通过GitHub Git blob API成功取得源码，未执行第三方代码。SDK/独立资产API license元数据为空，不将其源码或资产转载入项目Git。

[完整核查清单](../data/manifests/primebot_q1_ecosystem_review_20261101.json)给出固定commit、25份内容身份、30项资产身份和公开权重/动作身份。可用GitHub /repos/PrimebotTech/<repo>/git/trees/<commit>?recursive=1重新查询，并按Git blob API下载需要的小源码；不要以重新下载大权重作身份核查。当前阶段发布与三端小记录镜像状态以[publication旁收据](../logs/session_records/primebot_q1_ecosystem_publication_20261101.json)为准，不由本段预称完成。

原G1继续到160000轮并完成原版评估/本地原速视频。其释放GPU后，Q1先复核公开PPO候选的历史配置/导出和模型，建立已公开checkpoint的独立回放基线；SuperMimic则按力矩XML/完整配方独立复现。MorphMimic的多本体共享验收仍单列，Q1单机器人PPO或WM结果不替代它。
