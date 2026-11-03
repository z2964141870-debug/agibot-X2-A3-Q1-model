# 共享策略原生推理与同物理Q1评估准备

日期：2026-11-04。状态：准备中；共享V03已因原内存守卫停止，正式共享训练及初末检查点尚未产生。本页不是训练完成或步态效果报告。

本轮补齐G1/Q1原生mjlab回放、CPU碰撞几何诊断、原速媒体和原生初末比较入口。本地五阶段55项检查中50通过、5项因本机无Torch/真实服务器资产跳过；既有4项已在前次服务器补验，本次真实G1/Q1运动学分别12.134/11.082秒通过。没有新增仿真积分、策略回放或优化，没有启动V04。[准备时快照](../logs/session_records/q1_native_evaluation_geometry_preparation_20261104.json)/[实际CPU收据](../logs/session_records/q1_native_geometry_runtime_v01_20261104.json)/[发布核验](../logs/session_records/q1_native_geometry_publication_20261104.json)。

## 目标与范围

为五阶段后半段准备Q1 motion直接对照、KP单列、完整片与共同有效前缀两种误差和明确的等曝光缺失。固定MorphMimic v0.1来源41c8382f127a6f5109bfe047445a9180f71f1fc0，保留实际Q1 V3的22/29关节mask、state269/context62/morph154及原生控制适配；不使用发布125维ONNX或旧单G1检查点冒充新共享结果。

## 已实现

inference.py在bootstrap之后重建原Policy和四个冻结normalizer，调用原imagination.act/act_human，不写训练replay或优化。加载时核真实训练purpose、完整40位提交、同一plan、双rank、配置、实际iteration/五optimizer计数/曝光、motion2及KP10版本、hist10、参考坐标系、154形态向量和缺失关节mask。拒绝profile权重、错误机器人、混用训练计划、旧goal版本和不完整来源。

q1_native_review.py使用同一Primebot CPU Q1Simulator、原V3资产/PD限矩/初始姿态速度/终止规则，加载新共享策略，转换原state79到原生269和执行命令记忆。核共同Q1姿态、WXYZ、逐帧长度及实际FK；目标从t+1预览10帧、末帧hold不跨片，人参考核原sidecar。支持三固定动作各seed0/1/2及全393各seed0，motion和human单列，保存失败原轨迹、完整片及有效前缀误差、脚底/承重滑移/姿态/上肢时序和原速视频。此是同物理Q1比较，不能冒充原mjlab场景等同或G1原生评估。

comparison.py要求完整且不重复的case集合，同片同种子同参考身份，失败片不能写完整片指标。逐对调用原metrics.compare，排除初始化和失败终止帧；先种子内平均再逐片等权。任一case无可比较前缀时整组共同前缀汇总留缺，不只选择存活片。完整片误差另外报告完成数/总数，不能用短前缀改善当全动作误差改善。

exposure_comparison明确Primebot早停102563840条转移不足410009600，输出UNAVAILABLE_EARLY_STOP_BEFORE_TARGET，不补训练、不编造等样本结论。Q1在共享训练中见过，因此后续所有结果均不能叫未见本体零样本。

## 验证与执行边界

本机标准库9测试通过：inference2、reference_window2、comparison5，覆盖错误计划/计数/robot/goal、跨片预览、缺case/重复/提前终止/混接口、短前缀缺失、逐片权重和等曝光缺失。五阶段整套本地44项中40通过、4项需服务器Torch而跳过。

2026-11-04 02:13:26服务器真实CPU补验通过，测试耗时12.383秒：固定v0.1实际小网络的motion和human输出与原policy_loader逐值完全相同；四normalizer及RNG保持、缺失关节输出为零、错误形态拒绝、CUDA未初始化。此次使用合成小检查点验证加载器，没有正式共享模型或新物理回放/优化。[原始CPU收据](../logs/session_records/q1_native_shared_inference_cpu_v01_20261104.json)，1288B/SHA256 851a34af249756998e25e0e8dcb322702e24488e330e6d6d5f8da5f34590059b。执行来源ccb90c1a2d940696f7f94cbcfde29d3cbb5795d3已独立40位远端和10份363132B三端大小SHA核同；此处只报告实际补验，不重复运行已有测试。

Q1实际评估CLI：在未来正式共享训练完成且初末检查点通过独立审计后，以对应e1024_train_plan.json或合格e512_train_plan.json传入--plan，使用checkpoint、固定Primebot源码和其最终checkpoint、admission_frozen.json，独立新服务器output，--interface motion或human，固定片用--render、全库用--library。当前没有可使用的新共享正式检查点，因此不执行回放。

## 未完成与下一步

native_review.py要求正式共享完成的pipeline、相同plan和初末CPU检查点审计，才允许载入策略和建立CUDA环境；拒绝profile、失败或未完成训练、单rank审计和错误robot/检查点。遵循原benchmark的frame0重置、首步frame1目标、原终止和clip_end；失败终止帧不进入有效误差，失败完整轨迹仍保留。motion/human分别运行，固定三片三种子及全部393片均支持，种子重复是无随机扰动的名义评估，不冒充独立扰动试验。

native_metrics.py按实际碰撞球、盒、椭球、胶囊、圆柱或mesh支持点计算最低脚底位置；使用实际足体线/角速度算该材料点滑速。原生接触载荷读取现有MJWarp求解器法向力，标明是控制步末端forward样本而非连续承重证明。原生误差为实际存在关节和13个骨盆相对跟踪身体的位置，不能与Primebot全身体口径直接混算；同物理Q1比较继续使用q1_native_review.py。

native_video.py按记录的原生scene和qpos以50fps输出同片实际/参考双视图，提前跌倒后保留失败hold标记，保存全帧解码收据和三张联系帧；scene和NPZ只留服务器。native_compare.py独立复算每个共同有效前缀和完整片误差，检查原始reference、人参考、mask、回放计数和来源一致性。G1初末改善不等于相对单体G1的共享收益，后者缺受控对照时明确不作结论。

新增边界测试覆盖未完成/错误来源审计、首步和末步跌倒、缺少clip_end、错误提前timeout、缺失关节mask、根平移不变性、六类碰撞支持、非有限几何、失败帧排除和保存指标与真实trace不一致。2026-11-04 02:57:32两型真实CPU补验完成，检查运动学qpos往返、scene保存/读取、碰撞脚底和材料点速度、原生benchmark别名及eval配置，八输入前后逐SHA同、CUDA设备屏蔽且Torch未初始化CUDA。后GPU仍1MiB/25度/recovery=None、cgroup OOM0。未验证GPU求解器读取或实际媒体，不称已经评估训练后的共享策略。

仍须完成实际正式共享检查点后的原生回放/媒体检验、训练曲线接入、串行评估controller和Mac完整媒体验收。不能因原生入口或CPU测试通过称五阶段已贯通。用户对V04追加一次同档验证的决定仍待，原训练失败不自动重启；等待期间继续本页缺项。

本轮执行源码aaaebdea135bd98e5a99b28a764158978253384f已独立40位远端核验，13份523258B在权威Git、Mac主根、25362逐大小SHA一致后才执行上述CPU补验。三份原始CPU证据5628B已取回并逐SHA核同。原生初末比较仅使用13个跟踪身体和现有关节，不能替代同物理Primebot比较或G1单体受控实验；后续结果结论继续保留这些限制。

补验与目标复核发布d837be049f6168ec35c0b23cc643e676bdb78ad5，独立40位远端已核；8份347871B在权威Git、Mac主根及25362逐大小SHA一致。[发布核验收据](../logs/session_records/q1_native_cpu_goal_publication_20261104.json)。未启动V04或正式共享训练。后续发布继续显式stage、payload audit、push Q1和独立远端/三端核验。权重/原始NPZ仅SERVER_LOCAL_ONLY，认证不保存。目标及十分钟监督active，用户采纳null。
