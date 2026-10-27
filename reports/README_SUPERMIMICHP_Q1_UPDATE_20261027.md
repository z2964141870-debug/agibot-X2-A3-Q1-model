# SuperMimicHP 更新与 Q1 双向核查

2026-10-27 10:42，北京时间。用户本轮明确授权唯一子 agent 检查师兄仓库；本记录只覆盖源码审查和 CPU 小诊断，未启动训练、未访问师兄容器、未修改冻结来源。主代理负责接触/世界模型方向实测和最终方案。

## 已确认的实质问题

**P1：训练起点可能采到片段最后一帧，下一步奖励跨入另一动作。** 上游当前和本项目 RF09 固定实现均存在此边界。`motion.py:121-134` 按 `ceil(clip_len/50)` 建立 bin；`manager_terms.py:55-69` 在 `[bin_start, min(bin_start+50, clip_end))` 采样，但 `clip_end` 为包含式末帧。当 `clip_len % 50 == 1` 时，最后一个 bin 的起止相同，采样结果必为末帧。`supermimic_reward:515-522` 与 `supermimic_termination:525-527` 先取 `f+1`，而 `supermimic_clip_end:535-538` 只是同一步判断超时，不能使跨动作参考变正确。最后一条动作触发时还可能全局越界。

在用户 25245 容器使用现有 Q1 `manager_terms.py` 的三个实际函数 AST、真实 Torch CPU 与强制逐 bin RNG，逐个读取 AP02 的 330 个 CSV 行数后复现：

| 数据 | 片数 | 总 bin | 零宽尾 bin / 涉及片 | 初始均匀 bin 概率 |
| --- | ---: | ---: | ---: | ---: |
| AP02 train | 300 | 1710 | 15 / 15，即 5% 的片 | 0.8772% |
| AP02 validation | 30 | 173 | 1 / 1，即 3.33% 的片 | 0.5780% |

全部 CSV 行数与 manifest 的 `frames_30hz` 一致；长度按 MotionLib 实际公式 `floor(((rows-1)/30)/.02)+1` 计算。训练片包括 `salute_001__A070`、`chefs_kiss_R_001__A071`、`yawn_R_003__A141` 和若干走/跑/转/跳；本库这些 bin 未落在全局最后一片，所以复现是跨下一 clip，而非数组越界。验证片为 `rubbing_eyes_fist_002__A142`，正常验证从首帧开始，不受训练随机起点机制同等影响。

该问题还可能被自适应采样放大。用同一实际 `update_start_bin_returns` 和 `start_bin_probabilities`，令每个异常 train bin 仅完成一次 1 步、归一回报为 -1 的 timeout，合计采样概率即从 0.8772% 变为 23.2413%。这是控制情景，不是 RF09 实测采样占比；RF09 checkpoint 不保存 `bin_G/bin_w_avg`，不能追认历史占比。固定源码 `learner.py:534-556` 的 checkpoint 字段可证这一保存范围。

建议新版本按有效转移数建立 `ceil((clip_len-1)/50)` 个 bin，保证所有随机起点满足 `f < clip_end`，同时覆盖末帧的 `frame_bin` 所属和合法首末边界。若只在采样处夹到 `end-1`，虽能避免跨片，仍保留重复尾 bin 的采样偏置。此修复不删除动作、不改变时长、不添加数据入选门；旧 RF09 冻结实现和结果应保留。主代理应在独立 runtime 落地并做多片边界与真实 command 验证后训练。

## 上游到底更新了什么

本次 `git ls-remote --symref` 实际核得默认分支 `feat/x2-embodiment`，HEAD 为 `aa625adf04f97ad8bffe4ffe5ca5db2da84e6986`。较本项目固定 `310244b73e45775744a75b789eb25a6c9c5a81c9` 新增四个提交，时间均为 2026-10-26：

| 提交 | 内容 |
| --- | --- |
| `d08bd58006b1cf644b95f314ecc83fc166ec5aae` | SOMA pilot 选择及来源 manifest 归档 |
| `02937c48729adc9120c9c452582b7a012a3b31ff` | 工件跟踪/归档规则 |
| `d11821c325050b29fa024203117574c0c9475b32` | X2 训练、诊断、评测工具及模型资产归档 |
| `aa625adf04f97ad8bffe4ffe5ca5db2da84e6986` | 评测证据和工件引用同步 |

逐 blob 比较确认 `learner.py`、X2 `mdp.py`、`motion.py`、`manager_terms.py` 均未变；新 HEAD 没有新的策略观测、奖励或世界模型学习公式。新增 `mjwarp_v4.yaml` 和 `launch_x2_v4.py` 对应历史 **LAFAN1 40 片**，不是新的 SOMA 训练办法，不能把归档提交日期误解为效果刚改善。

新 `convert_gmr_x2_csv.py:44-108` 要求显式四元数/关节顺序，检查形状与有限值、拒绝大于 `1e-5 rad` 的限位越界，处理四元数归一化和相邻符号连续性。此差异修正的是 GMR PKL 导入的格式风险，没有新增脚底/根缩放算法；Q1 当前直接 SOMA 到 Q1，不经过该转换器。它的格式声明做法可借鉴，不能把 X2 的硬 QC 规则重新设为 Q1 数据准入门。

现在公开了师兄实际 100 train / 15 validation 名单及 source selection / retarget report / generation manifest；此前的“精确清单尚未取得”已过时。可用于同源命名对照和复核，不需要为此重新下载媒体或替换当前 300/30 库。

## 对当前 Q1 有价值的新证据

### 1. 负回合回报不一定是统计 bug

新 `tools/audit_soma_episode_stats.py:59-98` 独立累加训练噪声下每条转移、奖励分项与 done 回合。归档 `docs/assets/x2/soma_evaluation_20261025/raw/episode_statistics_audit.json` 记录 665 个回合，计账一致性通过、289 个负回合；平均单步归一奖励为 +0.77637，平均回合归一奖励为 -0.61114。首步均值 -1.02882，持续步均值 +0.80438；回合分项均值为 tracking +0.66505、action-rate -1.22003、joint-limit -0.05616。

`manager_terms.py:395` reset 将 `a_prev` 置零；关节状态却从动作任意参考帧重置，第一条控制指令可能需要明显偏离默认动作。`mdp.py:244-250` 的动作变化罚在该首步仍完整生效，尾部短回合会放大这类首步成本对回合平均与自适应 bin 回报的影响。上游示例最差回合仅 1 步但动作变化罚约 -6。

这不是本 Q1 已实测的分解，也不说明应简单去掉平滑惩罚。Q1 应记录首次动作、`a_prev`、回合长度、timeout/硬终止和分项，先修上述跨片 bug，再比较首步排除/重置动作一致性方案。不能只把 ep return 曲线抬高就称跟踪改善。

### 2. 现有上肢参考信息是间接的，当前关节误差可作为适配候选

X2 `MDP.obs:154-184` 和 Q1 适配目前显式包含当下自身关节、速度、根相对朝向及根 XY 误差；`MotionLib.goal:149-175` 含未来 `f+1..f+10` 的参考关节，而没有当前 `q_ref[f]-q_actual` 的独立直接字段。Critic 有这个误差（`mdp.py:320`），不能因此认为 actor 也直接获得。

但 actor 并非完全没有动作姿态信息：已有未来 q 和自身 q，可由网络学习对应关系。增加当前关节误差属于改善表示与优化条件的候选，不是已证明的信息不可观测 bug；需同步真实采集、replay 历史重建、imagine、评测、噪声/尺度与 checkpoint schema。上游此次没有发布该改法的训练收益。

### 3. 身体朝向大偏差时，原奖励梯度会很弱

`mdp.py:230-250` 先对 14 个 body 的平方位置/朝向误差求均值，再取指数，未含直接关节 tracking 项。大上肢误差可使整项朝向奖励变小，同时身体线/角速度和根奖励仍高。本项目 BD01 的手势朝向项只有满值约 2.6%，且策略 PD target 与参考偏差约 1.18 rad，实际关节相对其 target 仅约 0.062 rad，优先支持检查优化信号与目标表示。

可验证的候选是分开上/下肢 tracking，并给大误差仍有方向信号的关节或 body tracking 代价，保持总权重可解释；不能只调窄指数带宽，因为这会进一步压小大误差区域的梯度。PD target 还承担重力和动态补偿，不应强制其逐帧等于参考角。新旧 reward 的数值不可直接当效果优劣，需要同一物理轨迹协议、同一起点及未改的原始 tracking 指标。

### 4. Armature 不是整条关节负载的有效惯量

上游 `robot.py:7-31` 明确将各档 armature 称为反射转子惯量的估计，使用 `kp=armature*omega^2`、`kd=2*zeta*armature*omega`。本 Q1 `build_runtime.py:99-110` 沿用该启发式，而输入模型全部 armature=.004，故 22 轴都得到 kp15.791367/kd1.005310。这不等于已经做了厂家逐关节调参，也不保证整机各轴真实闭环均为 10Hz、阻尼比2。

MuJoCo 官方 XML reference 的 joint armature 定义是添加到广义惯量对角线的转子/反射惯量；position actuator 的 dampratio 则按有效惯量计算，包含质量矩阵的姿态依赖。可参考官方文档 `https://mujoco.readthedocs.io/en/latest/XMLreference.html` 的 joint/armature 与 actuator/position/dampratio。建议先从质量矩阵和几个参考姿态评估各轴有效惯量，再做小幅目标响应；不将 X2 值复制到 Q1，不据 armature 单一数字把所有失稳归因于 PD。

### 5. 接触诊断可以迁移，几何指标不能冒充实测接触

新 `tools/diagnose_soma_translation.py` 用足底材料点速度、不以水平低速预筛近地帧，并保留持续滑动时间段，适合移植到 Q1 碰撞模型。`tools/probe_soma_support.py:21-32` 用浮动基座 Newton-Euler 所需合力/合力矩求所需 CoP；这是参考动力学一致性的必要条件诊断，不是实际接触力，也不包含完整摩擦/限矩可行性证明。

归档 X2 对照承认部分参考 CoP 越界、近地滑动代理和留出窗口退化，零跌倒不能替代跟踪。其未新增真实接触力审计、未验证世界模型 action Jacobian 方向。主代理对 Q1 做真实 Warp 接触与小幅动作有限差分，正好补齐这两个证据缺口。

## 给下一轮的建议

1. 先修已证尾 bin 跨片 bug，并记录新训练的真实起点/片段/首步奖励统计；保留旧来源。
2. 用物理动作方向与上肢目标响应审计选择观测/奖励组合，不在未见方向证据前加大接触附近的模型梯度依赖。
3. 当前关节误差可作为一小时新试验候选；非饱和 tracking 项先用冻结轨迹检查梯度和旧 tracking 指标，再做闭环对照。
4. 对 PD 和重定向只采用 Q1 实测支持的修正，不把上游 LAFAN v4 归档或 X2 参数当作新的 Q1 已证方案。

## 候选方案的快速逻辑审查

主代理提出的候选为：增加 22 维当前关节参考误差（单帧 obs80 到 102）；身体/根 tracking 权重保留，身体线/角速度各从1降至.5，给 joint pose 总权重1（上肢8轴/腿腰14轴各半、std.8/.5）；朝向 std.4 到.8；修正 transition bins；仅将超出 clip_end 的未来 padding 速度置零、真实末帧速度保留。上述方案可以作为有界、从零的新 Q1 对照，但同时改变表示和奖励，最终不能把收益单独归因于某一项。

需要特别处理动作变化项的单位与强度。若从 `-.1 * sum((a-prev)^2)` 改成 `-.1 * mean((alpha*(a-prev))^2)`，同一单关节动作变化的惩罚比例是 `alpha^2/22`。实际 Q1 control.json 的 alpha 为 0.56993166（36Nm）或0.34829157（22Nm），因此新惩罚只有原来的1.476%或0.5514%，分别弱67.7倍/181.4倍。所有关节目标都跳0.1rad时，每步只扣0.00002，约为0.1理论最大奖励的0.02%。

这种变化不仅修正机器人尺度，也大幅移除了整个过程的平滑约束，并未消除 reset 首步从虚构 default target 到参考姿态的历史不一致。可比较物理目标差平方 sum（不除22），或为均值选择有明确物理尺度的系数；例如均值系数1仍比原式弱约6.8/18.1倍，不应仅凭更高 reward 选取。最终看目标差、真实关节变化、限矩、接触冲量与原始 tracking。

更完整的首步处理是将 reset 后 previous action 设为保持实际 q_reset 的归一目标，并同步 action term reset、观测与真实/imagine 的历史语义；这涉及契约变化，不能只改奖励一行。若本轮采用较弱平滑项作为受控候选，应明确其强度和局限，用实际物理短测决定是否合适。

本次无安装、无 X2 权重/媒体下载、无外部发信、无积分/优化器、未占 GPU。只 fetch 浅层提交和按需文本。首次直接调用训练 Python 导入 Torch 缺少 `libcudnn.so.9`；随后通过项目既有 `activate.py` 正常完成 CPU 探针，CUDA 仍未初始化。没有更换或修改环境。

源身份、逐片异常名单和控制情景数值见 `data/manifests/supermimichp_q1_review_20261027.json`。源码事实以固定 commit 的 `git show <commit>:<path>` 为准；上游结果仅为作者归档，未独立复跑 X2。主代理统一审阅、提交与三端同步本记录，本子 agent 未提交/push。
