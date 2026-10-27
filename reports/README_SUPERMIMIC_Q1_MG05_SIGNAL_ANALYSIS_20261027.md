# MG05：离线信号分析与物理诊断边界

记录日期：2026-10-27，北京时间。BD02：上游与离线分析完成，候选构建源码和离线检查已准备，原生接触复核、真实runtime契约和训练未完成。用户确认25245服务器已停止，等待新入口，期间先做可行的分析；不再连接旧入口、不重启或训练。MG05仍有未完成项；本次get_goal实际标签为paused，未创建/恢复/结算旧目标。

## 目标与范围

承接[MG05](README_SUPERMIMIC_Q1_MG05_DIAGNOSE_TRACKING_20261027.md)，区分上肢目标、真实接触/滑移、模型动作导数和上游边界。保留RF09原runtime、权重、参考和结果，不添加数据准入条件，不以新reward代替物理效果。权威Git为`/tmp/yuanqi-supermimic-q1-20261023`的Q1分支；服务器唯一根`/opt/data/private/ZHY`不变，新入口等待用户提供。尚无新训练，下一轮仍限4096环境、实际训练最多3600秒。

## 上肢主要是持续姿态偏差

新增`script/project_tools/analyze_q1_hp_offline_signals.py`，只读本机BD01完整JSON，0策略前向/积分/optimizer。输入1358975字节、SHA`e59a4bda110f08cc18fe4634cf72121c8d277ae4f3375971fa6d12b32a5c0dd7`；输出[离线分析清单](../data/manifests/supermimic_q1_MG05_offline_signals_20261027.json)。

按每片每轴`MSE = bias² + variance`，上肢8轴的bias²之和除以相应MSE之和；每片/轴等权，使用实际执行前缀。这里是每片内部的时间平均误差，不表示所有动作都偏向同一方向。

| 分组 | 片数 | 持续偏差占平方误差 | 策略目标相对参考RMSE | 关节相对策略目标RMSE |
| --- | ---: | ---: | ---: | ---: |
| 训练全体 | 300 | 74.24% | 0.93282rad | 0.08103rad |
| 训练手势 | 60 | 76.67% | 1.07881rad | 0.06506rad |
| 验证手势 | 8 | 81.72% | 1.18438rad | 0.06185rad |

验证手势8轴平均执行范围1.16863rad，参考0.90792rad，不宜笼统解释为“上肢幅度太小”。左/右肩pitch平均RMSE分别1.63769/1.64320rad、片内绝对bias均值1.54212/1.50813rad，而这两轴参考超软限位比例分别0/1.14%。优先解释策略为何给错姿态目标，不把主要误差简单归为肩pitch参考超限。

全上肢参考超软限位平均比例仍有7.12%，不能证明所有重定向动力学正确。PD目标承担动态补偿，不必等于参考；目标执行误差小也不证明逐轴带宽或力矩储备充足。

## 上游核查与采样修复准备

唯一子agent已完成[双向核查](README_SUPERMIMICHP_Q1_UPDATE_20261027.md)，未再开其他agent。上游`aa625adf04f97ad8bffe4ffe5ca5db2da84e6986`四次更新主要归档工具/资产/结果；learner、MDP、MotionLib、manager核心blob与固定`310244b...`相同。精确100/15名单现在可读；新v4是历史LAFAN1配置，不是新的SOMA学习公式。

已用实际Torch CPU、函数AST和330个CSV行数确认：原`ceil(length/50)`在长度`50n+1`时多建零宽尾bin，采样到末帧后奖励读`f+1`跨片。train15/300片、15/1710bin，初始均匀概率0.8772%；validation1/30片、1/173bin。本库异常是跨片，未发现全局数组越界。自适应采样23.24%是人为负回报敏感度情景，不能当作RF09实测比例，旧checkpoint未保存bin回报权重。

本地`script/supermimic_q1_hp/transition_bins.py`只对`[clip_start, clip_end)`合法转移起点建bin；末帧frame_bin归最后合法bin供记账。4测试通过，覆盖长度2至402、50倍数/多一帧、多片及其他bin宽，检验每个合法起点恰好出现一次且`f+1`不跨片。本次新增`tracking_motion.py`候选overlay接入此逻辑；**服务器实际runtime尚未生成或部署**，仍待真实MotionLib/command reset验证。不宣称训练端已修好，不删除动作或改时长。

## 独立物理探针的结果与限制

停服前`diagnose_q1_hp_physics.py`的V01因FK不包含world的查表错误在积分前失败；修正后V02退出0。服务器产物：

```text
data/supermimic_for_q1/TRACKING_BD02_PHYSICS_V01_20261027/
data/supermimic_for_q1/TRACKING_BD02_PHYSICS_V02_20261027/result.json
data/supermimic_for_q1/TRACKING_BD02_PHYSICS_V02_20261027/contact_trace.npz
logs/TRACKING_BD02_PHYSICS_V02_20261027.log
```

V02原脚本SHA`cb3cde3dbd5846da1e8d2cda8872ae31335038f8d115a5c57550dddc44e730b1`。以下是已返回工具输出的摘要，完整result因停服尚未取回核SHA，不能称三端证据交付完成。当前脚本只更正协议元数据，未重跑/覆盖旧输出。

4条固定参考、冻结策略、独立Warp world，每控制步4次5ms积分，20ms后额外forward刷新求解力。足地力大于5N判承重，使用受力加权接触点切向速度。两条普通走单支撑约48.4%/60.5%、支撑侧切换17/18次，承重接触点滑速均值约0.135/0.234m/s；两条手势基本双支撑。

**该探针没有精确复现原生330环境轨迹。** 两条走在此处执行320/210步，原RF09是291/142步终止；共同状态也明显不同。不能因此说策略恢复了行走、用这些力解释原摔倒，或当成新训练收益。

源码显示原生eval显式写机器人/地面滑动摩擦为1，独立探针使用XML默认摩擦而模型context填1；尚未比对XML实际数值，不能断言摩擦值必不同。额外forward、模型组装/solver、reset及读回时序也需核对。下一步应在原生backend原阶段读力/子步冲量，避免额外forward或auto-reset改变被测轨迹；先核同初态/动作/状态，再归因。

后续原生backend检查在`worker.prepare`阶段139段错误，日志`logs/TRACKING_BD02_BACKEND_INSPECT_20261027.log`；未获得backend对象清单，原生接触未验证。停服由用户确认，不将它归因于段错误/OOM。新入口恢复后只读此日志及项目自身可能的core，不自动重试旧命令或动他人文件。

### 模型动作方向

12个固定状态、22轴中心差分、扰动0.02/0.08、后续动作固定。0.08扰动下状态间余弦均值摘要：

| 指标 | H1 | H4 | H8 |
| --- | ---: | ---: | ---: |
| 原tracking，去动作率/限位项 | 0.945 | 0.556 | 0.540 |
| 负上肢关节平方误差 | 0.995 | 0.991 | 0.929 |
| 负腿关节平方误差 | 0.954 | 0.853 | 0.783 |

支持先看上肢表示/奖励为何没有用好局部方向；不证明H32闭环策略/critic梯度可靠，也不是按接触相位分层的生产环境验证。较长的综合方向比上肢局部方向弱，不能仅凭H1或模型loss宣布优化方向已通过。

**精度勘误：** V02的scope写“fp32 model arithmetic”不准确。`load_policy`会按checkpoint设置共享`networks.AMP_DTYPE`，RF09为BF16；状态FP32不等于MLP为纯FP32。现版会记录真实state/AMP dtype和XML摩擦；旧V02收据保留，以本条更正，未额外跑FP32或H32。

## 下一轮候选与控制变量

先修已证采样边界，再验证以下一个组合；尚未选成“已证最优”的配置：

1. Actor历史加入当前`q_ref[f]-q_actual`22维，obs80到102。原goal已有未来关节参考，新增是直接表示当前误差，并非原先完全缺姿态信息。采集、replay、imagine、评测及checkpoint语义必须一致。
2. 候选tracking最高值仍5：身体位置/朝向各1，身体线/角速度各0.5，根位置/朝向各0.5，上肢关节/腿腰关节各0.5。身体朝向std候选0.8、关节上/下肢0.8/0.5rad，先在原状态与物理微扰验证敏感度；不能由组平均RMSE直接重建非线性reward。
3. 首次对照保留原`-0.1*sum((a-prev)^2)*0.02`及控制参数，避免同时显著减弱平滑约束。改物理目标差mean却保留0.1系数，会弱约68至181倍。reset首步previous action归零另审计，未验证前不改reset语义。
4. 末尾padding仅对原始future超过clip_end的速度考虑置零，真实末帧速度保留；先量化影响，不宣称其为整段倒地根因。

数学形状例子：等效角误差1rad时，指数reward对误差的斜率绝对值在std0.4/0.8分别0.0241/0.6550；0.5rad时分别1.3101/1.0572。放宽带宽增加大误差区信号，也可能减弱小误差区约束，不是越宽越正确。这里不是把上肢关节RMSE等同身体朝向误差。

不直接抄X2 PD、不默认追加数据/时长、不因H8较弱就无验证改H32。恢复后按原生诊断选择措施，不把所有待证想法一起塞入训练。

## 离线候选实现与检查

新增`script/supermimic_q1_hp/build_tracking_candidate.py`，从已核验的固定HP/Q1 runtime生成独立版本，拒绝覆盖原runtime或现有输出。保留LICENSE/NOTICE、学习器和模型字节；原Q1 MDP另存为`mdp_base.py`，仅用AST在分项字典暴露已经计算的线/角速度误差，原公式AST保持。`tracking_mdp.py`和`tracking_motion.py`分别提供候选观测/奖励和合法起点bin，未修改RF09实际runtime、权重或参考。

构建器保留两个选项：`sampler_only`用于不变观测/奖励的契约对照，`joint_tracking`用于前述候选组合。这是代码/零优化验证选项，不授权第二次正式训练。当前误差观测的噪声与原关节观测使用同一传感器扰动的相反数，避免两字段描述互相矛盾的关节测量。`worker.py`契约可识别80/102维，根XY仍核78:80，新增误差核80:102；真实物理检查尚未运行。

2026-10-27 11:53，本机标准库测试共16项通过：12项构建/AST/语义测试，另4项采样边界测试。包括源文件/学习器/许可证保留、独立目标拒绝、路径重定位、输入配置不被修改、身份错误在创建输出前失败、80到102维schema，以及变换只增加两项诊断字段。第一次测试暴露目录别名重定位缺失，已修复并加入实际软链接回归测试；最终16项全过，没有伪造Torch执行。实际固定X2源码经原Q1Dimensions变换后，再做候选AST变换，删除新增字典字段即可精确恢复原AST；六份候选Python源码编译检查及`git diff --check`通过。

新增`verify_tracking_candidate.py`供恢复后使用已有服务器Python做最多4片的CPU张量契约：参考/goal不变、合法bin、批量与历史噪声一致性、观测导数、奖励分解/最大值/关节项方向、旧checkpoint输入schema拒绝。当前仅其CLI和语法通过，**实际Torch契约尚未运行**；即使之后CPU契约通过，也不能替代原生command/reset、replay/imagine闭环、真实接触和性能验证。本机不安装Torch/MuJoCo大依赖。

复现本机已完成检查：

```sh
cd script/supermimic_q1_hp
python3 -m unittest -v test_tracking_candidate.py test_transition_bins.py
python3 build_tracking_candidate.py --help
python3 verify_tracking_candidate.py --help
```

小收据见[候选离线验证](../data/manifests/supermimic_q1_MG05_candidate_offline_20261027.json)。这些测试仅支持候选可以独立构建且部分源码契约正确，不支持“跟踪已经改善”或“腿部原因已经查清”。

## 对照与恢复顺序

首选从零新训一次4096环境、最多3600秒，保留WM1024x10、actor1024/H32、critic5x4和UTD0.4。4096是环境数，不能说actor batch也4096。RF09五小时final保留强基线；另预选`ckpt_003500.pt`作近一小时比较：本机CSV为3500轮、wall_h0.951184（3424.26秒）、36102144转移。checkpoint实存/内部时长待恢复核验，CSV仅用于预选；新旧预算差如实列明，同seed且输入维度不同也不能声称初始化完全相同。

固定300/30库、普通走/原地抬手等展示清单，主要看真实关节/腕误差、姿态bias、根移动/漂移、完成/跌倒和同后端真实接触/滑移；同时给共同前缀及全片。新旧reward分别在同轨迹上计算，不拿不同口径reward判优。多项组合改变不能单因素归因，无新数据/性能硬门。六格PNG/PDF/CSV及至少30秒多类本地MP4照常交付，不为图像加工拖延数值结果。

## 命令、验证与保存

Mac实际执行`python3 -m unittest -v test_transition_bins.py`（目录`script/supermimic_q1_hp`），4项通过；4份新增/修正Python源码AST解析与`git diff --check`通过。离线重算命令：

```sh
python3 script/project_tools/analyze_q1_hp_offline_signals.py \
  --diagnosis /Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF09_SOMA300_HP_E4096_T05H_20261027/RF09_BD01_SAVED_TRACE_20261027.json \
  --output data/manifests/supermimic_q1_MG05_offline_signals_20261027.json
```

输出已存在，恢复只读。本机没有Torch/MuJoCo，未安装大依赖；真实/想象契约、候选训练均未运行。重要记录/源码显式纳入Q1，payload audit后提交/push并核远端，Mac小文件核大小SHA；服务器同步等待新入口，不称三端完成，未宣称网盘备份。无关`download_bones.py`未动。

下一步：新入口到达后核共享根/资源/进程身份，取回BD02收据和原生检查日志；补原生接触、H32/候选方向及契约验证后，才执行MG05唯一一小时训练。目前A部分完成、B部分准备、C至E待，MG05不重建、不标完成。
