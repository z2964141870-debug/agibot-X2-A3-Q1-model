# MorphMimic足量复现

北京时间2026-11-01，应用目标MORPHMIMIC_FAITHFUL01 ACTIVE。用户要求忠实复现MorphMimic，允许按硬件等比例缩小，接受8/16小时等充足训练时间。时间用于完成必要迭代，不以运行够小时数或报告完成替代效果验收。最新授权覆盖旧1200秒短预算和300秒physical_regression停训门；OOM、非有限值、原24/30GiB准入/容器保护、身份和共享GPU锁继续保留。

## 固定依据

固定官方仓库lzyang2000/morphmimic提交aa40fe46a7fa995ba7a7354ad065a2c8ad64a302；服务器已有183源码blob和原uv.lock环境，沿用，不重新升级。trainer.py SHA256为82519fcb992471eee910dcacf7f962a1e02174e4f7b32cd9d763105220993c67。README的实验命令显式40000轮；TrainCfg默认320000轮是另一个值，不能混称论文实验。main已有后续更改，尚未逐项审计，不静默切换版本。

上游README与固定源码位于data/references/morphmimic_review_20261029/source；来源为https://github.com/lzyang2000/morphmimic 与 https://lzyang2000.github.io/morphmimic/static/morphmimic.pdf 。跨本体为G1/X2 Ultra/T2各自物理环境、参考库、replay，统一29关节/14身体表征、125维形态描述、共享WM和策略并跨本体潜变量对齐。原实现每机器人一张GPU，当前一张24GiB 3090不直接等价。源码默认pool_norms=True与论文“按机器人统计”描述差异明确保留，不擅改后称原版。

## 比例候选

| 项目 | 官方实验 | 单卡首候选 |
| --- | --- | --- |
| 每机器人环境 | 4096 | 1024，待资源测试 |
| 总轮数 | 40000 | 160000 |
| WM批量/每轮批数 | 2048/10 | 512/10 |
| motion与KP想象批量 | 4096/4096 | 1024/1024 |
| WM-only轮数 | 500 | 2000 |
| 想象H/模型K | 32/8 | 32/8 |
| 三优化器LR | 2e-4 | 2e-4 |
| cosine衰减时钟 | 250000轮 | 1000000轮 |
| replay每环境容量 | 1000步 | 1000步 |
| UTD/gamma/lambda | .4/.99/.95 | 同上 |

网络宽度、原奖励、控制、随机执行器延迟、原BF16与compile设置、KP full接口、float16参考库尽量保留。1024放不下时备选512环境/320000轮，并记录真实原因。保留40000轮的迭代对齐检查点和最终采样量对齐检查点。单机器人每线训练采集409600000转移、WM训练819200000窗口、motion想象5177344000步；初始化和驻留replay缩小。更小batch及更多Adam更新不构成优化等价，必须同时列出更新次数。latent reference draw预算只在多本体对齐启用时适用。

结构化[比例计划](../data/manifests/morphmimic_faithful_plan_20261101.json)由script/morphmimic_faithful/scale_plan.py解析固定源码生成；7项CPU测试通过，0仿真/策略/优化。这些是候选，不是已资源通过或已启动的配方。

## 数据与执行

已有G1原始CSV1280条；旧试验只转换前128条，tiny50条/363.72秒。新阶段用固定版本原convert_seed、原name/feasibility过滤和seed0 build_subsets核实真实小时数，至少先复现官方1h reduced-data设置。仅G1有原始机器人CSV，X2/T2需要同批SOMA按上游retarget流程处理；同片交集与版本要核实，不把不同动作库算配对。

先资源测试G1完整WM/actor/critic/KP更新路径，再确定1024或512及吞吐估算，执行足量G1基线。完整跨本体单卡方案另行验证共享梯度及各本体独立状态；三个单体模型不等于unified，不跳过latent对齐后称跨本体复现。Q1 canonical兼容与接入列独立后续阶段，不承诺未见Q1零样本稳定控制。

正式运行需冻结源码/数据清单与准确配置、先发布三端核验；训练中记录迭代、采集/更新数、显存/容器峰值、非有限检查、曲线和检查点，保留足量训练机会，不使用旧300秒退步停训门。原评测、固定同片原速本地视频和完整失败结果用于判断，不以loss/reward或存活单独称稳定走路。

## 执行入口准备

script/morphmimic_faithful/worker.py直接调用原convert_all/build_subset和Trainer.train，训练仅外包迭代观察、优化器计数、非有限检查及额外40000轮保存，不替换算法。资源profile最多8轮、WM-only=0，刻意覆盖完整motion/KP/双critic/Adam分配，仅作资源测试，独立根/不续成正式训练。数据入口固定1280源、workers4、metadata v004，读取前后SHA不变和50Hz/29关节/数值有限检查；保留真实过滤结果，不自加物理数据门。主配置g1_data_20261101.json已准备。

supervise.py复用既有Q1Budget/StorageBudget/MemoryBoundBudget/guarded_run及同GPU锁；启动24GiB准入、运行30GiB守卫/OOM0/原boot/GPU/86度/100GiB磁盘余量，数据和profile最多1800秒、10秒回收；正式训练按迭代完成，仅1800秒无进度停，不套旧1200秒与步态门。每次新根/无自动故障重试；完整183上游blob在前后核验。12项CPU测试已通过，尚未物理/profile；数据/profile/训练三个阶段分别发布准确配方和收据。

新目标及旧交付提交9437f758bf9c968fc981830b19d64b597d00c590已独立核40位远端、9记录395881B Git/Mac/25386发布快照大小SHA一致。[发布旁收据](../data/manifests/morphmimic_faithful_goal_publication_20261101.json)。后续入口准备提交不改变旧底座训练冻结。

## 旧结果收尾

GOAL02底座301.575546秒/127轮physical_regression，停走倾斜RMS增31%-40%，脚底Z两走仅降2.35%/.59%，不达步态目标。12媒体与34保存审计通过、54工件28971941B Mac/25386大小SHA一致，原pipeline COMPLETED/进程退出；PPO从未启动，按最新目标停止推进。[交付清单](../data/manifests/supermimic_q1_gait_goal02_base_delivery_20261031.json)。旧G1 E64/600轮中500轮WM-only，仅100次actor是烟测，不是方法性能复现。

原生hp_han→Windows系统SSH→同一25386，唯一根/opt/data/private/ZHY，核原boot/GPU。无agent，不改其他自动任务和认证；权重/大NPZ仅SERVER_LOCAL_ONLY。发布以publication旁收据及独立40位远端为准，尚未核验不称已发布。新目标保持ACTIVE，不因旧交付、规划或等待结算。
