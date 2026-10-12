# A3 修正精度后的更新与梯度排查

记录日期：2026-10-12。独立实验 A3_ZHY_GRADIENT_20261012_E11。状态：7项CPU检查、真实FP32零更新梯度与独立CPU概率/损失核验通过；R12保存点模块变化已审计，策略退化唯一原因尚未确定。

## 目标与范围

用户要求继续查，并记住分析此前问题。本轮针对E09修正计算一致性后，E10/R12两更新仍使四诊断动作平均关节误差增加4.63%的原因。先审计已有保存点，再在真实FP32入口完成一次16环境、24步、四批的零更新梯度诊断。使用原官方初始化、数据和奖励，分离policy、value、entropy和aux梯度。

工作在ZHY `/opt/data/private/hechengzhu/ZHY`；原hp3090只保存/同步代码和记录。旧停止账本不改写，不重复E09/E10、不追加长训；本轮不执行optimizer.step，模型与原始张量新机LOCAL_ONLY，无真机操作。

## 用户问题与证据跟踪

| 之前的问题 | 当前证据与判断 | 本轮要补什么 |
| --- | --- | --- |
| 是否硬件导致A3问题 | 原机普通CUDA/YOLO也出现新boot及现场供电保护提示；新机短训练可执行。旧机重启与当前策略退化分别记录，尚未定位旧机具体器件 | 继续原机停载，只观察新机健康；不把数值错误当成旧机重启的解释 |
| 旧机是否也遇到零更新偏差 | 10/10 E08梯度记录初始ratio约0.53–1.96；10/12新机E04/E05也失败。并非只在迁移后出现 | 当年未捕获真实精度，不能宣称已独立证明旧机同一BF16原因 |
| 参数未更新为何内部结果已变 | E08真实Actor权重/输入/buffer稳定；采样FP32、真实训练BF16。E09统一精度后通过 | 区分参数、输出和Critic归一化buffer，记录哪些状态能在无optimizer时变化 |
| 训练是否按预期优化 | E10真实8次optimizer和保存重载通过，但四动作效果门槛失败 | 拆损失和梯度、实际更新模块；确认目标与评测指标的关系 |
| 官方是否说明过 | 已有官方合同/issue核验：有不收敛讨论，所查公开内容未直接说明本地零更新异常；selected20不是035训练语料 | 复用已有来源记录，不把升级vendor当作未经证据支持的修复 |
| 能否用代码修到预期 | 初始概率一致性已修复，效果改善尚未实现 | 本轮只提供下一项单变量试验的依据，不把“不崩”“小漂移”当作更好 |
| 是否一直绕着更深的问题 | 工作理解：需要一条能判断每次实验是否缩小未知的可靠训练/验收链；尚待用户确认 | 每项诊断写清回答了什么、仍缺什么、下一步为何这样选 |

## 关键决定与实现

使用已有`GradientProbeTrainer`官方损失分解与禁止optimizer预钩子，叠加`FullPrecisionContractMixin`；新增`ZHYGradientTrainer`保存四个真实前向、Critic buffer及精度。没有重写训练损失或修改固定vendor。

新入口`script/a3/zhy_gradient.py`先核验R12初始/更新1/更新2保存点，随后一次真实零更新诊断；独立CPU复算概率和损失，审计R12每组真实参数/buffer变化。梯度范数是运行中的诊断证据，CPU复算不伪称独立重算了完整梯度。

CPU90/GPU85摄氏度连续两次5秒保护、保留4GiB内存、同boot、遥测/OOM/关键内核异常停止沿用；无自动恢复、旧目录重置或第二次启动。

## 版本、命令与产物

基于main `7d9cd3596de0a27a5f1e3796e5c87c4f934cfdde`；固定vendor `fe6868ba37034f89b912f0fb851bce19f120266d`。代码/输入SHA在登记时固定；启动前scoped检查、提交推送并核验。

```bash
cd /opt/data/private/hechengzhu/ZHY
PYTHONPATH=.:script/vendor/sonic_for_a3 OMP_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m unittest script.a3.test_zhy_gradient -v
data/environments/a3-sonic/bin/python -m script.a3.zhy_gradient register
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_gradient run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_gradient verify
```

实验`data/experiments/a3_zhy_gradient_20261012_E11/`；日志`logs/a3_zhy_gradient_20261012_E11/`；清单`data/manifests/a3_zhy_gradient_20261012_E11.json`。实际运行版本`cf5383f6c3546b3bc7107148c1ee0aa9eaa75486`，启动前main推送及远端核验一致。Mac只留小型代码/报告/清单。

## 结果、边界与下一步

7项CPU检查通过，覆盖独立损失的PPO/value裁剪和padding、辅助系数、参数/buffer区分、状态形状不匹配拒绝、禁止optimizer、旧目录重置拒绝以及三保存点映射/列表覆盖。保存点std变化分别报告从初始加载的总差异和已知限幅后的剩余差异，避免把限幅误当成优化器更新。

登记前检查发现：目标机完整receipt的`checkpoints`为按step索引的映射，Git小型manifest为列表；首次读取将映射键当记录而失败，尚未创建E11目录或启动工作负载。新增结构化格式兼容和覆盖测试，保留这一预启动失败记录，不重置任何已启动实验。

北京时间18:11:48–18:12:58完成保存点审计及一次新回放；真实梯度分支18:12:06开始，0optimizer、global_step0、空optimizer state，梯度前后参数SHA相同。24次采样和4次训练Actor调用均FP32。384帧覆盖16环境，CPU独立复算四项loss、概率和保存点参数/buffer分类通过。四批ratio范围0.99998665–1.00001717，mean RMSE3.52e-7–4.55e-7，未再次出现已修正的初始偏差。

| 批次 | decoder PPO/aux梯度范数 | encoder PPO/aux梯度范数 | Critic初始价值解释度 |
| --- | --- | --- | --- |
| 0 | 164.254 / 0.03500 | 40.802 / 0.13001 | 0.64387 |
| 1 | 52.656 / 0.05950 | 13.467 / 0.05781 | 0.79534 |
| 2 | 17.159 / 0.01435 | 4.627 / 0.04018 | -0.77873 |
| 3 | 23.099 / 0.05714 | 7.928 / 0.05368 | 0.78738 |

本批原始梯度不支持aux主导actor：decoder主梯度约aux的404–4693倍，encoder约115–314倍。value梯度在所有actor组均为0；entropy仅作用std，decoder/encoder均为0。Critic第2批估计较差但其余3批为正，不能说全部Critic失效。原始梯度范数不是Adam后的步长，也不是R12历史所有更新的因果证明。

Actor无normalizer；Critic有。加载至采样结束buffer未变化；四次训练前向的running_mean/running_var发生小幅变化，Actor权重和buffer无对应变化。这属于需要单列的运行统计，不是optimizer提前更新；不能仅凭存在统计变化就认定是退化原因。

R12初始网络精确等于官方，保存点optimizer参数计数0/4/8。实际改变了59个参数张量：20个g1/a3_fast encoder、24个g1_dyn/g1_kin decoder、14个Critic与1个std。更新2相对初始，encoder/decoder/Critic参数L2变化0.027101/0.032138/0.034687，相对各组原值约0.0281%/0.0163%/0.0270%；std扣除预期限幅后L2变化5.2903e-5。网络变化虽小但范围广，不能由参数变化小推出闭环不受影响。

顺带更正E10记录：实际8次应用LR为3e-6、3e-6、3e-6、2e-6、3e-6、4.5e-6、3e-6、2e-6；2e-6是初始和末次值，不是全程固定。KL控制下的合法变化不等于此前已修复的scheduler覆盖；原始E10审计记录未改写。

16健康样本同boot，CPU最高40/GPU34摄氏度、采样功率最高130.23W、可用估计最低5.07GiB、OOM0、无新关键内核错误/遥测失败。采样18:11:04–18:12:57，最长43.36秒间隔位于登记到运行前；不宣称全时连续覆盖。计算已结束，GPU只读查询无计算进程。

此前源码确认当前奖励包含身体/根跟踪、速度和动作平滑等项，不直接等于MuJoCo四动作关节RMSE。官方selected20不是035训练语料，而这里每回放只有384帧；小样本/目标不匹配与全Actor修改值得优先检验，但尚未通过单变量证明是唯一原因。

同一批零更新梯度只能说明当时目标推动哪些模块；完整梯度几何来自运行中的计算，独立CPU核验只复算损失/概率和保存点差异，未伪称独立重算所有梯度。不能证明R12所有更新的历史原因、每轮必改善、真实遥操/机器人效果或数据泛化。

下一项优先固定E11同一批观测，重放已有初始/更新1/更新2Actor，并分别只替换encoder或decoder，记录动作漂移与FSQ token变化。先定位已保存策略哪部分放大差异，再选择冻结模块或参考策略约束的单变量训练；不直接加长训练或声称奖励是根因。

## 保存与恢复

本README、索引/current state、代码和小型清单已保存，结果按main scoped审计提交推送并核对远端。原始rollout/forward张量与模型新机LOCAL_ONLY，没有网盘上传/往返验证。恢复先读本文件和E11 state/log，状态VERIFIED；不重复已有实验。
