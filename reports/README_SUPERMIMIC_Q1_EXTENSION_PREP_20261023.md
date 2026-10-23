# Q1独立长预算续训入口准备

记录日期：2026-10-23。阶段：SM-Q1-E01。状态：代码准备与初始化/拒绝控制通过；尚未登记或启动长预算训练。

## 目标与范围

用户持续目标包括延长训练或优化方案，继续SuperMimic for Q1。当前R02成对试验仍运行，不能据中途在线奖励选赢家。原Q1驱动只支持从零初始化；本阶段在独立`script/supermimic_q1_extension/`准备固定追加预算的入口，不修改正在使用的`script/supermimic_for_q1/`文件集合或内容。

## 完成内容与决定

- `initialization.py`只接纳完整pair的注册臂：pair终态/比较/计划相互绑定、父final及原生完成收据、保存审计、final整片动力学与独立轨迹审计、控制器成功事件均匹配。策略可用门未通过可作为继续学习的起点；执行/审计失败、STOPPED父轮或未完成pair拒绝接纳。
- `train.py`从父配置恢复同样Q1控制/摩擦/参考/网络/actor与WM batch，不开放同时修改这些变量；保留WM、actor、critic与三组Adam状态及return-bin统计。追加轮数显式指定，至少3000且为500整数倍；不设置无限运行或自动恢复。
- 原检查点没有replay/仿真器完整进程状态，故新阶段使用显式不同seed、fresh replay/episode；保留原128采集轮和50组模型预热设置。step0在重置新episode前保存父权重/Adam/bin原状态；不能声称逐步无缝续跑。
- 学习率从父累计采集轮数偏移后继续原cosine；新收据/指标中的step、env_steps、WM/actor计数是本阶段新增量，`total_collection_rounds`另记累计轮数。Adam实际计数继续累计。
- `audit.py`独立对父→新step0逐张量/优化器/采样统计精确相等、新final网络变化、有限值、实际Adam累计计数、固定新增预算、累计学习率口径、所有日志轮数和OOM计数核验。2/5更新次数用独立整数算术计算，不以worker打印计数为证明。

## 版本、命令与证据

Q1工作副本`/tmp/yuanqi-supermimic-q1-20261023`；服务器`huanyu:/opt/data/private/YuanQi_Q1`；Python`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。不安装新依赖，不新取第三方模型；初始化思路参考项目已有A3独立续训模块，本轮不启动A3。

```bash
# 已实际执行；本地无PyTorch，9项通过、1项native fixture跳过。
python3 -m unittest discover -s script/supermimic_q1_extension -p test_initialization.py -v
# 服务器同10项全部通过，CPU-only。
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $PY -m unittest discover -s script/supermimic_q1_extension -p test_initialization.py -v
```

服务器native测试耗时2.959秒/退出0：三个2×2 Linear/Adam玩具fixture各执行一次更新生成非零动量，随后恢复到不同初始化的模块/优化器并逐项精确比较，且拒绝篡改admission。共3次玩具Adam更新，不是Q1训练或策略推理。其余9项覆盖未完成/停止/非注册父轮、比较及引用文件篡改、无事件绑定、Q1配置/batch变化和JSON tuple/list一致性。日志`logs/supermimic_q1_extension_controls_20261023.log`，小副本`logs/session_records/supermimic_q1_extension_controls_20261023.log`，1367字节/SHA`7aec299274909805db4efa948c9666859f1263ec775c844159f203bacebb5473`。

另用R02A实际admission重建嵌套StateSpecCfg并构造默认宽度WM/actor/critic，原生退出0；参数分别8729081/8172598/193793，配置JSON往返相等。0物理步/策略调用/optimizer更新，无FK资产加载。收据`logs/session_records/supermimic_q1_extension_config_20261023.json`绑定该次worker源码SHA；后续只加独立audit整数计数检查，worker未再变更。仅构造成功，不是载入实际父权重或前后向/完整训练通过。

实际worker阴性控制：以当前尚无pair终态的目录请求追加6000轮，原生退出1，在创建output/加载Q1资产前准确因缺少pair完成收据拒绝；预定输出`E01_unfinished_pair_rejected_20261023`不存在。0物理/策略调用/optimizer；日志`logs/supermimic_q1_extension_expected_rejection_20261023.log`及其`logs/session_records/`小副本。6000是阴性测试参数，不是已选择或启动的长预算。

native测试前后容器31005589504/31004872704字节，GPU40–41摄氏度、OOM仍14。独立目录部署后原pair的`verify_frozen`再次通过，活跃PID/start_ticks与原登记相符；未改变正在运行的科学条件。

## 结果与边界

已经支持可审计的父状态继承及对应计数规则，尚无真实Q1完整续训生命周期或新audit正例，因为R02对照尚未完成。不能把10项fixture通过或全宽构造称为长预算训练/效果验证。当前不加载父中期检查点训练，不抢跑下一阶段。

真正执行前，先确认两臂、保存审计及step0/final整片评测与比较完成，按腿部/水平根误差、位移、完整率选择具体父臂；记录选择理由、固定追加预算、新seed、输入/发布版本、资源及输出目录。CLI为`$PY -u script/supermimic_q1_extension/train.py --root "$PWD" --pair <已完成pair> --parent-run <选定臂> --output <新目录> --steps <固定追加轮数> --seed <显式新种子> --code-commit <发布提交>`。该命令当前未执行；不能用占位参数启动。

结束时使用新`audit.py --run <新目录> --output <新审计JSON>`，再复用原`run_contact_corrected.py evaluate`及轨迹审计/分肢和根误差分析。原`audit_training.py`假设初始Adam为空，不适用于续训；原`analyze_learning.py`只支持3000轮日志，长预算须单独汇总，不能直接套用或修改冻结文件。

## 保存与恢复

本阶段代码、小日志与结论按Q1分支提交并推送，发布核验以实际Git输出为准。大工件保留服务器LOCAL_ONLY。恢复先读R02运行README和实时进程/status；E01只是准备，不能当作R03已登记/启动或目标完成。后续实际长预算阶段使用独立README/清单，不重写旧停止或R01结果。

## 追加独立审计控制

初版代码发布为`9f77d0f2769ff5fecf20890531a2a74cd9e49b13`，已独立核对Q1远端。继续审查时发现新audit只逐个核对声明的父文件SHA、未主动重建完整父谱系；现改为重新执行`validated_parent`并与manifest证据完全比较，缺少或改变父pair/事件/收据即拒绝。worker及当前R02核心未修改。

北京时间21:06服务器新11项控制全部通过，耗时2.754秒，原生退出0。新增用小型合成保存数组与预设计数覆盖独立audit正例；final actor Adam少1次、step0清空优化器状态均准确给失败门；删除父成功事件会直接拒绝。该合成审计fixture没有执行3000轮训练、物理或optimizer更新，不能当真实Q1续训/audit结果。原3次玩具Adam状态继承控制也随套件通过，不计为Q1更新。

日志`logs/supermimic_q1_extension_controls_E02_20261023.log`及Git小副本`logs/session_records/supermimic_q1_extension_controls_E02_20261023.log`，1548字节/SHA`b23db7b3c9027632e77d430490b0e9492aff796b5a11428854a71f06db9a0615`。控制前后容器30995984384/30996811776字节、GPU40摄氏度、OOM仍14；源文件在独立extension目录，R02训练继续存活。

本地无PyTorch，最新套件9项通过/2项native跳过；服务器11项无跳过。下一仍是等R02两臂最终物理与匹配比较，再登记固定长预算；不因为fixture正例通过提前启动。

## E03续训登记与完整收尾流程

新增同一独立目录下的`pipeline.py`，使用已有R02子进程执行/缓存建议/资源保护实现release、register和execute。它先验证已完成的匹配pair及选定父臂，固定代码、输入、父证据、追加轮数、新seed与选择理由；执行时同时占用pair和extension文件锁，按训练→续训保存审计→step0/final各三段完整动力学/轨迹审计/分肢诊断推进。源码文件集合与内容、Python、包版本和boot在各阶段核验。失败或信号保存终态，不重复启动，没有自动增加预算。

评测片段直接继承父最终评测中注册的完整列表。实际父manifest的`heldout_clips`为空，不能据该字段只评测两段训练动作；本流程仍含已复用的出拳开发片段，不将其改称独立测试。执行完成状态与原有跟随门分别保留，科学效果失败不伪装成工程执行失败或成功跟随。原`analyze_learning.py`的3000轮专用统计未套用到长预算；完整JSONL和分肢/位移结果仍保存。

本地新增5项控制实际通过，0.158秒；服务器确认5项全部通过，0.155秒、退出0：发布后源码改动拒绝，开发片段不遗漏，未完成pair在创建输出前拒绝，pair锁阻止并发启动，子进程退出7阻止后续评测并保存FAILED、再次执行拒绝且终态字节不变。它们只使用临时Git/小文件/短CPU子进程，0机器人物理/策略/optimizer更新；之前11项状态继承与审计控制的证据保持，不把两批检查称为一次新完整Q1训练。

首次服务器测试在SCP尚未完成时提前调用，只发现0项，日志`supermimic_q1_extension_pipeline_E03_20261023.log`保留，不计通过。确认传输完成及两端源码SHA相同后，新独立E03B测试显式断言`countTestCases()==5`，排除0项退出0被误记通过。有效日志`logs/supermimic_q1_extension_pipeline_E03B_20261023.log`及Git小副本`logs/session_records/`，914字节/SHA`08aa52ed10998c42b9409bd0f4f37a46e6ef92c3aeee1c1cb46d56896af68a70`。

实际服务器CLI阴性控制：`pipeline.py register`以当前缺少pair完成收据的R02和A父臂请求登记，原生退出1，明确缺少该pair的`receipt.json`；预定`data/supermimic_for_q1/E03_unfinished_pair_rejected_20261023`不存在。6000轮/seed20261024仅为阴性参数，尚非已选预算。小收据`logs/session_records/supermimic_q1_extension_pipeline_rejection_E03_20261023.json`为2521字节/SHA`766e16282b99f5de48e5ae4acc8534dafb3089ed53ffc4c363dfb0df52d4912e`，绑定入口SHA、原生退出、输出不存在、资源和仍运行的R02身份；对应原始log亦保存小副本。

22:01:46再次核验R02冻结源码/输入/运行时和controller/B的boot/PID/start_ticks及非僵尸状态通过。B为1350轮/21600交互/439次actor，GPU40–41摄氏度、实时容器31018258432字节、OOM仍14。本阶段没有启动真实Q1续训；base核心相对冻结`6b43cc0`的Git差异为零。

发布并独立核对GitHub Q1后，实际使用以下入口；`<...>`参数必须待匹配对照完成后按结果填写，不执行占位命令：

```bash
# Q1本地发布副本生成release；服务器按SHA核对部署。
python3 script/supermimic_q1_extension/pipeline.py release --root "$PWD" --output <新release.json>
# 服务器root为/opt/data/private/YuanQi_Q1；登记后执行一次。
$PY script/supermimic_q1_extension/pipeline.py register --root "$PWD" --release <release.json> --pair <已完成pair> --parent-run <选定父臂> --output <独立R03目录> --steps <固定追加轮数> --seed <新seed> --reason <选择依据>
$PY -u script/supermimic_q1_extension/pipeline.py execute --plan <独立R03目录/plan.json>
```

输出为R03目录下plan/started/status/events/receipt/summary，训练工件在其`run/`，评测在`eval_initial/`和`eval_final/`；stdout在`logs/supermimic_q1_<R03目录名>/`。真正注册/启动/完整生命周期仍未运行，后续必须独立记录实际预算、父臂、发布版本、保存审计和效果，不能把E03检查替代它们。
