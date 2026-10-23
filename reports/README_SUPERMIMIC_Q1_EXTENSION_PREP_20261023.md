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
