# Q1 R04想象时域匹配训练

记录日期：2026-10-24，北京时间。阶段：SM-Q1-R04/R04V2。状态：原R04在采集前FAILED并保留；独立R04V2已启动，A_H32真实采集/策略更新、step0独立继承审计及进程/冻结核验通过，B_H8与最终效果待完成。

## 10:58启动与11:03核验

R04V2源码发布`6db6bb92521cf606dc534ff0dc5ebc7dcb8f742f`已推送并独立核对。10:58:58派发原controller793182/start3610274454，A worker793185/start3610274671，boot `d994e71f-bae0-46ba-a5f7-6c2163771c6c`。11:03:26独立实核原进程分别S/R、PGID各自匹配；stage为A_H32_train，未出现终态或B run目录。

当次最新trainer记录300/3000轮、4800控制转移、累计15300轮、69组WM/19次actor更新；记录中全部标量有限，已越过预热。其数值来自trainer实时日志，不冒充最终保存参数/Adam计数审计，也不把在线reward/RMSE称整片效果。B将从同一R03父重新开始，同seed与预算，仅H8；只有A的完整保存/评测/独立审计通过后才启动。

真实A step0只读审计原生退出0，7门全通过。三网络、完整Adam（含矩/组参数/计数）和bin_return/bin_memory共8项与R03精确一致；初始本地计数0，manifest匹配admission/登记，日程offset与总轮数15000，张量有限。继承Adam为58980/5798/115960。step0为205235623字节/SHA `40eead5da33d78a0bc2d4dff698af50dea72bdc9ef05402567770a1f1c1a6bfa`；审计1864字节/SHA `e24da0ad591384526c255149f1ff7ef4cb975ee33d4e3ef2fbf790fe5b7bf7eb`，0新增物理/网络前向/optimizer。后续保存及效果仍须完整流水线审计。

计划冻结条件复核通过；11:03:26 GPU39摄氏度/空闲16671MiB、容器29363347456字节、OOM14未变、磁盘约1.998TB可用。保护线不放宽。原R04失败证据/4份源码保留；V2不是修改或重试旧计划。全部大原件LOCAL_ONLY，未下载权重/数组至Mac，未验证网盘备份。

小证据`data/manifests/supermimic_q1_R04V2_{release,plan,dispatch,started,initial_audit,startup}_20261024.json`；控制/登记/初态审计原日志`logs/session_records/supermimic_q1_R04V2_*`。运行原日志在服务器`logs/supermimic_q1_R04V2_H32_H8_matched3000_20261024/{controller,A_H32_train}.log`；运行工件`data/supermimic_for_q1/R04V2_H32_H8_matched3000_20261024`。新阶段记录按Q1审核/推送，提交身份由Git历史给出。

恢复时先读取status/receipt与原日志，再核对PID/start_ticks/boot和资源/冻结条件；原controller负责本预算及16阶段，不重新execute，不创建自动恢复或额外训练。initial审计已存在，不重复运行。最终若失败则保留终态；效果未达时不追加预算或改门槛。A3保持停止，当前可用移动跟随目标尚未达。

## 以下为R04V2准备登记

R04V2源码`6db6bb92521cf606dc534ff0dc5ebc7dcb8f742f`已推送并独立核对Q1；服务器9项全通过、无跳过，原生退出0/2.758秒。新register退出0，计划63694字节/SHA `49219c4ab37a31b66a0ca39345f441a4706d2b22dab22ede1dbb180680113c89`，68份发布文件核对通过，55core/6extension/4horizon及134输入冻结；16阶段、V2入口、原同父/预算/seed逐项核对。此时尚未启动；下一只启动该V2计划一次，旧R04不重试。

原`script/supermimic_q1_horizon/`4文件不编辑。独立复制到`script/supermimic_q1_horizon_v2/`，健康检查显式使用已持有的`continuation.runner`，不再解析同名train模块；原子JSON同样直接使用runner API。新增针对依赖导入完成后health(root, oom)的绑定控制。其余训练公式、合同、父权重、seed、预算与16阶段相同。

本地9项控制中8通过、1PyTorch fixture跳过，退出0；服务器必须9项/0跳过再登记新`R04V2_H32_H8_matched3000_20261024`。新增只读`script/supermimic_q1_analysis/audit_horizon_initial.py`，用于独立检查真实step0的8项父状态保留、7门及日程offset，无物理/网络前向/optimizer。该入口随V2 release冻结。

新计划release为`data/manifests/supermimic_q1_R04V2_release_20261024.json`，执行入口改为`script/supermimic_q1_horizon_v2/run_horizon_pair.py`；复用下文register参数但output改为R04V2、release改为R04V2。新注册/启动日志与小证据都使用R04V2前缀。旧R04不重启、不修改终态；启动新计划前重新核对真实资源及源SHA。

## 10:51原R04启动失败

登记记录`77ddf4493c28aa40dd03e4280994f86b32ee65c4`已推送并在启动前独立核对Q1。10:51:02派发controller792833/start3610226784，worker792837/start3610227019；10:51:09保存FAILED，A_H32_train原生退出1，后续15阶段未运行。没有run目录、admission或step0，0真实控制步/optimizer，不能称训练开始或训练效果失败。

实际错误为`TypeError: health() takes 0 positional arguments but 2 were given`：`train_horizon.py`使用`from train import health, atomic_json`，依赖导入会将core目录置前，命中同名core/train.py而非extension/train.py。这是本次入口错误，已有8项控制未覆盖模块解析后的健康检查调用；不是GPU/数据或SuperMimic方法失败。错误发生于parent元数据只读验证之后、资产/仿真构造之前。下次必须加针对实际health绑定的控制。

原目录`R04_H32_H8_matched3000_20261024`、其plan/receipt/events/日志与4份`script/supermimic_q1_horizon/`原码保持原样；小证据`data/manifests/supermimic_q1_R04_{receipt,dispatch,started}_20261024.json`与`logs/session_records/supermimic_q1_R04_failed_*`。下一在独立`supermimic_q1_horizon_v2/`修正显式runner API，登记新R04V2，保持原共同父模型、单变量和预算；不解锁或自动重试原失败计划。

## 10:46真实登记

源码发布`21c38b0202dead227cc04eac8cb733baf3d707ca`已推送并独立核对Q1。4份新增源码与release SHA部署一致；服务器8项控制原生退出0、无跳过，含三套微型网络/非零Adam状态及return bins精确继承。控制耗时2.857秒，不当作真实Q1训练。

10:46:06 register原生退出0；真实计划63317字节/SHA `74e715bedaf4c06deb953746cf5f6d37f74f7d1b2b48b15dd07be01bbc14c69f`。冻结55份core、6份原extension、4份新horizon源码、133项输入；release另包含两个analysis依赖，共67文件。实际parent、seed、3000轮/臂、H32/H8顺序、16阶段和三片评测覆盖已逐项核对。GPU38摄氏度/空闲17788MiB、容器25411137536字节/OOM14。尚无started或训练成绩；下一仅启动此固定计划一次。

小release/plan与服务器控制/登记原日志保存到`data/manifests/supermimic_q1_R04_{release,plan}_20261024.json`及`logs/session_records/supermimic_q1_R04_{controls,register}_20261024.log`。训练启动/继承与最终效果仍须分别验证。

## 目标与单变量

继续SuperMimic for Q1世界模型复现。D07发现走路长时根预测偏差，D08在4个已访问状态验证局部动作响应：走路根H8为16/16正向、误差中位0.4412；H32为11/16/0.7595。不是完整actor梯度证明，也不保证短时域更好。本次检验将policy imagination horizon从32改为8能否改善完整动力学跟随。

两个独立臂共同从已完成R03 final加载三网络、完整Adam和return bins，父checkpoint SHA `10d0838663b5d24215f493ce0738228597452e899b1a378df6d6b453b7e10ea7`。A_H32为匹配基线，B_H8为候选；仅policy_trainer.horizon不同。两臂使用相同fresh replay种子2026102404、相同输入/新接触/奖励/WM k8/网络/优化器/学习率日程、相同3000采集轮及更新频率，不与原R03不同预算直接作为因果对照。

每臂16环境、48000真实控制转移、1149组WM及1099次actor更新，训练后累计18000采集轮；以真实receipt为准。保留128采集轮和50组模型预热，学习率日程继承offset15000，不重置Adam。新replay与episodes不等于原进程精确恢复。H8每次想象计算与critic样本量自然更少，本试验匹配真实交互/更新次数，不匹配想象帧数、算力或墙钟时间。

## 实现与继承

新增独立`script/supermimic_q1_horizon/`，不编辑已冻结`supermimic_for_q1/`或`supermimic_q1_extension/`。旧续训入口只接受已完成R02 pair，且其父args结构不适用于R03；新入口明确验证R03的原终态、8个成功事件、完整summary重算、保存及匹配评测审计，再登记共同父模型。

`horizon_contract.py`复用原严格张量/Adam加载器，只对已登记H32/H8规范化horizon后核对完整合同，其余字段仍拒绝变化。进程内临时绑定校验器、finally恢复；新绑定/源码身份写manifest。证据字段兼容别名明确记录`pair_receipt=R03终态`、`comparison=R03 summary`，不称其为R02批量结果。

`train_horizon.py`在独立目录沿用旧采集、2/5更新、预热、学习率、检查点与健康保护循环；读取R03顶层warmup字段。三网络/Adam/bin从原权重加载；新入口/全部源码进入manifest。`run_horizon_pair.py`复用原controller的子进程运行和健康保护，串行执行两臂共16阶段；新保存审计复用原13项数值审计，并替换为明确的R03谱系与单变量合同。

每臂训练→保存审计→initial三片整段动力学评测/独立重算/位移分析→final相同流程。最后检查两臂设置仅H/输出不同、各自初始权重精确继承同父、三段初始轨迹SHA相同；自动输出完整匹配比较，不自动选赢家或追加预算。大权重原件留服务器LOCAL_ONLY。

## 控制与评判

本地`python3 -m unittest discover -s script/supermimic_q1_horizon -p test_horizon_pair.py -v`原生退出0，7通过、1跳过（Mac无PyTorch）。已覆盖仅H变化接受、额外gamma/资产/H16变化拒绝、匹配种子保留、加载抛错恢复、共享锁、失败阻止下一臂及禁止重跑、三片完整评测安排。服务器必须8项全通过，含3个微型Linear/Adam继承fixture；fixture不是Q1训练或科学结果。

主要判断：走路整片完成/早倒、腿部关节RMSE、水平根跟踪RMSE和离起点最大水平距离；同时报告全身、浅蹲和复用开发出拳，避免只按奖励选方案。未达原0.15rad或实际移动要求时仍如实报告失败；不因某项改进自动称可用。单种子/固定A后B顺序与反复使用开发片段，不能支持统计显著性或独立泛化结论。

最多新增96000训练控制转移；四次三片评测的完整上限13948控制步，共109948控制步/2198960物理子步。真实早倒保留，不重置补齐或替换。A保存/评测任一阶段失败则整次终态失败、B不启动；不覆盖失败、不自动重启。

## 版本、入口与产物

代码副本`/tmp/yuanqi-supermimic-q1-20261023`、分支Q1。D08结果已发布并核对`00936b81b9a69e0ea866ab517352a97a78e2c793`。服务器`huanyu:/opt/data/private/YuanQi_Q1`，Python`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。冻结输入科学发布`d5f9476125f2c3c7ed2d7c1e6ac898e1f69d6429`，新增代码由release/Git历史/执行SHA标识。

先在已推送Q1的Mac副本生成小release，再部署全部新增源码/两个analysis依赖并核对release。服务器项目根运行：

```bash
"$PY" -m unittest discover -s script/supermimic_q1_horizon -p test_horizon_pair.py -v
"$PY" script/supermimic_q1_horizon/run_horizon_pair.py register --root "$PWD" --output data/supermimic_for_q1/R04_H32_H8_matched3000_20261024 --release data/manifests/supermimic_q1_R04_release_20261024.json --parent-experiment data/supermimic_for_q1/R03_B128_extend12000_20261023 --diagnosis data/supermimic_for_q1/D08_R03_action_response_20261024
"$PY" -u script/supermimic_q1_horizon/run_horizon_pair.py execute --plan data/supermimic_for_q1/R04_H32_H8_matched3000_20261024/plan.json
```

计划目录`data/supermimic_for_q1/R04_H32_H8_matched3000_20261024/`；A/B各自`run/`保存权重/metrics、各自评测/审计/位移结果；顶层保存plan、started、status、events、comparison、receipt。日志目录`logs/supermimic_q1_R04_H32_H8_matched3000_20261024/`。重启/失败保护沿用容器30GiB、GPU低于86摄氏度/空闲至少2GiB、OOM不增、磁盘至少10GiB及15秒controller采样。

## 保存与下一步

先审核/提交/推送并独立核对Q1；服务器8项控制通过后register，检查真实输入/16阶段/预算与SHA，再启动一次有限流水线，确认起始继承/真实采集与原进程存活。立即记录启动证据并同步Q1。尚无R04最终结果，原R03仍保留，A3停止；全部模型/视频/数组不下载到Mac，网盘未验证。
