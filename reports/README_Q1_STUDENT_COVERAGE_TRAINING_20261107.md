# Q1 因果学生扩大覆盖实验

记录日期：2026-11-07。实验：COVERAGE01。状态：独立实现及九本地检查完成、源码和193项清单已发布；部署接收器无回执，正文未发，真实采集与训练未下发。

## 给用户的简报

done：Agent自行复用53893进入原Linux，已核原训练暂停、GPU空闲及成本账本1606秒且无未结算任务。独立采集器、拟合器、来源连接和训练六宫图入口已实现；九项CPU合同检查通过，0.680秒。夹具中的三次Adam不是32k真实训练。

to do：发布后仅采30新片、最多9585物理步，复用旧18片和两份状态数据；在原1740秒及累计4GPU小时内拟合学生。先看固定旧验证集与新扩展验证集的各自变化，再决定是否有新策略值得闭环；原0/7同策略不重跑。

## 依据与变量

[INPUT_AUDIT01](README_Q1_STUDENT_INPUT_COVERAGE_20261107.md#实际审计结果)已核训练仅12/22来源组和12/167片，启动200ms样本稀少，局部来源截断明显。扩大覆盖是可检验假设，不预设它就是失败原因。此前增加辅助下一状态、教师latent和FP32物理loss均未胜原R2，不延长相同数据拟合。

采用已发布的固定哈希候选，train22组36片、validation6组12片；原18片前缀严格复用，仅30新片。新片最多384步且保留最早因果可用的偶数帧。物理中提前终止如实保存，不把前缀称全片完成。来源分组及原test划分不变，不读test人体数组。

采集行为为有十未来帧的冻结32k教师，属于特权离线采集，不是因果遥操。标签必须与同一实际状态的执行动作一致；学生输入仍是可见当前人体加十槽历史，120→50Hz插值右支持和包到达时钟纳入可见性检查。新采集不混入学生动作，也不重跑旧混合状态采集。

冻结原R2归一化、教师、decoder和WM，复制原R2已选编码器，只做动作蒸馏。训练按来源组、片段、可用状态原点三级均衡。因同时扩来源和启动覆盖，本轮不能单独区分这两项的因果贡献。

## 固定训练和选点

seed2026110704，Adam学习率0.0001、batch128、最多3000更新、梯度范数上限1，每500更新验证且包含0。连续六次没有合格改善则停。记录六个Adam真实计数、每来源/原点曝光、有限CPU selected/last保存与复读，教师和decoder前后SHA不变。

扩展验证风险按来源组→片段→可用状态原点均衡计算；另外保持原R1验证集和原混合验证集为两个固定锚点，逐点单独报告。只在两锚点均不高于update0时，才按扩展验证风险选更优点。不能把扩展集loss直接与历史不同集合的loss作改善比较，也不使用test或未选last挑模型。

## 实现与验证

独立模块：`student_coverage_training.py`、`student_coverage_worker.py`、`student_coverage_data.py`、`student_coverage_fit.py`和`deliver_student_coverage.py`。计划：[student_coverage_experiment_20261107.json](../script/supermimic_teleop/student_coverage_experiment_20261107.json)，候选：[原覆盖清单](../data/manifests/q1_student_coverage_candidate_20261107.json)。原184执行来源和200冻结来源不修改。

本地九项检查通过：旧两原点精确连接且未来标签数组不进学生、未知来源/test/跨组拒绝、三级均衡、物理状态/动作对应和首个可用启动样本、未来依赖/捕获拒绝、三次夹具Adam冻结统计/decoder/teacher/CPU保存和锚点选点，另三项原候选元数据合同。首次测试缺少测试文件自己的import已修正，真实服务器CUDA未触发。

## 资源与执行

固定独立根：`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_STUDENT_R3_COVERAGE01_20261107`。策略和拟合CUDA，物理继续共同Primebot CPU；无媒体重渲染或旧物理重跑。原非阻塞GPU锁、22GiB准入/24GiB退出、GPU18GiB/80度、磁盘100GiB和private500GB边界保留。

控制器最多1740秒、原阶段1800秒和输出1GiB，采集预留120秒给拟合/保存，训练预留30秒退出；同一累计账本R3记账。入场前完整核原身份、原暂停和五PID退出、账本及全部来源，退出后核来源、回收与结算。没有新的缓存清理。现场内存8048500736/34292629504B，GPU1MiB/25度/OOM0、磁盘184009240576B；这是准备时快照，不替代正式准入。

调用独立入口沿用原`train_preflight`参数，并增加`--initial-r2-root`、`--aggregate-root`、`--input-audit-root`、`--coverage-plan`及其SHA、`--candidate-plan`；先核发布清单和原运行时九项，再唯一新根派发。SSH由Agent按用户固定source/ssh入口自行尝试，不请求用户代测。

## 结果与恢复

源码8432242d412bcbe50e52d2bf6be94363aab93c2a、清单8991ca35c84ded05d1c1af1e2ff226e4594a1109均显式push origin Q1，独立远端完整40位相同；Mac15文件863217B与Git逐字节相同。193项清单35639B/SHA`d9fc8182b5b5a648b7fed837e2f0007500734993d7228226f958ab7b0140cec8`，12成员代码包42163B/SHA`589e21d7d9dbebc3ca9cda149eeebc40b935733cff29f6cf23c5367d026e8b70`已准备。

53893旧会话超时后，Agent自主按固定source/ssh入口新55892已进Windows，再原生第二跳密码成功到原Linux，并确认`/opt/data/private/ZHY`。提交部署接收器后尚无READY/部署回执即该SSH明确Timeout退出255，代码包正文发送0。独立一次54567仍原样固定入口，仅第一跳Operation timed out，未到Windows认证或Linux。这个新重连失败不等于Linux不可达；没有请求用户代测。详见[发布与连接实录](../logs/session_records/q1_student_coverage_publication_and_connection_20261107.json)。

下次自主沿固定入口恢复，先查`q1_student_coverage_deployed_20261107.json`、12目标大小SHA、接收器实际进程及唯一COVERAGE01输出/派发/日志。接收器无正文不可当已部署；若旧接收器仍在，先核其确切命令/启动tick和退出状态，不重复启动接收器。不因SSH退出推断后台状态，也不重派未知任务。补缺部署后以固定原运行时做九检查，再完整`coverage_preflight`和原锁/账本准入，保存回执；实际训练Popen尚未提交，任何待发命令不得当已开跑。服务器镜像和原运行时均未完成。

当前真实新物理/训练仍0；新代码发布、服务器运行时与准入尚待。若选点仍0，须CPU核实际编码器及策略元数据身份，再复用原0/7历史结果；不能为产生视频重跑同策略。若选出不同策略，后续在独立来源/资源门下做三名义和四事件闭环、原速视频、脚底换脚/滑移/侧倾/上肢时序/动作突变及完整软件时钟。六宫图是编码器训练记录，不编造RL回报或500iteration滑动均值。

原共享训练人工暂停、原final门、用户采纳null保持；本轮不调用goal工具、不新agent、不改其他自动任务。阶段提交/远端40位和Mac/服务器大小SHA核验以后续publication为准。此前两份输入审计publication收据随本次实质实现归档，不为收据单独循环提交。
