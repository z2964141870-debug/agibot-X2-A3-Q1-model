# Q1低成本下一状态辅助预测对照

日期：2026-11-07。状态：PAIR02真实完成，两个训练六宫图已交付；辅助预测学会了下一状态，但没有改善validation动作风险，两组都选update0。CPU精确策略身份审计通过，复用原R2失败证据而未重复七例闭环。继续既有低延迟目标，不另建目标；原共享160k训练保持人工暂停，32k教师、世界模型、共享解码器、动力学、保存频率、原final门和采纳null保持。

## 用户版done / todo

done：严格同初始化、同批次、各3000Adam的两组真实GPU拟合完成，来源门与清洁回收通过。下一状态validation MSE下降42.15%，但动作风险由0.094864变为0.115836；action-only末值0.114754，也未改善。两组均选初始策略，八个encoder张量和八项策略元数据与原R2精确相同。两张训练六宫图PNG/PDF已经生成并目视核看；本轮持锁成本131秒，首派代码故障14秒另计，总累计1266/14400秒。

todo：停止追加本配方迭代。下一项先核查教师隐变量监督是否比单独动作蒸馏更有效，独立登记后只改变这个训练信号，复用本轮严格匹配的action-only控制。未来教师信息只能用于离线标签；部署仍须通过f0可见性、真实闭环和完整软件周期检查。R1视频part继续补传，不重渲染。

## 真实完成与结论

PAIR02控制器354888/start34987257、worker354961已退出并被守卫回收；campaign为`PAIRED_FIT_PASSED_PENDING_CLOSED_LOOP`，`source_integrity=PASSED`。两组encoder六个Adam计数各为3000，辅助头四个计数各为3000。真实新增物理0、动作拟合6000更新，辅助头只有next_state组更新。原PAIR01失败根保留。

| 实测指标 | action_only | next_state，权重0.1 |
| --- | ---: | ---: |
| 初始validation动作风险 | 0.094863855 | 0.094863855 |
| 末validation动作风险 | 0.114754134 | 0.115836185 |
| 相对初始动作风险变化 | +20.97% | +22.11% |
| validation选中更新 | 0 | 0 |
| 初始validation状态MSE | 0.114778208 | 0.114778208 |
| 末validation状态MSE | 0.114287422 | 0.066393794 |
| 实际拟合秒数 | 52.057131 | 53.832584 |

两组批次SHA精确同为`862a2d90afcf5d9eec560de409bd5f445a7f8b7d0ca25690b64db33eb711a6a5`，24个origin/source格每格曝光16000条。原/混合train与validation的当前history重建最大差均0，没有跨终止行。下一状态标签、实际执行动作和保存物理trace均通过；这说明本项实验语义成立，不等于遥操效果改善。

CPU选中策略审计于15:40:58通过：两组八个encoder张量最大差0、八项策略元数据和教师来源相同、辅助头未部署、CUDA未初始化。序列化文件SHA因外层元数据不同而不同，不能仅比`.pt`文件SHA判策略变化。两组selected encoder SHA均为`8824f3098459816c08d882de89199183e22ffc3626231242b9823d1847b5a1b9`。原R2的0/7完整动作只作为精确同策略的历史证据复用；本轮新闭环0，不能称新的R3质量验收或尝试未选中last权重挑test。

结论仅否定当前数据、容量、辅助权重与训练预算下的这一项迁移：辅助状态预测改善没有转化成validation动作改善。不能据此否定完整GigaBrain行为世界模型，也不能说当前f0自然遥操已可用。完整worker约31ms超过20ms的旧问题仍未解决，硬件端到端延迟未实测。

本轮131秒已SETTLED，加首派14秒，本项r3累计145/7200秒；原R1 549秒、R2 572秒，首轮总1266/14400秒、余13134秒。峰cgroup10414170112B/GPU480MiB/OOM0；完成现场原训练暂停、五旧PID与本轮子树均退出、原锁释放、GPU计算空。缓存回收没有重复尝试，原共享训练信号0。

### 原始结果与训练六宫图

- [真实派发](../logs/session_records/q1_student_auxiliary_pair_r2_dispatch_20261107.json)、[真实完成](../logs/session_records/q1_student_auxiliary_pair_completed_20261107.json)、[精确同策略审计](../logs/session_records/q1_student_auxiliary_pair_selected_identity_20261107.json)。
- [action-only六宫图](../data/analysis/Q1_STUDENT_R3_AUXILIARY_PAIR02_FIGURES_20261107/action_only_training_six_panel.png)、[PDF](../data/analysis/Q1_STUDENT_R3_AUXILIARY_PAIR02_FIGURES_20261107/action_only_training_six_panel.pdf)。
- [下一状态辅助六宫图](../data/analysis/Q1_STUDENT_R3_AUXILIARY_PAIR02_FIGURES_20261107/next_state_training_six_panel.png)、[PDF](../data/analysis/Q1_STUDENT_R3_AUXILIARY_PAIR02_FIGURES_20261107/next_state_training_six_panel.pdf)。
- 原始18成员结果包`data/deliveries/Q1_STUDENT_AUXILIARY_PAIR02_RESULTS_20261107.tgz`：90849B/SHA `09b989efb2fc20c7659d22ba7d6e68553d9456a3f4340f0d13e06afe0db0446e`，17份原始工件432402B逐项大小SHA核同。权重未下载到Mac。
- [交付记录](../logs/session_records/q1_student_auxiliary_pair_figures_delivered_20261107.json)。无新策略媒体；原R2两失败视频保持原路径，R1七视频包仍仅部分传回，不能记作新视频交付。

以下为准备、首派失败和修订准入的历史记录；最新实际状态以上述完成与身份审计为准。

## 为什么做这项实验

R2纯因果学生已有0/7完整动作；上一轮状态聚合3000Adam后验证选0，最后风险增加20.97%。没有证据支持继续增加原配方迭代。先区分数据覆盖与辅助目标的作用，再决定是否改变网络或采集来源。

参考固定源码`open-gigaai/giga-brain-wbc@fa44c2418e7be86c97655e5cb0c809213b2c7dca`中`giga_brain_wbc/learning/losses/token_losses.py`的下一proprio状态MSE：沿真实时间移动一帧，保留gravity，去掉previous action。这里只借鉴该训练性质，不称复现完整BWM、获得论文训练规模或达到其遥操表现。

## 标签、因果性与单变量

严格复用原R1和AGGREGATE01的12个train/6个validation来源、真实状态和真实执行动作。原教师看到过393全库，既有测试也曾被查看，因此后续比较仍有探索性边界，不能冒充全新盲测。此训练不读取test、不新采物理、不改变教师或human统计。

每行目标取同片保存状态`frame+1`，检查当前状态、teacher查询、实际behavior与物理trace对应，并将当前93维normalized观测与保存history最新一帧核同。跨终止边界的行排除，两组使用同一行集合。目标只保留Q1真实22个关节的位置/速度及3维gyro、3维gravity，共50维；缺失关节和previous action不作为预测目标。

新头输入为human latent、当前50维normalized状态和实际执行的22维action，预测下一状态。AGGREGATE01的下一状态由半教师半学生的混合动作产生，不能将它当学生假想动作的结果；显式使用保存的执行动作解决此标签语义问题。未来状态只作离线监督，不进入部署编码器。这个头也可能主要依赖当前状态/动作，是否帮助human表征必须由对照确定。

action_only组只对原encoder施加动作蒸馏；next_state组增加0.1倍next-state MSE。两组从同一原R2选中encoder开始，原统计和decoder冻结，seed2026110703、batch128、学习率1e-4、各3000Adam、每500验证、无不同步早停。采样按origin再按source均衡，逐批记录身份SHA并要求两组最终相同；编码器和辅助头分别裁剪梯度。两组均只以同一validation动作风险选点，允许update0入选，不以辅助loss或test选模型。

部署仍为原f0十步history编码器和冻结decoder，辅助头从选中产物排除，因此没有新增部署网络调用；完整软件周期约31ms的旧问题仍单独保留，不能据此宣称20ms周期或硬件端到端达标。

## 工程与资源

独立四执行模块为`student_auxiliary_training.py`、`student_auxiliary_worker.py`、`student_auxiliary_fit.py`、`student_auxiliary_data.py`，独立plan和六宫图入口另存。原256执行代码和200冻结输入不热改，旧聚合授权中的来源逐项复核。新manifest可省略旧无执行关系的绘图/测试入口以遵守原256上限，旧256仍由原聚合授权整体保护，不能削弱原来源门。

新根`Q1_LOW_LATENCY_CKPT32000_STUDENT_R3_AUXILIARY_PAIR01_20261107`，阶段1740秒/原1800秒保护、输出1GiB、GPU18GiB/80度、cgroup22GiB准入24GiB退出、磁盘余100GiB及项目500GB保持；原GPU锁非阻塞独占、自有子树退出回收。复用原R2一次cache边界记录，不为缓存数值重复回收。

继续同一持久账本r3上限7200秒、总14400秒；开训前已用1121秒、总余13279秒。两组拟合及其持锁墙钟均计入r3，不重置旧r1/r2预算。首轮真实更新目前0，原新任务结果须实际来源门/回收/SETTLE后确认，训练完成不等于闭环质量通过。

## 证据

- [保存残差原值](../logs/session_records/q1_student_state_residuals_saved_20261107.json)：67570B/SHA `f5a69d0d9d28502a0bcb5a34163648e3310a739e33c3aec6f1f297a5f2ac406d`；两输入SHA前后同，新推理/物理/训练均0。首次导入mjlab失败未产生分析根；改为只读常量后成功，未改训练依赖。
- [上一轮聚合负结果和六宫图](README_Q1_STUDENT_STATE_AGGREGATION_20261107.md)。
- [原R2闭环失败](README_Q1_F0_HISTORY_STUDENT_20261107.md)。

本阶段发布、服务器检查、准入、派发、完成与镜像分别以随后实际收据为准；不因旧SSH会话退出断言整个入口不可用，不要求用户先代测连接。

固定用户入口56557已自主进入Windows并通过原生25362密码到原Linux。原boot/GPU及唯一private根同、五旧训练PID退出/STOPPED_SIGNAL、原锁可独占后释放、GPU1MiB26度/计算空/OOM0/总7981060096B限34292629504B均已实核。服务器残差原文件67570B及SHA也与Mac精确相同。这一成功不据此判断旧超时的唯一原因；没有询问用户代测或更换第一跳。

## 真实服务器准入

源码728f804cbffcde90c13f45083784f08ddbc1a0e4已push并独立40位远端核同；八成员23272B包SHA `c3b51e368a3e3be4f880db2dea8438424704b007b85cd890ff1e60ff8898cffc`已沿原会话送达。仅新增清单外文件，旧授权700份保护输入前后同；原运行时六检查全过，外层4.193942秒/测试1.524秒，仅随机CPU夹具各三更新。

07:25UTC附近真实准入163新执行来源/200冻结/729保护输入通过，原12train/6val数据复用和原GPU锁检查后释放；GPU1MiB26度、总8073879552B/限34292629504B/OOM0。三原始回执已Mac精确取回，2410B归档SHA `b3b93ee2405f9c740a6eaec03c56f0c584daaeebe9cad5b1c330b2693f9ae5ff`。真实新训练仍0；先发布本准入再单派新根，不能据准备或随机夹具当实际学生训练。

[部署](../logs/session_records/q1_student_auxiliary_pair_deployed_20261107.json) / [原运行时检查](../logs/session_records/q1_student_auxiliary_pair_native_checks_20261107.json) / [准入与参数](../logs/session_records/q1_student_auxiliary_pair_admitted_20261107.json)。新执行清单163项；省略的旧非执行测试/绘图仍受旧授权保护，原256和冻结200文件没有修改。

## 首派初始化错误与修订

PAIR01于15:28派发354447/start34938997，worker354519在导入顺序门退出：`analytic_obs`间接先加载native state，而稀疏Q1 contracts必须先安装。尚无action_only/next_state拟合目录，真实optimizer更新0；守卫已回收、原锁/计算空、费用14秒SETTLED，总1135/14400秒、r3已14/7200。原FAILED根和23272B精确源码包保留，不能把这次代码故障称策略失败或桥接故障。

修订仅新worker的初始化边界：先`install("q1")`再导入解析观测，增加真实调用同一初始化函数的本地回归。旧256/200冻结代码不动，原失败源码由发布提交和原八成员精确包保留；下一新PAIR02根，原1740秒阶段和r3累计预算不扩。发布修订后还须原运行时七检查及真实初始化加载、再准入与唯一派发。

源码修订5f430b4362423521234fcb7ebcc9eed0e9cbb8ad已push/独立40位远端核同。三成员14112B包SHA `1e902719f04ad4c34daf864794d83adb174be97f80bdd4dda6f3f62a89f95807`已部署；原700保护来源同，原八成员源码包原SHA保留。15:33原运行时七检查1.529秒通过；额外使用新worker实际`native_contracts`入口完成稀疏Q1安装和解析状态导入，CUDA未初始化，两项总22.766922秒。新manifest独立路径，旧manifest未覆盖。

PAIR02准入163执行/200冻结/729保护和原锁全过，GPU1MiB26度/总8073822208B/限34292629504B/OOM0。修订部署/原运行时/准入和首派dispatch/故障五份原始回执4090B包已Mac SHA `931701f15b4b6b3991d0c113eaccae8067cb86319372b5f1f01e7e55e6727f29`核同。真实新拟合仍0，发布本准入后唯一派发PAIR02，费用从1135秒继续累计。
