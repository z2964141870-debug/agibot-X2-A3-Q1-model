# Q1低成本下一状态辅助预测对照

日期：2026-11-07。状态：独立实现和六项本地检查完成，真实服务器训练尚未开始。继续既有低延迟目标，不另建目标；原共享160k训练保持人工暂停，32k教师、世界模型、共享解码器、动力学、保存频率、原final门和采纳null保持。

## 用户版done / todo

done：只读已保存的学生状态聚合数据，完成真实动作残差分析；验证集腿部RMS为0.409864，训练集为0.083068，单位为canonical动作delta，并非关节跟踪rad。两集上肢为0.133632与0.040352。该差异提示当前学生泛化仍弱，但不单独证明倒地原因。新增两组严格同初始化、同3000更新、同采样批次的训练入口，本地六项检查通过；随机CPU夹具不是32k真实训练。

todo：发布后在原服务器运行时验证、核身份/暂停/资源/原GPU锁，再唯一派发配对实验。只改变辅助预测这一项；完成后交付两张训练六宫图，根据validation动作误差选择模型。选0或模型精确未变时复用既有失败证据，不重复相同七例物理；选中策略确实变化后才做独立闭环验证。R1视频part继续补传，不重渲染。

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
