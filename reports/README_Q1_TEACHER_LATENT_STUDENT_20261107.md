# Q1当前历史学生的教师隐变量监督

日期：2026-11-07。状态：LATENT01在聚合数据缺少`full_goal`时退出，优化器更新0；进程已回收，27秒成本已结算。独立离线标签重建修订与五项本地检查已完成，修订已发布并确认部署。17时Agent自行固定入口曾到原Linux，但只读核验没有返回回执即会话超时，同入口一次自主重试在Mac→Windows连接超时；原运行时检查/准入结果仍待核实，LATENT02尚未派发。原共享训练保持人工暂停，首轮累计1293/14400秒成本不重置。

## 用户版done / todo

done：上一项GigaBrain下一状态辅助预测完成，状态预测改善42.15%，但动作validation未改善，两组选0。已实现另一项独立假设：用冻结32k教师的32维human隐变量作训练标签，帮助f0当前＋历史学生学习原共享decoder使用的表示。本地四项检查通过，历史action-only控制不重训。

todo：自行固定入口恢复后先核运行时检查/准入回执，再发布准入并唯一单派LATENT02/最多3000Adam。只有选中策略真实改变且通过validation选点，才开展新的独立闭环；若仍选0，核精确身份后继续复用原失败证据。每轮交付训练六宫图，真实机器人端到端延迟另测。

## 为什么尝试与文献边界

本项参考RMA训练期使用特权隐变量、部署由历史推断表示并复用基础策略的思路。原始来源为[RMA论文](https://arxiv.org/abs/2107.04034)及[作者论文正文镜像，图2与第二阶段](https://web.eecs.umich.edu/~stellayu/teach/2023action/papers/2021malikRMA.pdf)。原RMA估计环境因素；这里监督的是冻结SuperMimic人体目标编码器的表示，是独立迁移假设，未复现RMA完整方法，也不是GigaBrain完整行为世界模型。

已保存数据中，f0编码器仅通过冻结decoder的动作误差训练。动作监督不直接约束其隐变量是否落在原human编码器的表示区域，这是本轮检查的假设，尚未证明是此前倒地原因。教师的未来意图可能无法由当前与过去唯一推断；增加隐变量监督也可能强迫学生拟合不可观测信息而进一步退步。

与原下一状态辅助头不同，本轮没有新增可绕过human隐变量的状态/动作预测头。训练损失为`L_action + 0.1 * MSE(z_student, z_teacher)`。当前/历史输入、encoder容量、train-only统计、冻结decoder、教师、数据、seed与优化器曝光全部保持；不得将未知未来作为学生输入或部署标签。

## 标签与控制复用

只复用R1与AGGREGATE01原12个train/6个validation来源。读取保存的`full_goal`仅供冻结教师生成离线标签，逐行核同clip/frame、f0、930维机器人history与教师动作，并在相同机器人状态下确认`decoder(z_teacher, history)`复现保存动作，容差2e-5。原R3的终止mask与实际状态重建结果必须精确一致；本轮不丢行、不新采物理、不读test。

部署仍为同一`CurrentHistoryEncoder`和冻结decoder，只接收f0十步人体历史与原机器人history。教师未来窗口、32维标签、旧full_goal、下一状态和真实执行未来动作均不进入部署输入。所选checkpoint不包含标签网络。旧120→50Hz右支持、包龄、启动与丢包边界继续成立。

控制严格复用`Q1_LOW_LATENCY_CKPT32000_STUDENT_R3_AUXILIARY_PAIR02_20261107/action_only`的已完成真实拟合。新组从原R2同一选中encoder开始，seed2026110703、batch128、lr1e-4、3000Adam、每500validation、梯度上限1。每个validation点均要求批次SHA和24格source曝光与历史控制精确相同，初始validation动作风险一致；不因重新画图或写比较再训练控制。选择仍只看同权重validation动作风险，允许update0，不以latent loss或已看过的test挑模型。

原教师见过393片，旧测试也已被查看，后续比较仍属探索性，不能声称全新盲测。即使latent误差下降，也必须按动作泛化与闭环结果判断；不能代替离地/换脚、滑移、侧倾、上肢时序、动作突变与恢复检查。

## 实现与有界执行

独立入口为`student_latent_training.py`、`student_latent_worker.py`与`student_latent_fit.py`，另有plan、四项检查和训练六宫图入口。原256执行/200冻结及R3已用代码均不热改，新增文件必须列入新执行清单；旧控制与其来源、训练数据和计划仍逐项SHA保护。

新根为`Q1_LOW_LATENCY_CKPT32000_STUDENT_R3_LATENT01_20261107`。本轮最多1740秒，原1800秒阶段保护、1GiB输出、cgroup22GiB准入/24GiB退出、GPU18GiB/80度、100GiB磁盘余量/项目500GB和非阻塞原GPU锁保留。沿用r3累计145/7200秒、总1266/14400秒；本轮实际标签生成与拟合持锁墙钟均计费，阶段预算不重置。复用原R2缓存边界，不重复缓存清理。

本地四项检查于16时附近通过，耗时0.748秒：未来标签与学生输入分离、同状态动作及完整教师mask、clip/frame精确连接、真实合成CPU三Adam及历史控制的逐点批次SHA/曝光/冻结decoder/选点/CPU有限保存。合成夹具不等于真实32k训练；真实本轮GPU更新仍0。

## 接续与保存

下一步先发布本方案，原运行时四检查与新源码/计划/旧控制来源审计后再派发，不重复R3控制拟合或原七f0物理。结果先交训练六宫图；选中策略未变则精确身份审计并复用原0/7，改变后另建真实闭环入口，不绕过原final门。

固定用户入口由Agent自行执行`source /Users/yu/.zshrc`再`ssh -tt hp_han`，沿Windows原生第二跳交互密码；不要求用户先测试可达性。66625退回本地后已关闭本地shell，句柄结束，不能再探旧句柄或向其发送Linux命令。服务器原始收据、publication和新任务状态各以实际回执为准。

## 16:30服务器准入

Agent在现有本地交互zsh66625中自行执行用户固定入口，进入Windows后原生第二跳交互密码成功进入原Linux。此前同一终端的SSH已退出到本地；本次将终端类型从`dumb`设为`xterm-256color`后再连接成功，仅是本次环境与结果记录，不能据此确定此前超时的唯一原因。后续优先复用66625，不要求用户代测。

16:27:16北京时间实核原root/boot/GPU，旧五训练PID及R3控制器/worker均退，原暂停收据仍有效，GPU计算为空、1MiB/25度，磁盘余185495429120B。先确认新七文件/代码包/部署收据、LATENT01输出/派发收据/日志和接收器进程全部不存在，才部署新七成员包；735项原保护来源前后SHA全同。

原Primebot运行时四项检查全过，0失败/0跳过，耗时1.813833秒。随后`latent_preflight`检查170执行、200冻结源码资产、745保护输入、原真实数据和已完成对照；原GPU锁独占准入后释放。总内存8096813056B/实际限34292629504B、GPU1MiB，成本账本1266秒、r3已145秒且全部SETTLED。真实本轮训练仍0，先发布本准入，再唯一LATENT01；原共享训练不恢复。

三份服务器原始部署/运行时/准入回执已Mac与仓库原字节核同；2003B包SHA为`488d333e693d1282a14ffeb356adb8dfcafd4f15aaaa7d5bc1a47f1743fda820`。原始记录为`q1_student_teacher_latent_deployed_20261107.json`、`q1_student_teacher_latent_native_checks_20261107.json`和`q1_student_teacher_latent_admitted_20261107.json`。本次新说明的服务器镜像另以实际回执为准。

## 首派数据结构错误与修订

准入307a3b76b8b37a8797ffd830e0a53ab5fe490813已显式push并独立40位远端核同，Mac八份726457B核同后，唯一LATENT01真实派发355713/start35341011、worker355785。它在`attach_goals`读取AGGREGATE01数据时出现`KeyError: full_goal`。原R1数据保存了归一化完整教师目标，聚合数据只保存f0/history/教师动作/实际状态/帧号，首版本夹具未覆盖这两种真实保存格式的差异。此错误发生在拟合与第一条progress之前，不是新步态或连接故障。

原guard已WORKER_FAILED/return1并回收，355713/355785与五旧训练PID都退；原锁释放、GPU计算空。实际27秒SETTLED，累计1293秒、r3已172秒；峰总10491744256B、OOM0。原FAILED目录、日志、170项原清单与22261B原代码包保留；后者包含旧worker完整字节，不将首派改写成功。原dispatch和失败回执1802B原包已Mac大小SHA同。

修订只改已退出的新latent worker与检查，不改原256/200冻结、已成R3控制、共享教师或decoder。原R1继续使用保存目标；聚合行从原配对human NPZ、同clip/frame和保存实际机器人torso重建相同十步教师目标，沿原human/heading normalizer和完整mask打包。人体和collection plan前后核SHA，`latent_records`仍要求同状态动作复现误差不超2e-5后才允许任何Adam；未来目标仍仅离线标签。保存f0/history/动作/终止行和历史控制batch曝光不变，无新物理。

新增真实聚合格式的回归合同：不含`full_goal`时必须显式重建，全部原行/输入不变，重建目标错误形状拒绝；最新五项本地CPU检查0.756秒全过。计划、学习率、3000更新和action-only历史控制不变。下一发布新来源清单/原运行时五检查和准入，唯一新LATENT02根，仍在原r3/总成本内，不重试旧输出或恢复原共享训练。

## 修订部署与当前恢复点

修订906ce50265400a3cabd296a43bf54d69dc47fbbf与清单记录932883df6f66b9a6e04129d01048e32cc4087305均显式push/独立40位远端核同，Mac十份783356B大小SHA同。新170项清单31594B/SHA350816d0188f143536b37077b07749fcdaa7cce86ced7441dc1f4327cbf83c8b；三成员修订包15828B/SHAfbc4651f195df3f0b373feaa592f42104b8e9b0431b32592fbc5f439b5e72688已经66625送达，收到`Q1_LATENT_R2_CODE_DEPLOYED 3 735`，735项旧保护来源前后SHA全同。旧失败源码仍在原代码包，旧清单不覆盖。

随后提交外层300秒保护的原Primebot五检查，收到前三个进度点，尚无完整成功/失败回执；同会话后提交的重新准入命令也未收到回执。客户端明确`Timeout, server 100.65.183.91 not responding.`后回到本地zsh。Agent立即自主原样source再ssh，第一跳明确`Operation timed out`，未Windows shell、未尝试Linux；仅据此记录该次第一跳失败，不概括Linux或后台停止，不让用户代测。TERM调整后仍发生超时，更不能当作网络问题已解决。

下一周期自行沿固定入口恢复，先查新deployed/native_checks/admitted收据、实际timeout/test/preflight进程与LATENT02输出/派发/日志/账本，未核前不重传、不重复检查或Popen。LATENT02真实新训练派发0，不能把部署或进度点称为训练开始。新文档服务器镜像待；最新接续参数在`q1_student_teacher_latent_r2_runtime_unacknowledged_20261107.json`。既有目标和q1监督继续，原共享训练不恢复。

恢复点：源码33ec52ca1fe42b5ade001fd964f2af82a5fc9566已push且独立40位远端核同，170文件执行清单、7成员22261B包已本地完成；原163来源SHA全同。56557在提交接收器命令后明确Timeout255退出，未收到READY，归档正文发送0字节，尚未派发任何新训练。随后Agent原样固定入口自主重连40295在Mac→Windows阶段返回Connection reset255，未Windows shell/未尝试Linux。不能概括为Linux不可达；部署收据/目标文件/接收器现场仍须恢复后核查，未核前不重传。此前15:55服务器八份R3完成记录镜像成功仍有效，不被这次新会话故障改写。继续有界自主尝试固定入口，不要求用户代测或更换认证路线。

## 17:04自主恢复实录

Agent自行在交互zsh50274中执行`source /Users/yu/.zshrc`，再执行`ssh -tt hp_han`；实际收到Windows PowerShell，再经原生第二跳交互密码收到`root@interactive67462:~#`。随后已收到切换`/opt/data/private/ZHY`的Linux提示符。TERM为xterm-256color，只登记终端环境，不将其视作网络故障修复证据。

提交的只读检查覆盖三个r2收据、LATENT02输出/派发/日志、相关进程、GPU/磁盘和成本账本，但没有返回其结果。后续轮询明确收到`Timeout, server 100.65.183.91 not responding.`并退到本地zsh；Agent立即自行原样source再ssh，有界重试明确`ssh: connect to host 100.65.183.91 port 22: Operation timed out`，未Windows shell，未尝试Linux。只说明该会话结束和该次第一跳失败，不能推出Linux服务器或后台检查停止。没有向用户询问可达性。50274随后关闭本地shell并结束，不能再探该句柄。

本轮没有提交LATENT02 Popen、重复上传、重复测试、准入或训练。已确认的修订部署保持；五项原运行时检查与准入收据仍未核验，不能将旧进度点称为通过。最后核实的成本仍1293/14400秒、r3已172秒，原共享训练保持人工暂停。下一周期Agent继续自主固定入口，先读三个原始回执和实际进程/输出/日志/账本，再只补缺失步骤；不要求用户代测。原始小记录为`q1_student_teacher_latent_r2_autonomous_recovery_1704_20261107.json`。新文档服务器镜像待实际连接回执。
