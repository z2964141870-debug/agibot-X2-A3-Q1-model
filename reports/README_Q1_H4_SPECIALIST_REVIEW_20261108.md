# H4专用编码器：七例匹配闭环验证

2026-11-08，北京时间。检验H4-only继续适配的0.923%验证收益是否成为闭环质量收益。新权重固定为H4_SPECIALIST01的validation-selected1500，不读取last或按这七个场景换选点。本阶段是已见片段的探索性诊断，不作新留出集或自然遥操合格声明。

## done / todo

done：训练本身已完成，训练六宫图和307711B证据包已交付，结果发布0e7a3ea远端40位与Mac八文件820432B核同。独立student_h4_review入口、固定plan和四项CPU合同实现；四检查0.002秒通过，AST与CLI通过。现有45014仍正常返回原Linux，03:51准入前快照总8154480640B/限34292629504B、OOM0、GPU1MiB25度、余盘183630389248B，原五PID和H4训练三PID均退，原GPU锁可独占获取。

todo：首派已有一条完整物理回放，媒体入口错误已定位。独立续行入口及保存分析已通过本地检查，待发布、原运行时媒体依赖和来源资源准入；只补六新物理与七媒体。完成后只读分析保存工件、生成七组对比六宫图及原速媒体，完整性与失败同时报告。

## 首派媒体错误与独立续行

首派372290/start39480877已退出；前走A022物理1031/1031步完整，两个守卫均已回收。媒体子进程372840退出1，错误为`ModuleNotFoundError: No module named 'video'`。新入口漏掉既有媒体分支使用的`runtime_paths(runtime)`，是实施错误，不能当步态失败或网络断联。原FAILED根、201项已执行源码及训练五源码保留不改。

原始失败诊断、派发、result、resources、两个guard及媒体日志共七件已Mac字节核同：`data/deliveries/Q1_H4_SPECIALIST_REVIEW_FAILURE_RAW_20261108.tgz`，4661B，SHA256 `efe15ad11bcd084a67946717ae66071bb5cffc2803c0636da71803add801fb90`。独立失败记录见[原始诊断](../logs/session_records/q1_h4_specialist_review_media_path_failed_20261108.json)。929保护输入未变，17件失败工件和首轨迹身份均保留；首派未完成原source_integrity正式门。

新增`student_h4_review_resume.py`使用唯一`Q1_LOW_LATENCY_CKPT32000_H4_SPECIALIST01_REVIEW_RESUME1_20261108`根。必须先核原诊断SHA、全部失败工件、selected1500、参考/场景、有限数组、1031条可见性时钟和两个守卫回收；再逐字节复制首物理，仅更新新副本trace路径。只补6新物理＋7媒体共13阶段，复用17旧对照，总计24比较；原worker、物理与时钟不变，CPU加速不接入。

首派原wall100.032008秒，保守扣101秒。账本已SETTLED累计2670/14400秒、无RESERVED，r1已727/2700；续行最多1699秒，不重置原1800秒验证预算或账本。累计wall检查、原22/24GiB、GPU18GiB/80度、100GiB余盘、private500GB、原GPU锁和自有子树回收保持，新增Adam0。续行四合同覆盖真实媒体分支路径顺序、6＋7阶段、原失败和累计预算；保存分析另核13守卫、7新例及17旧对照，失败不造全片误差或整组均值。

八项原运行时检查及真实NVIDIA EGL模块依赖已通过；首次续行准入因205项清单把`q1_checkpoint_review`只读分析目录加入执行代码而被原目录门拒绝。续行未派发、输出未建立、账本无新预留，成本仍2670。保留被拒清单，独立V2执行清单只追加resume和其测试，203项中原201记录不变；已发布的两份保存分析源码另行核验，不放宽原目录门。原运行时八项和EGL检查不重复执行。

04:25:02真实V2准入已过：948项保护输入、原1031条时钟与首物理、原失败及两个回收守卫核同；3.001292秒，原预算101＋剩1699秒。GPU为NVIDIA RTX3090/EGL，原运行时八检查本体各0.002秒、进程共0.569秒。总8153034752B/限34292629504B、OOM0、GPU1MiB25度、余盘183626768384B。五原部署/检查/拒绝/修订/准入记录6756B已Mac字节SHA同，2709B原包SHA `2085558cbad0635545b3a38139c76723f10ce2024aaa90219b6bff8945fbfee5`。下一唯一RESUME1输出、日志、dispatch及精确进程防重后单派，服务器仍无新续行物理/媒体或训练。

04:27:17完成唯一防重及实时资源门后已单派：timeout373346，controller373347/start39615599，独立RESUME1根，外层1699秒。启动3194B原收据已Mac SHA同。随后首1031步物理直接复用，首NVIDIA EGL原速媒体通过；三名义片物理及媒体已保存，当前继续同一次四事件验证，尚未下闭环收益结论。原共享暂停/训练更新0/采纳null。独立保存分析已发布，新Mac七六宫图及证据打包入口仅CLI准备，不读取未完成结果或重复物理。

## 唯一变量与对照

只替换R1的human前缀编码器，复用原student_review.replay和原student_review_worker；固定variant=student、请求H4/80ms、原decoder与normalizers、共同Primebot CPU物理、固定种子0、原120Hz依赖支持/50Hz控制时钟。CPU转换加速作为独立已过等价的结果保留，本轮不接入，以免把软件时序变化混入权重比较。

新增前走A022、起走A023、停走A024三片名义，加stop/reverse/dropout/late_burst四事件前走，共七片，每片各一次物理及NVIDIA EGL原速50fps媒体。全部复用原R1保存14个H4教师/学生对照及三个H10名义上下文，共24个比较条目；不重跑旧策略或84事件。每条目标SHA与原保存对照必须相同。合成事件与原始人体预处理offline_unverified边界保持。

指标保留：整片完成率，左右脚承重滑移、离地和换脚，侧倾，上肢全片/首秒窗口/描述性相位，动作突变和丢包fallback恢复；失败片只出保存前缀诊断，不造全片误差或整组三片均值。同步目标、完整worker及本机pipe RPC的初始化/去首及去前两次长尾均报告，嵌套区间不相加，不当硬件端到端延迟。

## 来源与安全

新入口必须先核已完成H4来源915项、R1真实训练和保存分析、selected1500的原始拟合/张量/decoder身份，再核全部新执行来源。原共享训练仍人工暂停，采纳null，原final门不变。

仅沿现有45014；必要新第一跳原样source /Users/yu/.zshrc再ssh -tt hp_han，第二跳原生密码。hp_han获批三项90/60/5已实核，由Agent自行测试，不要求用户代测。

原账本2569/14400秒全部SETTLED，r1=626且原2700秒余2074；本轮GPU独占验证也计入原r1账本，先reserve、全部guard回收后settle，不重置。总1800秒、阶段1800秒、输出3GiB，外层1810秒TERM/10秒KILL；22GiB准入/24GiB退出、GPU18GiB/80度、磁盘余100GiB、private<500GB、原非阻塞锁及自有子树回收保持。无新增Adam、缓存清理或原训练信号。

唯一输出：`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_H4_SPECIALIST01_REVIEW_20261108`。stdout：`logs/q1_h4_specialist_review_20261108.log`，启动收据：`logs/session_records/q1_h4_specialist_review_dispatch_20261108.json`。未回执先查同任务，不重复派发。

实现与可复现CLI：`script/supermimic_teleop/student_h4_review.py --help`；固定参数见同目录`student_h4_review_plan_20261108.json`及后续dispatch原件。原运行时和实际派发以后续回执为准。

## 下一决定

若七例有一致改善，再在独立未用于选点的新事件和片段验证；若主要仍回退，停止延长该小数据的相同拟合，转向有证据的闭环状态分布/控制目标适配。不能仅凭MSE小幅改善或文档完成结算整项低延迟探索。

## 原运行时与真实准入

执行源码`0fe9dc83087fa20f5add5577f59895f18d656b0b`及201项清单`9c00112048e4aa3855a13672033b4e3f7b690e1c`均已push origin Q1/独立40位远端核同，Mac九份821602B与清单两份40005B相同。30242B十二成员部署包经45014逐片14精确ACK，11目标大小SHA同、915原保护输入前后相同。仅新增代码与本阶段小文档，旧大包和全索引仍PENDING。

北京时间04:00:57四项原策略运行时合同全过，本体0.002秒/进程0.434秒，201执行来源已核；04:01:02真实准入5.292秒通过，927保护项、17保存对照、selected1500的十个编码器张量已改变而冻结策略元数据相同，CPU核验未初始化CUDA。总8182833152B/限34292629504B、OOM0、GPU1MiB25度、余盘183630184448B；账本2569无预留、r1余2074。

[部署原件](../logs/session_records/q1_h4_specialist_review_deployed_20261108.json) / [四项原运行时合同](../logs/session_records/q1_h4_specialist_review_contracts_20261108.json) / [真实准入](../logs/session_records/q1_h4_specialist_review_admitted_20261108.json)。2734B四成员原始回执包SHA `edd6e3c12e07af926a6fd41673cf2ab33e0e76311698694627fd9164544f600c`，Mac两分片精确校验/三个原件7527B逐件SHA同。此时尚未派发七例或预留新GPU预算，下一只核防重门后单派。
