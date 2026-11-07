# H4专用编码器：七例匹配闭环验证

2026-11-08，北京时间。检验H4-only继续适配的0.923%验证收益是否成为闭环质量收益。新权重固定为H4_SPECIALIST01的validation-selected1500，不读取last或按这七个场景换选点。本阶段是已见片段的探索性诊断，不作新留出集或自然遥操合格声明。

## done / todo

done：训练本身已完成，训练六宫图和307711B证据包已交付，结果发布0e7a3ea远端40位与Mac八文件820432B核同。独立student_h4_review入口、固定plan和四项CPU合同实现；四检查0.002秒通过，AST与CLI通过。现有45014仍正常返回原Linux，03:51准入前快照总8154480640B/限34292629504B、OOM0、GPU1MiB25度、余盘183630389248B，原五PID和H4训练三PID均退，原GPU锁可独占获取。

todo：发布源码和清单，服务器原运行时合同与来源/资源/成本准入后，核唯一output/log/dispatch/精确进程防重再派一次。真实新物理、渲染及GPU调用仍0。完成后只读分析保存工件、生成七组对比六宫图及原速媒体，完整性与失败同时报告。

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
