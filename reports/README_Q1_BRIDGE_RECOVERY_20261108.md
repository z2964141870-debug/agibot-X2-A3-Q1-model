# Q1固定入口观察恢复

日期：2026-11-08，北京时间。状态：三项用户批准设置已生效；随后固定入口45014已成功进入原Linux，三份原准入回执已传回Mac并核原始字节。00:20前只读核实原训练仍暂停、旧五PID退、锁及GPU计算空、资源过门，唯一七例输出/派发/日志/进程全部不存在。下一发布恢复点后单派，尚未提交Popen。前两次失败保留，不归因于单一参数。

## 用户简报

done：Agent自行固定source/ssh两次到原Linux后，查明hp_han原超时8秒、保活20秒×3。用户明确批准后仅改为90秒及60秒×5，其他配置行SHA前后一致；实际ssh -G已核生效。原训练完成、selected500准入及分析代码发布的证据保留。

to do：复用45014，发布原准入回执与本次现场后单派七例，闭环完成再只读分析28例保存比较、交付六宫图和原速视频。原连接命令、主机及密码认证方式保持，不重复已成功训练和检查。

## 固定入口实际恢复

45014仍原样source/ssh进入Windows，再原生交互密码进入root@interactive67462。原boot、GPU和唯一根均同；本会话关闭终端输入回显后短命令已返回，未据此认定故障根因。原回执归档首次实际证不存在，才创建2124B三成员包；SHA `e913ae61f64c41e31d648415efa6dec1c29dc25dcd6b7a0a84774e3cded772a2` 经当前会话48字符base64行传回Mac并核同，三原JSON共5744B完整解包。

当前cgroup总7999332352/实际限34292629504B、OOM0、GPU1MiB/25度/None、磁盘余183916695552B、private分配266912190464B，原GPU锁空和计算进程空。paused_state实际核原五PID退、原pipeline人工停止及guard STOPPED_SIGNAL；账本2181秒且无RESERVED。唯一新review的输出、派发、日志与精确控制器进程皆不存在。实际证据见[恢复回执](../logs/session_records/q1_coverage_review_recovered_20261108.json)。不重做成功部署/四检查/CPU模型审计，原训练不恢复。以下两次失败属于恢复前历史。

## 实际观察

44573在原Linux已返回项目目录提示；首条只读核验误用系统不存在的`python`，未执行审计。改用既有环境绝对解释器后只读命令已提交，但未返回结果即该SSH明确Timeout255。因此没有新的boot/GPU/资源/工件回执，不能把旧准入现场当当前状态。没有提交Popen或训练评估信号。

本次唯一有界恢复98513仍原样入口，再次进原Linux。进入项目目录的命令未收到回显，一次追加Enter后仍等待，随后该会话明确Timeout255。静默期间没有判断断联或重复命令，两句柄现已结束，不再探测。后续连接由Agent自行执行，不要求用户代测。

只读`ssh -G hp_han`及配置来源检查显示：实际`ConnectTimeout=8`、`ServerAliveInterval=20`、`ServerAliveCountMax=3`、`ControlMaster=false`。实际ssh为用户zshrc载入的shell函数，最终调用系统原生ssh；本轮只读了路由及超时设置，未获取认证材料。该配置可能影响慢链路耐受性，但不能据此认定全部故障单因；尚无修改后对照。

## 受约束的改动方案

| 仅Host hp_han | 当前 | 拟议 |
| --- | ---: | ---: |
| ConnectTimeout | 8秒 | 90秒 |
| ServerAliveInterval | 20秒 | 60秒 |
| ServerAliveCountMax | 3 | 5 |

第一跳仍原样`source /Users/yu/.zshrc`和`ssh -tt hp_han`，第二跳仍原生交互密码。AGENTS.md原约束“不读取/记录认证或改全局配置”，用户最新明确允许上述三项具体例外，其他约束保留。用apply_patch精确替换第5至7行数值，排除这三行的其余配置SHA前后均为`192f66bd6913c96473a242b8e3eeb5e37048405859d2111ad799319eead0b4cf`；ssh -G确认新值生效，ControlMaster仍false。32715自主原命令首次测试在第一跳TCP等90秒后明确超时255、未Windows认证；21051本次唯一有界重试第一跳reset255/未Windows shell，未尝试Linux。两句柄已结束，不探旧句柄或同周期循环认证。没有要求用户代测，也未记录认证材料。

## 训练与下一步

[新候选准入及七例计划](README_Q1_COVERAGE_SELECTED_REVIEW_20261107.md)及[上阶段发布核验](../logs/session_records/q1_student_coverage_selected_review_publication_20261107.json)保留。原共享160k训练不恢复；本轮新增模型训练、物理回放及渲染均未派发。COVERAGE01扩展验证改善仍不能当闭环成功。

恢复后优先使用短命令分开读取原回执和现场，原成功部署/四运行时检查/已选CPU身份审计不重复。若输出或派发已经存在则只观察同一任务；未知派发不得重发。原22/24GiB、18GiB/80度、100GiB/private500GB、原非阻塞GPU锁和来源门保持。新文档服务器镜像待连接，发布及自动任务恢复点以后续收据为准。

证据：[本次两会话与只读配置](../logs/session_records/q1_bridge_timeout_config_observed_20261108.json)。
