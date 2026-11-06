# Q1 桥接观察与判断规则

日期：2026-11-06。23:51:01北京时间已重新进入原Linux并取得真实回执；本次是在用户明确告知当前可连接之后进行的一次原生密码恢复。

## 纠正

此前实际证据是观察用SSH客户端报错退出，后续一次登录也未收到Linux shell就被外层Windows连接超时终止。将其概括为整条桥接不可达超出了证据。23:38心跳没有重新登录，只读取已有故障记录，不能将该轮等待状态当成当时网络不可达的独立验证。

用户自己的终端可连接和某一旧SSH会话退出可以同时成立。旧会话失效不能证明新登录必然失败；退出码255本身只表示SSH错误，不能单独判定密码错误、具体故障跳或Linux停止。

## 使用的证据

| 实际证据 | 记录方式 | 后续动作 |
| --- | --- | --- |
| 暂无新输出、认证后尚未收到shell、正常命令仍在运行 | 等待或状态未确认 | 在当前有界等待内继续读取，不据此宣布断联 |
| SSH明确报Timeout/Broken pipe/reset并且客户端退出 | 该SSH会话退出，注明进程及报错地址 | 保留其已退出事实，不推广为服务器不可达 |
| 新原生登录在某跳明确失败 | 本次Mac到Windows或Windows到Linux登录失败 | 记录实际到达的提示/哪跳失败，不把外层错误归到内层 |
| 本轮没有重新测试 | 当前可达性未核验 | 使用带日期的最后成功快照，不伪称实时监督 |
| 新登录取得Linux shell和服务器身份/命令回执 | 本次连接已恢复，注明采样时间 | 复用该会话，先核原身份、暂停、任务和资源 |

后续仍沿原地址`yu@100.65.183.91`原生SSH交互输密码，再在Windows执行原生SSH到`root@59.77.15.15:25362`并交互输密码。密码提示立即处理；不从文件、镜像取认证，不引入其他认证或路由。记录不包含认证内容。

## 本次等待边界

新会话89657两跳只在本次命令行设置：

```text
PreferredAuthentications=password
PubkeyAuthentication=no
NumberOfPasswordPrompts=1
ConnectionAttempts=1
ConnectTimeout=90
ServerAliveInterval=60
ServerAliveCountMax=5
```

没有改用户全局SSH配置。ConnectTimeout控制建连和初始SSH握手/密钥交换；ServerAlive是未收服务端消息时发出的加密通道探测，连续无回应达到数量后结束该会话。按本次60秒和5次设置，失响应容忍约300秒；它不是普通命令没有终端输出就必然失败的计时器，也不是机器人执行超时。

上轮18867使用ConnectTimeout30/ServerAliveInterval30/CountMax3，对持续失响应约90秒后结束该会话。本次放宽等待以适应用户描述的高延迟，成功连接不能据此确定此前故障的唯一原因。具体是网络、客户端、Windows或其他因素仍缺少足够的分段证据。

OpenSSH一手说明：

```text
https://man.openbsd.org/ssh_config.5
https://man.openbsd.org/ssh.1
```

## 本次现场

89657已先后收到Windows PowerShell和`root@interactive67462:~#`。23:51:01服务器回执核原boot`90395f47-4035-4785-858e-b3d05c44a738`、GPU`GPU-c602a45a-0dd8-9ad2-9adc-4f863facd1d4`和唯一根一致，五旧训练PID54868/55672/55677/55678/55679均不存在，原pipeline FAILED与guard STOPPED_SIGNAL仍对应此前用户主动暂停。

原事件控制器280863/start28796218仍运行并持原GPU锁。任务RUNNING/109已完阶段，当前丢包hold-H4前走；stop21/21、reverse21/21、dropout12/21物理媒体完整，共54/84，late_burst尚未开始。原任务未重派，原训练未恢复，没有新训练更新。

总量10704003072B/实际限34292629504B/即时余23588626432B/OOM0，GPU资源样本400MiB36C/随后即时37C/Recovery None，峰总量10735480832B；训练信号0、回收未尝试。完整84例、来源终门和正式分析仍待，不提前判自然遥操合格。

q1仅更新prompt，加入新会话、以上观察规则和已部署分析入口；名称、ACTIVE、十分钟和原聊天不变，其他自动任务、应用goal均未改。保存文件关键字段、规则和SHA已重新核验。真实收据为[q1_bridge_judgment_corrected_2351](../logs/session_records/q1_bridge_judgment_corrected_2351_20261106.json)，分析和后续交付按[事件README](README_Q1_LOW_LATENCY_EVENTS_20261106.md)继续。新文档服务器镜像待补。
