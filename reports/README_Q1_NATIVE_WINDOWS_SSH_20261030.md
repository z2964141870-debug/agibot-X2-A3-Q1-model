# Q1原始Windows两段式SSH

2026-10-31最新授权：用户要求检查到25386等既有通道未恢复时，通过Windows跳板重新打开一次。执行含义为重建原生SSH连接会话：先ssh hp_han，再在Windows内ssh -p25386 root@59.77.15.15。每个独立故障允许一次自动恢复，无需重复询问；尝试前在logs/q1_connection_recovery_state.json登记故障标识和次数，同一故障的后续心跳不得重置计数。成功后核原boot/GPU/唯一根并继续既有维护，仍失败则一次有界只读核查后保持UNKNOWN并只通知新增异常。失联本身不授权重启宿主机、容器或训练任务。

本次旧断连已按新授权一次恢复：2026-10-31 16:17:53北京时间核interactive40173、原boot/根/GPU一致，GPU15MiB/0%、无计算任务，OOM0；16:16:14未见项目训练/渲染进程。容器总4446236672B、文件cache3719962624B，均为该时刻快照。本次只连接与只读状态检查，未执行缓存扫描或新物理/推理/训练，原失败和采纳null保留。q1已通过原生automation_update更新并读回，仍ACTIVE每30分钟/同聊天；其他自动任务未改。[新授权与现场清单](../data/manifests/q1_native_recovery_policy_20261031.json)。下文2026-10-30连接历史和交付记录保留。

日期：2026-10-30，北京时间。状态：原生两段式已登录原25386，补齐此前7个待镜小记录。用户明确“别天天用转发啊，用原始的连接方式”，覆盖旧本机18873/控制socket连接偏好；训练、观测和目标范围不变。

## 已验证路径

Mac加载既有自动登录助手后，原生登录Windows：

```sh
source /Users/yu/.zshrc
ssh -tt hp_han
```

Windows返回DESKTOP-3C1DPPR，在其PowerShell中用系统SSH进入同一容器：

```powershell
ssh -p 25386 root@59.77.15.15
```

本次没有创建或依赖18873、-L/-R/-D转发或ControlMaster/socket。以后只检查实际两段SSH返回，不因本机无监听判服务器失联；连接异常按上方2026-10-31新授权每个独立故障经Windows恢复一次，再按实际结果继续维护或保持UNKNOWN。

## 严格公钥校验

Windows原无此25386的ED25519记录，StrictHostKeyChecking=yes正常拦截。只复用Mac已信任的精确25386主机公钥，由Windows原生KnownHostsCommand在进程内返回，保留严格校验，不改持久认证文件。指纹SHA256:ptg8SMNQvUQYHV0/hafU4hBlhZ/vb4wJUmM1XTVlROI。已有可信主机记录可用上面的原始命令，缺少时用本次成功方式：

```powershell
$Q1KnownHost = '[59.77.15.15]:25386 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILSnz7OktfsDmOU13oL2sqyKrw0S7SKDo2v//baPDW1h'
$Q1PinScript = "[Console]::Out.WriteLine('$Q1KnownHost')"
$Q1PinEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Q1PinScript))
ssh -p 25386 -o ConnectTimeout=8 -o ServerAliveInterval=20 -o ServerAliveCountMax=3 -o NumberOfPasswordPrompts=1 -o StrictHostKeyChecking=yes -o UserKnownHostsFile=NUL -o GlobalKnownHostsFile=NUL -o "KnownHostsCommand=C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe -NoProfile -NonInteractive -EncodedCommand $Q1PinEncoded" root@59.77.15.15
```

相对powershell.exe最初被原生SSH拒绝，改成上述绝对路径后校验与认证正常，进入root@interactive40173。公钥不是密码或私钥；沿用已获用户提供的交互认证，容器密码未保存，认证内容和本机配置未复制到项目或镜像，Mac助手保持原样。

## 身份与补镜

只读核对原boot4d859008-4c83-4a6d-814d-cbb9dfe885c9、唯一实体根/opt/data/private/ZHY、GPU-26028785-9fb9-45af-e614-4cc53af15d08，GPU15MiB/0%、无计算进程。这是身份快照，不是新物理审计；本次未核新的OOM计数，不沿用历史OOM0冒充当前。

2026-10-30 18:11:07北京时间，36daafb10a4a1b7ee87c7aad48853b0a0dad166f的7个小记录366473字节已通过同一原生会话补到25386；每份先核父版/当前SHA，再原子复制并核大小/SHA，Mac主根与权威Git blob/工作树亦一致。实际完成标记Q1_NATIVE_MIRROR_VERIFIED，精确名单在[连接清单](../data/manifests/q1_native_windows_connection_20261030.json)。原14观测和9分析工件验收不重复，72个NPZ和权重保持SERVER_LOCAL_ONLY。

[子步报告](README_Q1_GAIT_GOAL01_SUBSTEP_OBSERVATION_20261030.md)与[原发布收据](../data/manifests/supermimic_q1_gait_goal_substep_observation_publication_20261030.json)补记恢复状态。连接说明本身仍按README/显式stage/payload audit/commit/push Q1、独立40位远端再镜像；最终证据见q1_native_windows_connection_publication_20261030.json，不提前称新记录三端发布完成。

## 定时维护与边界

已用应用automation_update更新原q1并读回automation.toml：仍ACTIVE、每30分钟、同目标聊天，仅连接与主机公钥校验改成本文两段式。缓存阈值/目标/范围/时限、GPU和空闲保护、静默通知原则不变；原训练监督、yuanqi、supermimichp-x2及其他任务未改。按OpenAI Docs技能和应用工具操作，未手改automation.toml。

本次未新增缓存扫描、预检、仿真、策略推理、优化或训练；已用完的两条观测及保存CPU分析均未重做。原901.178391956928实训秒/physical_regression、采纳null和GAIT_GOAL01 ACTIVE未达保持，不结算或暂停目标，不开agent。

首个Python身份打印多了右括号而失败，后续独立原生命令才取得实际身份；本机机械传输包首次输出超限发生在服务器写入前，随后分块取包并完成核验。上述错误与公钥路径错误保留在[会话日志](../logs/session_records/q1_native_windows_connection_20261030.log)，不据失败输出声称成功。

后续维护先读AGENTS、当前状态、本页和[缓存说明](README_Q1_FILE_CACHE_MAINTENANCE_20261029.md)，使用原生两段式；认证和大工件不入库，连接更正不授权新训练或实验。
