# X2 Windows 桥接在线状态与实际可达性

2026-11-15，北京时间00:18至00:24只读诊断。结论是当前Mac没有收到目标Windows的数据响应，尚不能区分Windows休眠/卡死、远端网络或Tailscale故障、本机至节点的路径问题；没有证据认定仅SSH服务异常，也不能把设备列表绿点当本次SSH连通证明。

## 实测证据

用户截图中`desktop-lql43si`、`100.92.227.62`显示绿点。本轮本机CLI读取同一节点却是`Online=false`，`LastSeen=2026-11-14T16:10:00.1Z`，即11月15日北京时间00:10:00.1；这是状态来源的不同观测，不直接断言截图缓存原因或Windows死机。

| 检查 | 实际结果 |
| --- | --- |
| 原生`ssh sshuser@100.92.227.62` | 会话33296，TCP连接Operation timed out，退出255；未到密码提示，未尝试Linux第二跳 |
| 普通ICMP ping | 3次发送、0回复、100%丢失；单项不证明关机 |
| Tailscale默认ping | 3次，每次5秒上限，全部无回复 |
| Tailscale TSMP ping | 3次，每次5秒上限，全部无回复 |
| 本机Tailscale状态 | BackendState Running、Self Online=true、Health空列表；不等于任意节点数据路径通过 |
| 对照hp-han | 状态Online=true，但两次5秒Tailscale ping也未回复；不是该节点操作授权或SSH测试，没有登录它 |
| 本机netcheck | 退出0、UDP=true、IPv4可用、CaptivePortal=false，多处DERP测速有响应；最近测速点Los Angeles 177.1ms，目标所在sin测速275.9ms，hkg没有测速值。仅此诊断样本，不归因为代理、地区选择或特定服务 |
| 本机目标路由 | 指向utun4；未核该接口与扩展的对应，不能据接口名认定路由正确或冲突 |

两次读取目标状态均Online=false且LastSeen相同，RxBytes=0；后一次TxBytes=6084。Active由true变false是观测值，不把它解释为认证成功或端到端连通。对照节点同样无回应，故保留更广的数据路径问题，不能只归责目标Windows。没有更改路由、认证、服务、功耗或机器人进程，没有新派发推理、反馈或训练。

## 恢复与待确认

一次原生SSH恢复已经使用，不因无响应反复认证或换转发入口。若现场Windows仍能操作，下一有区分力的证据是该机本地的Tailscale连接状态、网络是否正常以及Tailscale/sshd服务状态；若本地也不响应，才有直接支持卡死或休眠的证据。本次没有重启任何设备或服务，也没有将这种假设记录为结论。

原R2补测仍为已派发、终态未核验，PID35179/start247845仅历史定位；恢复后仅读原输出和monitor、来源终门及自有进程回收，不重派或声称补测失败。见[补测恢复点](README_X2_ALLOCATION_STARTUP_20261114.md)。本轮[小记录](../logs/session_records/x2_bridge_diagnostic_20261115.json)不含认证信息。

本阶段报告、索引及记录在Mac主根与权威Git镜像；Git发布以后续实核为准，设备文档镜像待连接恢复。没有新增六宫图或性能结论。
