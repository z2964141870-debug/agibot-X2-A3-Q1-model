# Q1低延迟事件R2时限退出与保存结果

记录日期：2026-11-07。实验：32k human事件R2。状态：部分完成；正式campaign失败，原因已确认。

## 目标与现场

继续四种合成事件、七种配置、三固定片的84例CUDA策略/共同Primebot CPU物理/NVIDIA EGL原速媒体比较。原共享训练保持用户暂停，本轮没有新增训练或重派事件。

原生逐跳密码会话89657仍在原Linux。00:28:28核原根、boot与GPU身份同；五个旧训练PID均不存在，原STOPPED_SIGNAL保持。连接等待和客户端退出仍按具体会话判断，本次是实际任务时限退出。

## 已完成与缺项

| 事件 | 完整物理 | 完整原速媒体 |
| --- | ---: | ---: |
| 急停 stop | 21/21 | 21/21 |
| 反向 reverse | 21/21 | 21/21 |
| 丢包 dropout | 21/21 | 21/21 |
| 突发延迟 late_burst | 15/21 | 15/21 |
| 合计 | 78/84 | 78/84 |

缺六例均为late_burst下damped_cv：H5与H4各前走/起走/停走三片。已完成78例均完整，不能把被中断的第79例计为步态失败或完整例。

## 时限与清理证据

controller280863/start28796218于2026-11-06 22:24:05.155066开始，2026-11-07 00:24:05.337526结束，实际7200.182460秒。完成157阶段后，在late_burst_damped_cv_h5_walk_forward_loop_001__A022_replay触发：

`RuntimeError: Paused campaign whole/phase wall-time bound reached`

原Budget每次检查要求总时长小于7200秒、单阶段小于1800秒；当前阶段不足一分钟，故本次为总时限。guard FAILED，330190返回-15且worker_reaped=true。00:28实核controller、worker及worker session无存活，原GPU锁释放/计算进程空。没有扩大预算、自动重启或改写原FAILED结果。

## 来源与资源

独立只读重核221份执行源码1697002B、200份冻结源码/XML1528474B、4份控制输入49086B、46份复用名义证据13621725B，大小SHA全部相同。这是退出后的来源复核，不冒充原campaign的正式source_integrity完成门；comparison未执行，原84例分析入口不运行。

保存资源峰总量10779512832B、GPU400MiB；训练信号0/项目回收未尝试。00:28当前总量7750115328B、限额34292629504B，OOMkill0/under_oom0；GPU1MiB/26C/Recovery None，磁盘余187764174848B。原22GiB准入/24GiB退出、GPU18GiB/80C、100GiB余量等边界保持。

[原始现场与逐例覆盖](../logs/session_records/q1_event_r2_wall_bound_20261107.json)经既有原生密码会话传回，zlib/base64解码后的21899B与SHA-256 `2da0282c18a0bdb6c29fa21c9a7611600e3c6adc07271ac360e4551e4a8b8961`在Mac独立匹配。认证内容不入记录。

## 分析与下一步

原`event_summary.py`仍要求正式COMPLETED/source_integrity PASSED/84例，不能对本失败根放宽。新增独立`script/q1_checkpoint_review/event_partial_summary.py`，明确78可读、6缺失、原状态FAILED；读取保存trace和telemetry，保留事件统一时钟窗口、离地换脚/承重滑移/侧倾/上肢/动作突变/丢帧恢复及去首调用后的软件长尾。不加载策略、不积分、不产生训练更新。

Mac仅运行新增5项检查，全部通过/0跳过/0.002秒：精确总时限与清理门、存活/其他失败/单阶段超时拒绝、缺片或失败无全组均值、重复片不计覆盖、原完整入口仍拒绝FAILED。没有重复此前7项或45项未变检查。新入口要求退出/21组动作门/32k审计/全部来源重核，逐例核78视频当前大小SHA、trace有限性与可见性/时钟；300秒只读分析上限，不依赖服务器Matplotlib。当前准备完成，尚未服务器部署或实际分析。

部分分析可报告已经完整的事件与三固定片配置，但不能给缺失配置编造均值、完整84比较或自然遥操验收。是否采用新独立有界方案补六例，须先保存本失败根、来源与资源审计并发布，不能直接重发原controller或热改其总预算。

## 保存与同步

本阶段记录、索引、CURRENT_STATE和原始JSON已纳入本次待审计变更；实际commit/push及Mac镜像以随后publication收据为准。服务器文档镜像待补。原约15MB历史证据ZIP保持不变，新增媒体仍服务器。应用goal保持用户清除，不调用goal工具；不新增agent，不修改其他自动任务，用户采纳仍null。
