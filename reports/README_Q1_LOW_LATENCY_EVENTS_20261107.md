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

部署前Mac仅运行新增5项检查，全部通过/0跳过/0.002秒：精确总时限与清理门、存活/其他失败/单阶段超时拒绝、缺片或失败无全组均值、重复片不计覆盖、原完整入口仍拒绝FAILED。没有重复此前7项或45项未变检查。新入口要求退出/21组动作门/32k审计/全部来源重核，逐例核78视频当前大小SHA、trace有限性与可见性/时钟；300秒只读分析上限，不依赖服务器Matplotlib。该准备状态的后续实际执行见下节。

部分分析可报告已经完整的事件与三固定片配置，但不能给缺失配置编造均值、完整84比较或自然遥操验收。是否采用新独立有界方案补六例，须先保存本失败根、来源与资源审计并发布，不能直接重发原controller或热改其总预算。

## 保存与同步

本阶段记录、索引、CURRENT_STATE和原始JSON按范围审计后显式提交并推送origin Q1，独立核远端40位提交及Mac主根大小SHA；实际完成以本次交付包旁的publication收据为准。服务器文档镜像待补。原约15MB历史证据ZIP保持不变，新增媒体仍服务器。应用goal保持用户清除，不调用goal工具；不新增agent，不修改其他自动任务，用户采纳仍null。

## 部分分析已实际完成

已发布准备提交`d1c3bae718cae6e96a1fceed15b7215a480f58e9`，显式push origin Q1后独立40位远端核同，8份714564B Mac主根镜像匹配。仅新增两文件的4843B压缩包沿现有逐跳密码SSH会话89657到原Linux，包/源码SHA均同，目标不在221执行或200冻结来源中。原运行时新增5检查全过/0跳过/0.003秒；原未变7项和45项未重做。

独立只读任务330423/start29605111于00:38:52派发，外层300秒timeout/5秒清理保护，6.460222秒实际完成`PARTIAL_ANALYSIS_PASSED`。78/78可用例完整，缺六例；全部读取输入前后SHA同，视频当前大小SHA与原完整解码记录相同。新增推理/积分/训练均0。原R2结果仍FAILED，原正式84分析运行0，正式source_integrity仍未通过。

### 80ms阻尼与200ms基线

每行都是同一事件下前走/起走/停走三固定片的均值，不是全393库或实测人体：

| 合成事件 | 上肢RMSE 200ms→80ms(rad) | 上肢变化 | 双脚平均承重滑移200ms→80ms(mm/s) | 滑移变化 |
| --- | ---: | ---: | ---: | ---: |
| 急停 | 0.08182→0.08216 | +0.42% | 24.69→22.29 | -9.73% |
| 反向 | 0.09005→0.09037 | +0.36% | 15.77→12.58 | -20.23% |
| 丢包 | 0.08120→0.08205 | +1.04% | 29.53→22.37 | -24.23% |

这里减少的是需要真实未来输入的预览等待，从200ms到80ms，少120ms；不等于机器人端到端少120ms或真机延迟80ms。阻尼候选在这些离线片上保持相近上肢误差并降低平均滑移，支持继续验证，不能称整体质量更优。

共同capture时钟下事件开始后1秒的上肢RMSE均值，急停0.07742→0.07826、反向0.07747→0.08098、丢包0.09199→0.09301rad，均有回退；反向窗口比整片均值更敏感。侧倾RMS均值也从0.07055/0.04616/0.07178变成0.07088/0.04841/0.07479rad。全片均值不能掩盖这些代价。

丢包前走片中，两配置均有200ms合成fallback；包窗口结束后恢复为H10的180ms对H4的60ms。这是保存合成控制时钟的调度差，非硬件实测。恢复动作最大变化约1.707对1.691（canonical29控制值），未证明真机关节冲击安全。

78例去首调用后共41106次软件目标调用，336次超过20ms，最大565.169166ms；首帧之外的长尾仍存在，不能只凭预热或p95称实时deadline已解决。脚底高度、换脚、侧倾等完整数值保留，急停/反向参考合成冻结/倒放的动力学可行性未验证。原模型仍未证明已训练短预览输入分布。

### Mac工件与传输边界

Mac交付目录：`/Users/yu/Documents/ChatGPT/元启/data/deliveries/Q1_EVENT_R2_PARTIAL_20261107`。四份数值/审计/启动记录逐份大小SHA核同；CSV传输文本的CRLF正规化用标准CSV解析与序列化恢复，最终与原26970B/SHA完全一致。源summary471062B/SHA `78985d701c1a00b0e6ee631302716e23b091e65c50a3bbd177496ee0dbc6bfde`。

已复用本任务原隔离环境中的Matplotlib3.9.4/NumPy1.26.4，pip确认依赖均已存在，没有新增包或修改服务器依赖。读取保存数值生成12张六宫图PNG、12份PDF，共2662283B；标题标PARTIAL和7of7/5of7配置，缺阻尼六例未画成零值或均值。12图总览及丢包/突发延迟两张原图已目视，无空白或文字遮挡，生成前后summary SHA同。

六段代表原速视频已在服务器归档`/opt/data/private/ZHY/data/deliveries/Q1_EVENT_R2_PARTIAL_RAW_20261107.tgz`，2991565B，SHA `e5a4bf494c82297e56f752ec2e32f1ecfb3ce2e20246f8c6a055503e14aab574`。Mac目标为主根`data/deliveries/Q1_EVENT_R2_PARTIAL_RAW_20261107.tgz`，当前待传，不能称已交付视频。

取视频的新原生密码会话90853在Mac→Windows认证后Broken pipe255；唯一有界重试83393认证后peer reset255，均未取得Windows shell。失败只涉及这两个新会话，原89657随后确实返回数值，不概括为整桥接或Linux不可达。未换认证/端口/转发，不再循环新认证。旧15MB证据ZIP不改写，新图表包单独交付。

[结果、限制与交付审计](../logs/session_records/q1_event_partial_results_20261107.json)。原共享训练继续暂停、服务器新增训练0/4GPU小时预算未用、应用goal保持清除、用户采纳null。下一步先分析保存的长尾时序并制定只补六例的独立有界方案；没有自动重跑84例或恢复原160k。

新图表压缩包为Mac主根`data/deliveries/Q1_EVENT_R2_PARTIAL_FIGURES_20261107.zip`，3109491B/SHA `f9419ec9ac54ff405b81cc9a1d1a7e6df3ed13021c9502eeaf4bdf5af6943342`；34文件含README、12PNG/12PDF/总览/原始部分分析与时限证据，CRC及每成员大小SHA、7相对链接全部通过。包中新增事件MP4为0，视频仍待补，不覆盖旧ZIP。[交付逐文件清单](../data/manifests/q1_event_partial_delivery_20261107.json)。

q1仅更新prompt为本次实际失败、已完成部分分析与后续入口；工具回执ACTIVE，文件独立核heartbeat/十分钟/同聊天不变，配置SHA `d4faa306ba97e320aa9148a47aa3d65a8e0b2ef5d67086ba9d53f1a9a8adff08`。未改其他自动任务，未调用goal或新建agent；本次更新不是新训练或原任务重启。

00:57:26最后只读回执由89657返回，909B与SHA `4b10f9fe0cf2b853b9e09fa9ff0e1b82b2f566165d94c82e448f86f5c96b7175`解码核同：原boot/GPU相同，原FAILED结果SHA未变，五旧训练PID、事件controller/worker及部分分析PID均已退出，原训练guard仍STOPPED_SIGNAL。总7751073792B/限34292629504B/OOMkill0/under_oom0，GPU1MiB/26C/Recovery None/计算进程空；部分分析审计仍PASSED且新增推理/物理/训练0。[最后现场](../logs/session_records/q1_event_partial_final_observation_20261107.json)单独保存，不修改已封存图表ZIP或其成员。
