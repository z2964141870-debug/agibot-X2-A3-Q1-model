# 2026-11-15 研究检查

触发 `2026-11-15T01:02:14.213Z`，北京时间 09:02:14。完成一项实质源码更新的静态分析，决定 `NOTIFY`。见[简报](research/2026-11-15/README.md)、[来源清单](../data/manifests/research_watch_check_20261115.json)；实际 Git 提交与独立校验由[发布收据](../data/manifests/research_watch_sync_20261115.json)记录。

## 目标、已读入口与工程背景

读取本地 AGENTS、reports/README.md、README_CURRENT_STATE.md、README_RESEARCH_WATCH.md、research/README.md、README_IDEAS.md 和 main 的 research_watch_state.json；核 Q1 最新保存入口及人体条件/多步 WM、X2 启动补测、桥接诊断、衣服链路复核阶段。上次成功研究检查 `2026-11-14T03:09:14.784+00:00`，去重 262 条。

主目录不是 Git 工作树。main 独立研究副本起点 `90bb3964c533132c1563738110e941fcde21344c`；Q1 只快进到远端 `5cd6280d08f8f31e229698f5a085347466970af6`，两副本写入前均干净。只读 x2 引用 `79c4a1e7c019fe57130a2f530fcec78875eb7e4f`，本次不写该分支。main 维护完整历史和去重；Q1 只接本期及指针，保留其工程入口与中央去重文件。

最新保存记录覆盖旧摘要：

- Q1 仍为 huanyu:`/opt/data/private/ZHY`、大小写准确的 `Q1`。人体条件两组各 3000 更新已完成、均选新增 0，不能据选 0 推出所有非零点被拒。恢复先原 CPU 审计、七 JSON 与逐点曲线，再已有多步 WM 来源/资源准入；不重训 6000。原暂停、采纳 null、5036 秒台账保持。
- [X2 R2 启动补测](https://github.com/z2964141870-debug/agibot-X2-A3-Q1-model/blob/5cd6280d08f8f31e229698f5a085347466970af6/reports/README_X2_ALLOCATION_STARTUP_20261114.md)：64 前缀及实际 TID 亲和性已通过，原 joint_r2 初始化失败保留；独立 startupfix 唯一派发 PID35179/start247845、monitor35181 已有回执。终态、源终门、回收和性能未核，不能继续写“仍准备”，也不能重派。本研究不执行压测、推理或设备命令。
- [11/15 桥接诊断](https://github.com/z2964141870-debug/agibot-X2-A3-Q1-model/blob/5cd6280d08f8f31e229698f5a085347466970af6/reports/README_X2_BRIDGE_DIAGNOSTIC_20261115.md)保存第一跳超时及分层观察，不能据此诊断 Windows 卡死或整桥接故障。本轮不重复认证；服务器 Git `NOT_INDEPENDENTLY_OBSERVED`、研究镜像 `PENDING`。GitHub 核验不作服务器/机器人现场证明。
- [衣服链路复核](https://github.com/z2964141870-debug/agibot-X2-A3-Q1-model/blob/5cd6280d08f8f31e229698f5a085347466970af6/reports/README_EGOLOCATE_FGP_CHAIN_RECHECK_20261114.md)：网页自定义后端使用 U100h FGP→Full_GR_OV_10，绕过原生 FastPIP；23MB PIP 文件不能代表整条链路。当前加速 CANO S1/LIP/GP 另列。180ms 预览、默认和原训练不变。

## 检索覆盖与失败

131 次 HTTP 尝试，80 份完整 200，包含恢复请求而非 80 篇新文；所有存在的缓存重核实际大小和 SHA-256。初次 GitHub API 的 47 次超时保留；强制解析路径不通后，用系统原有解析器做一次有界探测成功，再恢复其余小请求，没有修改系统 DNS/SSH/认证配置。公开 CLOTHO 猜测地址及 SuperMimicHP API 404，只说明该公开入口未取到，不否认工程来源。Giga 旧树/README blob API 两次 403，通过固定 Git 对象补齐；不把失败响应当源码。网页工具未返回可用结果，不虚构引用；事实依据为保留的原始 HTTPS 响应和 Git 对象。

RO/LG/CV 的 new/recent 最新可见 11/13，与 11/14 检查相同，新增 listing key 为零；RSS 的 pubDate 为 11/14 00:00:00 -0500、build 为 05:00:00–03 UTC，条目零。不能据此推出 11/14–15 没有新论文或已完整覆盖公告。34 个共同仓库头只有 Giga 变化；11 组发行核心列表相同，17 个页面文本相同。Giga 论文摘要仍指向 v5，11/5 修订；未重新包装旧稿为新方法。

## 固定源码核验及结论边界

作者仓库 `open-gigaai/giga-brain-wbc`：旧 `fa44c2418e7be86c97655e5cb0c809213b2c7dca` 是新 `73d58601fa7e6b9a3d72d06a40a1cf6088803314` 的直接父提交；作者/提交时间 `2026-11-14T10:16:28Z`，本次 11/15 首次发现。六个文件有变化：`.gitignore`、tracking/l01 配置、observations、token_losses、universal_token_modules。未改 README/许可/选读 trainer，未附新基准或权重工件。

API 递归树顶层 sha 意外等于提交号，而提交元数据和原生 Git 给出的树为 `729d0a423c21177e8ccb32c8c4c5793c7dbfddec`。未把前者当正确树 SHA：独立浅取新旧提交，比较全部 627 项 path/mode/type/object，与 Git 树完全相同；该元数据差异原样登记。旧树 403 不用于验证。仅补九份选读文本的新旧 blob，共 344429B，当前 201681B；七个 API blob 与 Git 逐字节一致，SHA 和大小通过。对 Python 文件只 AST 解析，没有导入或执行作者代码。

已读调用路径、配置和完整相关差异：

- `block_mean` 是各语义块均方误差按权重求平均，新增分项记录。权重会改变目标和梯度；维度处理不等于物理单位归一化，更不证明避免梯度冲突。配置注释的误差占比/梯度预算没有原始实验核验，未作性能结论。
- `aux` 组关闭观测噪声，按策略项顺序构造干净下一状态标签；L01 同步关节顺序。`SPredLoss` 的 t 预测对齐 t+1 标签，沿当前回合掩码屏蔽跨回合/预热配对。缺干净目标可回退原观测，不能说所有配置都已清洁。动作在构造辅助损失输入前已解码；未在所读路径把答案作为新动作输入，未验证运行或 ONNX。
- 根目标来自机器人世界根 pose，转成当前根坐标系的位移和相对旋转（3+6）；G1 示例 64→73 辅助输出。旧 strict-load 需显式关根扩展或处理形状；未测试加载、部署兼容或训练收益。此辅助预测不证明 Q1 的多步物理模型或动作梯度可靠。
- 项目代码 Apache-2.0，第三方与资产保留各自许可。只核本次源码，没有下载/验证模型与动作资产，不声称有可用 Q1 实现或新权重。

仅登记一项阅读候选，main 去重 262→263；H001 保持，新增假设 0。它映射到已观察的“总拟合下降但旧能力退步”，不作为该现象的原因判断，也不引入新训练门。

## 产物、复核与下一步

原始响应和小型 Git 对象在 `data/research/2026-11-15/`；请求时间/URL/错误、解析、SHA、源码差异和发布日志在 `logs/research_watch/2026-11-15/`，均不作为原始代码提交。复核入口：`watch.py inspect`、`verify_sources.py`、`publish.py`。只执行本地结构化读取和 Git 载荷审计，未安装依赖、下载大模型、启动训练/仿真或操作真机。

仅暂存本次显式研究文档与小清单，检查 diff 和 `script/project_tools/audit_git_payload.py`，普通推送 main/Q1，再独立取远端引用和所需 blob 逐字节核验。工程入口在各自版本上前置研究指针，不用旧镜像覆盖；Q1 中央状态保留原始字节。GitHub 实际完成状态以发布收据及独立日志为准，huanyu/设备镜像仍待。

下一次从本期成功检查时间接续，有实质变化才再报同仓库；当前优先原 Q1 训练证据取回与已有多步诊断，X2 先取唯一 R2 终态，本研究不派发工程。
