# 2026-11-02 具身智能研究检查

记录日期：2026-11-02，北京时间。状态：来源检索与一项静态代码核查完成；服务器Git核验和镜像未完成，GitHub以收据为准。

## 目标与范围

接续上次成功检查2026-11-01 11:05:47+08:00，优先检查过去24–72小时的世界模型/动作跟随、重定向、控制、状态估计与遥操更新。开始已读AGENTS、当前状态、研究跟踪、简报索引与main去重；最新Q1工程记录优先于旧main工程摘要。本轮只检索、读取小文本与维护文档，不安装依赖、运行上游代码、训练、推理、仿真、操作真机或修改goal/其他自动任务。

## 完成内容

- 新简报仅一项：[FRDV运动学代码公开](research/2026-11-02/README.md)。论文v1已于10月22日报告，本轮按代码发布去重；保留旧论文记录，新增一个固定仓库提交记录，main完整去重163→164，H001不变，无H002。
- 106缓存请求105成功；一次FRDV非固定摘要连接超时，随后固定v1摘要获取成功，不删失败。核18个已有仓库头：17个与前次缓存相同，Primebot118338b与已有工程核查相同。新增FRDV作者仓库不计入这18个。IsaacLab/RSL-RL/MorphMimic发行核心未变，Primebot Releases为空，不重复发新闻。
- FRDV固定`3600f965a8e02e430fed11e42b8ed4ed1f0e5fc9`完整树209项/177blob、不截断；12份文本116014字节逐大小、Git blob与SHA核验。G1模型/样例仅核树身份，不下载；未找到Q1/A3/X2任务根或常见权重扩展名路径。MIT是代码许可，数据和SMPL+H模型须各自遵守许可。
- 核对README、标定/并行默认、AMASS输入契约、优化目标约束、SQP、输出与MuJoCo视频源码；运动学发布与动力学验证未发布分别记录。存在`dynfilter.py`不能升级为论文CI-iLQR已公开；渲染逐帧qpos/mj_forward，不能当策略物理回放。
- 更新简报索引、已有方法候选和当天来源清单。Q1只收当前研究与指针，Q1中央去重状态保持原字节，main保存完整历史。

## 工程上下文与关键决定

当前实体根仍是用户25386的`/opt/data/private/ZHY`，沿用data/logs/script/reports与大小写准确的Q1分支；hp3090仅通用研究历史。本地原始根不是Git工作树，使用既有main/Q1研究发布副本，先fetch核对最新远端。Q1发布副本此前从5610617a快进到db17f305，保留22项较新工程路径；没有拿main替换Q1工程入口。实际构建基点、工作树状态和保留检查见`logs/research_watch/2026-11-02/context_check.json`与`build_bases.json`。

近期工程按固定aa40上游/71859ca执行来源的MorphMimic G1足量复现及后续独立共享验证推进，E1024/160000轮、2000轮WM-only；本轮没有实时训练观测，不能把旧RUNNING/PID/轮数当现场证明，也不套旧300秒/1200秒训练门。共享CPU替身通信通过不等于原生Trainer/CUDA/三本体训练通过。Primebot是后续Q1优先对照，不等同论文作者原码。用户11月2日已因匹配官方policy缺失而搁置衣服演示；FRDV代码消息不改变这一决定。

选择只登记FRDV为基线后候选，因为当前公开物主要解决人体动作到机器人参考，尚缺Q1任务适配与完整动力学验证。标定、接缝和输入格式可帮助区分参考与控制误差，但不是新增参考硬门或原创创新；没有新假设、实验、训练预算、采纳检查点决定。

## 版本、命令与证据

- 简报：[当天README](research/2026-11-02/README.md)；检查清单：[research_watch_check_20261102.json](../data/manifests/research_watch_check_20261102.json)；同步：[research_watch_sync_20261102.json](../data/manifests/research_watch_sync_20261102.json)。
- 来源原件：`data/research/2026-11-02/sources/`；12文本解码：`data/research/2026-11-02/text/`；完整请求/比较/静态核验：`logs/research_watch/2026-11-02/{fetch_results,source_comparison,source_inspection}.json`。
- 本地只读核验命令：`python3 logs/research_watch/2026-11-02/inspect_sources.py inspect`。只解析本地文本和验证散列，不导入/执行上游源码；原始网络请求不要求重复运行。
- 发布前逐分支显式暂存本轮文件、检查diff并执行`python3 script/project_tools/audit_git_payload.py`；普通push main/Q1，不强推。独立GitHub ref与固定内容核验记录见当天`documents_publication_verification.json`和`receipt_publication_verification.json`。

## 结果与边界

作者仓库11月1日创建并提交，11月2日首次核实已公开；创建时间不是精确转公开时间，提交信息v1.0不等于已有GitHub Release。论文原始提交时间10月20日16:22UTC/北京时间10月21日00:22，既有10月22日正文阅读证据不改为今天新全文核验。

RO/LG/CV的new/recent共六个列表仍显示10月30日；189/485、608/1438、374/926个条目顺序均与上次一致。RSS标记11月1日但为空。公告、替换版本与历史缺口保留，不声称覆盖之后全部新研究。本轮未查HF资产，也未新增已完成资产核验的断言。

按用户原生路线经`hp_han`进入Windows，内层25386一次严格主机键校验的BatchMode只读hostname请求返回`Permission denied (publickey,password)`；未执行任何容器命令，Windows会话已正常退出。与上次同类自动认证缺口，未反复探测、索取/保存凭据、改认证或恢复计数，也没有建立转发/旧socket。服务器Git未知/写入0，不推断服务器宕机、密码错误或训练终态。原始证据摘要：`logs/research_watch/2026-11-02/native_connection.json`。

## 保存与同步

当天简报/检查/小清单进入main与Q1各自入口；main完整去重与候选历史仅在main及Mac更新，Q1中央状态不改。提交前后保护AGENTS、衣服/训练/共享/Primebot阶段文件及连接恢复计数，检查工程副本未被本任务修改。较新并行工程内容按原分支保留。

Mac镜像须在远端固定文件核验后执行：对当前入口仅增加研究指针，保护本地更新；索引用并集保留历史，完整去重从main有界更新。GitHub和本地实际完成状态以同步收据及最终日志为准；服务器镜像延续11月1日缺口，本轮写入0，不称三端完成。

## 下一步

研究候选留待基线后评估；本轮不启动FRDV转换或恢复衣服演示。下次从最新AGENTS/Q1当前状态及main去重恢复，关注FRDV真正动力学验证发布、Q1配置/资产变化及当前动作跟随瓶颈。服务器只在可用的授权原生会话中补小研究记录；同一认证缺口不重复通知或修改训练。
