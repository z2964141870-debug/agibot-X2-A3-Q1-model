# 2026-10-29研究检查

状态：原始来源检查、静态阅读与本地研究记录完成；实际发布和镜像结果见[同步回执](../data/manifests/research_watch_sync_20261029.json)，未完成的步骤不预记成功。

## 目标与范围

本次接续10/28 09:26:47成功检查，触发时间为北京时间10/29 09:03:06。研究服务于SuperMimic for Q1的自然走路、脚底离地/滑移和上肢幅度取舍，仅检索和文档维护。[四项简报](research/2026-10-29/README.md)区分正式发行、独立复现代码更新和首次阅读的新稿。

## 工程上下文与决定

已读AGENTS、当前状态、研究说明/索引/候选、OVERNIGHT_RESULTS、RF16_RESULTS和overnight_research_recheck。07:00后不再新物理/优化，用户采纳null，原训练监督ACTIVE静默等待审阅；本次没有更改监督、目标或其他自动任务。昨夜完成/资源释放只作为既有证据，不作本次实时进程证明。

当前用户容器25386:/opt/data/private/ZHY可读，7份上下文/指针与Mac匹配；候选清单和中央状态在分支/主根有不同职责，未用旧镜像覆盖。服务器仍没有git可执行文件，不能声称其工作树已核为干净。已实际fetch main/Q1，两个权威工作副本均干净并与origin一致：main `534e3810f55db87667fecfcf45e8cdcfa0a1c5e1`，Q1 `cb00a4d90d5a7ca26fb3f93cc54dcd799cc5249b`。之后提交号由回执记录。

09:24通过既有环境解释器仅读取importlib.metadata和源文件，确认rsl-rl-lib为5.5.1；不导入torch/rsl、不积分、不推理。三份安装源码及四份Q1适配源码SHA存档，后四份与权威Q1 Git一致。残差配置为4096×24/4，观测用新TensorDict与拼接；没有据新发行认定旧实验根因，也没有执行新测试或升级。最早RF12收据的3.0.1是历史元数据，不再当作当前事实。

## 来源与结果

- RO/LG/CV完整new与recent、RSS均缓存；10/28公告new分别125/437/245，recent 434/1306/831。今天三类RSS数量和new相符；这不补齐历史10/8、10/13等覆盖缺口。
- 96次有界请求全部HTTP200。16个既有仓库中15个头未变，独立SuperMimic复现有6个新提交；RSL-RL新发行5.5.2。其余已查项目文本/链接和既有摘要无实质变化。
- 固定`8bfc1158bd968f729fcc65d5e897a05b05ea3d36`的7份独立复现文本、`11563319134f987f70fb034410496fe3cc7327a9`的5份RSL文本，分别对41项和138项未截断树核字节数/Git blob SHA/SHA256。未执行上游代码/测试，未取二进制或模型。
- Rollback、动作Token及校准论文读取相关正文；其余新候选只核摘要/版本史。校准理论的一步边际保证不能扩写为整条步态稳定保证，其实验为192物理转移/15episode池的受控重采样。原页显示8/20提交、编号为2610，保留冲突，不认定近期新稿。
- `2610.37330`在RSS和原始摘要页均为Tokens of Motion标题，但摘要描述AFIL/VLA失败引导。只记录来源不一致，不将摘要拼接成已核验动作primitive方法，也不据此声称论文造假。
- 昨夜HugWBC/HOMIE/ExBody/ExBody2/BeyondMimic、ConstrainedMimic、X-WBC/XHugWBC与SONIC文档的阅读沿用有日期收据，补充中央去重；不是本期新发现。未重复视频/保存审计。

## 版本、命令与证据

原始下载：`data/research/2026-10-29/sources/`。执行与结构化证据：`logs/research_watch/2026-10-29/`，含`context_check.json`、`fetch_results.json`、`source_comparison.json`、`static_text_verification.json`、`installed_rsl_static.json`。来源清单与关键SHA见[当天清单](../data/manifests/research_watch_check_20261029.json)。

```bash
python3 logs/research_watch/2026-10-29/run_check.py context
python3 logs/research_watch/2026-10-29/run_check.py compare
python3 logs/research_watch/2026-10-29/inspect_sources.py verify
```

检索助手首轮抽取本地Listing类时缺少urlparse导致NameError，补齐标准库命名空间后完成解析；原缓存未重下，科学代码未运行。这是本地检索工具错误，不是模型或服务器训练故障。两次静态文件名探测未命中记录保留，未据缺失路径推断整个工程缺失。

## 下一步与保存

完整研究历史/去重由main维护；Q1只接收今天文档和研究指针，保留其工程入口及中央历史状态。候选清单标明现在阅读/基线后再试/背景，无新H002，也不将MCTS、离散token、LoRA或一般融合登记为原创。Q1后续是否训练仍由用户审阅决定，本轮0新物理步、0策略推理、0优化、0依赖安装、0陌生代码执行、0大模型下载、0真机操作、0子agent。

按项目规范仅暂存本期文件、检查diff、运行`script/project_tools/audit_git_payload.py`后普通提交推送，独立核远端与固定内容；镜像采用复制前SHA比较，拒绝未知并行修改。具体完成状态和完整提交号以同步回执为准。
