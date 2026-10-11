# hp3090 逐级负载与普通训练排查

记录日期：2026-10-11。实验 HOST_LOADCHECK_20261011_E01。状态：已停止，四worker CPU档触发温度保护；后续分支搁置。

## 目标与范围

用户在两小时空闲观察完成后明确授权逐级CPU、内存、GPU与普通神经网络测试。目标是缩小重启触发范围，不做A3训练、强化学习、仿真或真机操作；A3旧服务和巡检保持暂停。本次授权替代此前仅限空闲观察的计算限制，但不授权改驱动、功率限制或BIOS。

## 已有证据

空闲窗口北京时间2026-10-11 10:44:35至12:44:35，122采样、0boot转换，CPU37-49摄氏度，GPU30-31摄氏度。此前重启跨GPU训练、CPU标签准备和GPU推理，尚未确定共同操作。见 `README_A3_IDLE_WATCH_20261011.md`。

## 决定与执行安排

每档先30秒，再120秒，最后600秒；档间300秒冷却。第一分支固定stress-ng int64计算方法，单worker25%、50%、100%，再2、4worker各100%。负载百分比针对每个worker，不是整机。单独内存分支使用一个worker、256MiB与1024MiB、write64验证；不能把它称为整机内存完整检测。后续GPU、小网络与YOLO按前序结果继续，发生保护/计算错误或确认重启则停止升档、保存故障，不自动重复。

保护保持CPU90/GPU85摄氏度连续两次、5秒采样，增加至少4GiB可用内存保留。温度或GPU遥测不可读、关键内核故障、超时均停止。程序及子进程由临时systemd用户unit运行，Restart=no，不enabled，不开机续跑。测试工具从Ubuntu仓库取得，独立解包于data，不安装到系统、不改Python环境。

## 版本与可复现入口

服务器 `/media/yu/FAFF-E9771/YUANQI_A3`，分支main；代码 `script/diagnostics/host_loadcheck.py`、保护契约 `script/diagnostics/test_host_loadcheck.py`。复用标准库 `script/a3/host_health.py` 和原子保存函数，但不调用旧训练控制器。

工具stress-ng 0.13.12-2ubuntu1，Ubuntu jammy-updates/universe；包2106882字节，SHA256 `72578e3087c7a678555d2076ae7d4e1274b597e763dfc9d957b8a178600b43f2`。GPL-2.0及各源码许可见解包的 `usr/share/doc/stress-ng/copyright`；工具和依赖包只保留data、不提交第三方二进制。

```bash
cd /media/yu/FAFF-E9771/YUANQI_A3
/usr/bin/python3 -m unittest script.diagnostics.test_host_loadcheck
/usr/bin/python3 -m script.diagnostics.host_loadcheck --initialize
# 注册仅执行一次；已存在job不可覆盖，执行器只接受READY。
/usr/bin/python3 -m script.diagnostics.host_loadcheck
/usr/bin/python3 -m script.diagnostics.summarize_host_loadcheck
systemctl --user show yuanqi-host-loadcheck-E01.service -p ActiveState -p Result
# 如需停止当前独立诊断：
systemctl --user stop yuanqi-host-loadcheck-E01.service
```

## 产物与日志

- 固定参数：`data/experiments/host_loadcheck_20261011_E01/job.json`。
- 状态与逐档结果：同目录 `state.json`。
- 只读汇总入口：`script/diagnostics/summarize_host_loadcheck.py`；生成 `data/manifests/host_loadcheck_result_20261011_E01.json`。这份清单是带时间戳的快照，运行中须重新汇总，不把旧RUNNING快照当作当前状态。
- 每5秒监测及单档stdout：`logs/host_loadcheck_20261011_E01/`。
- 工具/依赖：`data/tools/host_loadcheck_20261011/`，LOCAL_ONLY。
- 独立测试不加载A3模型，不改旧R01-R09/E10/E20/E21台账。

## 结果与边界

2026-10-11北京时间13:11:40开始单worker25%首档，4保护契约通过。临时unit `yuanqi-host-loadcheck-E01.service`，CPUQuota410%、MemoryMax6G、RuntimeMaxSec3h、Restart=no；该CPUQuota为四worker档与监测的执行上限，不是单worker实际占用目标。首批实际CPU约40摄氏度、GPU30摄氏度，GPU利用率0、内核故障列表空。完整结果尚未生成，不能把启动成功记为稳定性通过。

早期快照（82条采样）：首两项30秒及120秒已实际完成，stress-ng返回0并报告successful run completed，stdout SHA独立复核一致；600秒当时仍运行。同boot、CPU采样最高49摄氏度、GPU空闲，没有关键内核事件。预计CPU/内存完整序列约118分钟，不含准备时间；最终结果以运行完成后的state/日志为准。GPU、小网络和YOLO未启动，等待本分支结果，不并行制造第二负载源。

13:38阶段核验：单worker25%三项30/120/600秒全部通过；50%档30/120秒通过、600秒运行中。327条采样，同boot，CPU采样最高57摄氏度、GPU30摄氏度且利用率0、无关键内核错误。以上是逐档有限窗口结果，不能排除其他指令、更多线程或GPU负载触发的问题。

新boot是确认重启的证据，SSH失败只记UNKNOWN。温度采样不能排除间隔内峰值，各档未重启只支持有限窗口结论。

### 单worker CPU分支完成

2026-10-11北京时间13:11:40.046369至13:59:28.815452，固定int64方法的25%、50%、100%三档，各30/120/600秒，共9次实际测试全部PASSED：均返回0、successful run completed，9份stdout SHA复核一致。14:00:13汇总包含598条记录、同boot、0解析错误、0关键内核事件；该分支CPU采样最高58摄氏度，GPU保持空闲。

三个600秒窗口整机逻辑CPU平均占用约1.65%、2.36%、5.14%；单worker100%不是整机100%。这是一个worker的指定整数计算验证，没有逐物理核固定亲和性扫描，也未覆盖AVX、浮点/BLAS、全核心并发、GPU或普通网络训练，不能排除这些不同路径。现有控制器冷却300秒后按原job进入2/4worker与256/1024MiB内存分支，未重置/重启测试，A3继续暂停。

### 双worker CPU档完成

2026-10-11北京时间14:04:28.849319至14:17:05.134285，双worker各100%固定int64的30/120/600秒三次测试全部PASSED，均返回0，三份stdout SHA独立复核一致。14:21:11汇总854条记录，累计12项通过；boot仍为cb282100-387e-400f-bff0-dab2de7282b8，0日志解析错误、0关键内核事件、无保护触发。此档CPU采样最高72摄氏度，GPU最高29摄氏度、利用率0；可用RAM最低29.54GiB。

600秒窗口整机逻辑CPU平均占用10.09%。结果仅覆盖两个worker的指定整数计算，不代表全核心、浮点/AVX、GPU或普通训练稳定性。核验时原unit处于300秒冷却，按原job继续四worker和内存档；未重启或重置实验，A3训练/评测继续暂停。快照为 `data/manifests/host_loadcheck_result_20261011_E01.json`，巡检证据为 `logs/host_loadcheck_20261011_E01/watch_20261011T062020Z.json`，原始日志LOCAL_ONLY。

### 最终停止与测试矩阵

北京时间2026-10-11 14:25:11.881939、14:25:16.925666，两次相隔约5秒的CPU采样均为90摄氏度；GPU28摄氏度、利用率0。14:25:16.963990控制器保存STOPPED/TEMPERATURE_PROTECTION。四worker600秒档从14:24:36.584924运行40.377秒后终止，未完成600秒；此前同档30/120秒连续执行且无档内冷却，最高温度分别73/87摄氏度。失败档整机逻辑CPU平均20.34%，不能将“四worker各100%”表述为整机满载。

| 测试 | 状态 | 证据与边界 |
| --- | --- | --- |
| 单worker 25/50/100%，各30/120/600秒 | 通过，9项 | 固定int64，最高58摄氏度 |
| 双worker各100%，30/120/600秒 | 通过，3项 | 最高72摄氏度 |
| 四worker各100%，30/120秒 | 通过，2项 | 最高73/87摄氏度 |
| 四worker各100%，600秒 | 失败，温度保护 | 40.377秒停止；连续两次90摄氏度 |
| 256/1024MiB内存，各30/120/600秒 | 搁置，6项未运行 | 首保护停止；无内存验证结论 |
| GPU矩阵、小CNN、YOLO | 搁置，未启动 | 前序保护触发；YOLO工具和数据尚未核验 |

本实验最终909条采样、14项通过、1项保护停止、6项未运行。全部采样同boot cb282100-387e-400f-bff0-dab2de7282b8，0日志解析错误、0关键内核事件、15份已执行stdout SHA复核一致；执行器SHA与注册job一致。stress-ng失败档返回2及子进程killed by signal是控制器SIGTERM后的退出记录，没有据此认定计算错误。unit退出1、failed、Restart=no；核验时无stress-ng或GPU计算进程，无重启复现。

14:27:14停载后只读采样CPU55摄氏度、GPU28摄氏度；并非重新开启监测或负载。原A3服务inactive/disabled；临时hp3090 heartbeat已PAUSED且回读核验，yuanqi-a3仍PAUSED，每日研究yuanqi仍ACTIVE。不启动后续分支，不重复或恢复E01，不放宽保护、不改驱动/功率/BIOS。

结果证明当前四worker整数计算序列会触发既定CPU温度保护，并未证明历史重启由温度、CPU、电源或某一硬件导致。温度升高是后续排查线索，具体原因未定位；用户后续可先检查CPU散热器固定、风扇/水泵运行与积尘，记录检查或处理结果，再决定独立新实验。CPU功耗、频率及逐核热情况不在本次记录中，未验证这些解释；内存、GPU和普通网络能力仍UNKNOWN。

最终小型证据/SHA清单：`data/manifests/host_loadcheck_stop_20261011_E01.json`；逐项汇总：`data/manifests/host_loadcheck_result_20261011_E01.json`。触发巡检与停载采样：`logs/host_loadcheck_20261011_E01/watch_20261011T062520Z.json`、`logs/host_loadcheck_20261011_E01/after_protection_20261011T062520Z.json`。原始日志和工具留服务器LOCAL_ONLY，未宣称网盘备份完成。

## 同步与恢复

阶段代码、说明及小型结果清单审计后scoped提交main并核验远端。原始日志留服务器、LOCAL_ONLY。若重启，先核验boot并登记最后RUNNING档为中断；不自动重启测试。每日研究yuanqi保持不变。

为跟进后台测试，新建临时heartbeat `hp3090`（名称“hp3090逐级负载诊断”），每5分钟检查并继续已授权的后续GPU/普通小网络分支；新故障或重要分支完成才通知，状态不变保持安静。遇首故障或全部完成后记录并暂停。原 `yuanqi-a3` 仍PAUSED、每日研究 `yuanqi` 不变，不重新启用A3训练/评测。此监测独立于已经结束的固定两小时窗口，不重置或延长原观察。
