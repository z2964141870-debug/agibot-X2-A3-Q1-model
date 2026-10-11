# hp3090 逐级负载与普通训练排查

记录日期：2026-10-11。实验 HOST_LOADCHECK_20261011_E01。状态：运行中，CPU/内存分支。

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
```

## 产物与日志

- 固定参数：`data/experiments/host_loadcheck_20261011_E01/job.json`。
- 状态与逐档结果：同目录 `state.json`。
- 每5秒监测及单档stdout：`logs/host_loadcheck_20261011_E01/`。
- 工具/依赖：`data/tools/host_loadcheck_20261011/`，LOCAL_ONLY。
- 独立测试不加载A3模型，不改旧R01-R09/E10/E20/E21台账。

## 结果与边界

2026-10-11北京时间13:11:40开始单worker25%首档，4保护契约通过。临时unit `yuanqi-host-loadcheck-E01.service`，CPUQuota410%、MemoryMax6G、RuntimeMaxSec3h、Restart=no；该CPUQuota为四worker档与监测的执行上限，不是单worker实际占用目标。首批实际CPU约40摄氏度、GPU30摄氏度，GPU利用率0、内核故障列表空。完整结果尚未生成，不能把启动成功记为稳定性通过。

新boot是确认重启的证据，SSH失败只记UNKNOWN。温度采样不能排除间隔内峰值，各档未重启只支持有限窗口结论。

## 同步与恢复

阶段代码、说明及小型结果清单审计后scoped提交main并核验远端。原始日志留服务器、LOCAL_ONLY。若重启，先核验boot并登记最后RUNNING档为中断；不自动重启测试。每日研究yuanqi保持不变。
