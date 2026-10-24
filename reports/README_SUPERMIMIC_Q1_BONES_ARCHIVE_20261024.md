# BONES原包流式接入准备与控制验证

记录日期：2026-10-24，北京时间。阶段：SM-Q1-BONES-P02-A01。状态：接入实现及两端11项控制完成；完整真实原包仍未到位，尚未执行真实接入。

## 目标与范围

在[BONES主线](README_SUPERMIMIC_Q1_BONES_GOAL_20261024.md)下，为Windows到huanyu直接迁移的完整G1 tar.gz建立可核验入口。只提取[P01冻结候选](README_SUPERMIMIC_Q1_BONES_METADATA_20261024.md)，避免全量解压占盘；保留原包及旧数据/实验终态。CSV格式与原速接口见[F01](README_SUPERMIMIC_Q1_BONES_FORMAT_20261024.md)。

## 完成内容与决定

新增`script/supermimic_q1_sources/intake_bones_archive.py`及对应控制。入口要求来源树内的完整常规`.tar.gz`、明确总字节数和完整SHA-256，拒绝未完成下载、已有输出目录、变化的冻结候选。原包全文件哈希后流式扫描全部成员，读过gzip尾部核完整性和CRC；只提取冻结train1024/dev128/test128个CSV，逐文件计大小与SHA。记录完整成员索引、metadata覆盖及多余CSV数，不调用extractall。

拒绝路径越界、链接、特殊/稀疏文件、重复规范路径及成员/总大小超限。归档根差异由显式`--member-prefix`表达，不猜测或改名掩盖来源。单选中成员64MiB、选中总逻辑8GiB、归档声明展开250GiB、成员50万上限；复用已登记100GiB空闲、12GiB实验、250GiB来源预算，进度时核OOM计数与磁盘。真实长任务仍需经P00外层worker保护启动。成功仅代表传输、容器完整性和名单提取，不代表CSV或Q1动作合格。

## 实际验证与版本

Mac与服务器同版各11项控制通过，分别0.021和0.037秒、原生退出0。包含完整run()合成往返、原包保持、错误SHA、冻结名单改变、按名单提取/hash、显式目录前缀、路径/链接/重复成员、缺项和预算、gzip截断尾部及CRC破坏、拒绝复用输出。合成内容不是真实动作，新增Q1参考0、优化器更新0。

两部署代码SHA逐项一致，精确值及服务器原生控制输出见[收据](../data/manifests/supermimic_q1_bones_archive_controls_20261024.json)。服务器日志`/opt/data/private/YuanQi_Q1/logs/BONES_P02_archive_controls_20261024.log`。阶段前已发布提交`0661a55aa31602e950506721edadf98bd462a25b`；本阶段代码和记录由后续同次提交保存。

```bash
/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python -m unittest discover \
  -s script/supermimic_q1_sources -p test_intake_bones_archive.py -v
```

真实原包命令必须先取得完整大小/SHA并实核OOM基线，以`--archive`、`--source-root`、`--expected-bytes`、`--expected-sha256`、`--metadata-summary`、`--goal-config`、`--output-dir`显式登记；本阶段未执行此命令，不能用控制通过代替接入收据。

## 源模型候选与限制

实际MuJoCo编译并FK了现有GEAR SONIC `g1_29dof_rev_1_0.xml`：nq36/nv35/nu29，29关节顺序与CSV解析器一致，默认骨盆高度0.793m，SHA及绝对路径在收据。仅证明一个可载入且名称匹配的候选模型；尚未证明它与BONES生成动作使用的模型几何/零位相同。重定向前须补来源和几何核对，不能据29DOF相同直接认定匹配。

## 下载、资源与结果边界

用户确认完整动作包仍在Windows `D:\tmp`下载。21:16读目录未见完整G1/SOMA包，最大crdownload为2720542770字节，与20:51观察相同；未读取浏览器下载状态，不能仅据大小判定暂停或失败，未将partial当训练输入。

21:22:06实核huanyu仍无BONES tar.gz；RTX3090为38摄氏度、17788MiB空闲、利用率0%，容器内存27013308416字节、OOM14，磁盘空闲1965293809664字节。只读资源探针首次使用错误cgroup根路径退出1，随后复用项目已验证health入口退出0；没有因此启动/停止训练。GPU列表中的namespace外PID不能当成本项目活跃进程证明。

本阶段无真实CSV接入、Q1转换或训练；主目标仍未达成。Mac只增小代码/记录，原包及大型产物后续留服务器，网盘LOCAL_ONLY。

## 保存与下一步

本README、收据、当前状态及索引同次保存，按Q1分支审计、提交、推送并独立比对远端。先同步小文件，再继续源模型核验及独立BONES重定向/质检准备。原包完整后直接Windows到服务器，核稳定大小/完整SHA、流式验收、真实CSV协议与帧数，再形成合格Q1参考；不得将metadata候选时长算作合格训练时长。
