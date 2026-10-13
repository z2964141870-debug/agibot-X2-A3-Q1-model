# ZHY文件页缓存维护与失败留证

记录：2026-10-14。状态：维护诊断改进完成，定向释放待单次核验。此前内存分类见README_ZHY_MEMORY_20261013.md。

## 目标与用户授权

用户2026-10-13 23:49明确要求：文件内容缓存积累导致可用估计低于8GiB启动线时，代理自行清理。沿既有只释放闲置页缓存的方法；不删除数据/模型/日志，不改内存口径或阈值，不触碰全宿主drop_caches或其他用户进程。

## 实测与修改

E48在2026-10-14 00:02正常退出，最低运行余量4.513GiB>4GiB。随后一次缓存维护在require_idle阶段以Unresolved process visibility拒绝，未生成释放intent、0释放/删除。原报错没带PID，所以无法确认当时对象；不把失败说成清理成功。

只读复核当前同boot/OOM0，176僵尸无RSS、3原小基础休眠进程，无RSS>1GiB进程，GPU26°C/0%/13MiB，可用估计8.225GiB。已为维护脚本补充PID/comm/状态/RSS的拒绝证据与blocked收据；未知进程或大进程仍拒绝，门槛未放宽。3项标准库控制通过：僵尸无RSS、小型已知服务、未知/偏大/缺RSS仍拒绝。仅处理退出竞态，不将不可读活进程当空闲。

## 执行与保存

当前脚本`script/project_tools/zhy_file_cache.py`；仅ZHY根目录使用系统Python：
`/usr/bin/python3 -m script.project_tools.zhy_file_cache --trim --protect-plan data/manifests/a3_candidate_retention_plan_20261014.json --baseline data/experiments/a3_zhy_fresh_batch_20261013_E48/registration.json`。

cache>4GiB才检查，每次最多32文件/2GiB，成功释放间隔>=1800秒，只选择已终结实验工件，保护下一E49计划、非终结登记输入、打开/映射文件。无合格文件或COOLDOWN正常保留，不清环境树、不强行腾空所有cache；足够资源就继续主线。以后低于启动线且确认为缓存时先尝试该维护，再三次复验；维护本身不授权启动其他训练。

原日志`logs/zhy_file_cache/` LOCAL_ONLY，小收据另存清单。失败先留证/只读排查，不盲复跑；首机器异常仍停工程。E48源码/结果保持不改，不借本维护重复实验。
