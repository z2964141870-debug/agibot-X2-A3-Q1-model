# ZHY文件页缓存维护与失败留证

记录：2026-10-14。状态：维护诊断改进及单次定向释放完成。此前内存分类见README_ZHY_MEMORY_20261013.md。

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

## 00:23单次维护结果

实现提交`b09a9619ece8ae2d4e989f6f9cf3e44afa50de9a`发布/同步后，3标准库控制在ZHY通过。空闲实测再次通过原进程检查，按E49计划保护待用输入；在24个合格目录工件中仅2文件有可释放驻留页，共释放129.77MiB，文件大小/mtime/ctime/身份均不变，0删除。原可用估计8.324→8.450GiB，计账口径不变、同boot/OOM0。

释放对象为已结束实验中不再作为下一步输入的工件，具体路径与页驻留前后见`data/manifests/zhy_file_cache_20261014.json`。仍有约7.22GiB文件cache，不能称全部清空；其余不是本次可释放闲置候选，未触碰环境/待用输入或宿主缓存。维护后已达到启动余量，继续主线，不为追求“cache归零”额外扫描或反复清理。

自动巡检保持30分钟，若缓存积累使可用估计低于8GiB，会自主执行该有保护的维护并重新核验三次。没有合格对象或仍不足时保存WAIT，不把保护线降下去。

## E49停止后维护入口记录（01:00附近）

E49后调用缓存维护被原空闲门拒绝：`Other compute/large process; defer cache maintenance`，在require_idle首检查处停止，0目标释放/0删除。此处执行包装器同步启动了缓存维护子进程，父进程本身是Python；现有门明确拒绝任何其他Python，包括该标准库父控制器，因此这次调用方式自身不满足门，不能当作训练占用或系统内存故障。

随后独立只读快照无其他大进程或Python计算，OOM0，可用估计8938430464 bytes（8.325GiB）仍过8GiB线，file cache7896920064 bytes、RSS369504256 bytes。没有重试清理、没有终止进程/改阈值。以后维护从shell直接启动模块或用exec替换标准库包装进程，不能放宽闲置保护。保护计划已更新为E50候选保留接续，可信基准E49 registration；完整小证据见`data/manifests/a3_candidate_retention_followup_readiness_20261014.json`。

## 09:57用户再次要求内存不足时直接清理

用户重申已有授权：“当内存不够的时候，该清理就清理”。内存不足且满足原维护条件时，直接在既有范围回收闲置项目文件页，无需再次询问。这项授权本轮用于检查/缓存管理，未扩展为修改内核保护、重建基准、改变E51计划或恢复已暂停自动任务。

先前SSH复用socket已不存在，首检查认证失败记为UNKNOWN；用已有授权恢复连接后完成只读检查，不据认证失败认定重启。北京时间2026-10-14 09:57:34，16GiB容器按原口径可用9.966GiB，高于8GiB；文件内容缓存5.415GiB、内核计账0.619GiB。GPU45%/3879MiB，另有Python约4.924GiB RSS，当前不满足原空闲维护门。

因此本次无需为达到8GiB清理缓存，也没有尝试在其他任务计算期间运行维护。维护执行0次、释放0字节、删除0文件；没有全局drop_caches、终止进程或调整计账。保持同boot和本容器OOM0，宿主内核相对E50的FATAL记录仍在，未过滤或重新基准化。E51 OUT仍不存在，本工程heartbeat保持原暂停状态。

当前回收入口仍是`script.project_tools.zhy_file_cache`，待用输入保护计划为`data/manifests/a3_finite_retention_plan_20261014.json`，可信基准为E50 registration；原32文件/2GiB/1800秒、idle/boot/OOM/kernel等条件保持。以后有内存不足且全部原维护条件满足时直接执行；若被非内存门拦截，准确记录原因，不以“清过缓存”冒充门已解除。

小清单`data/manifests/zhy_file_cache_user_check_20261014.json`；原始快照`logs/zhy_file_cache/20261014T015734Z_user_cleanup_check.json`仅LOCAL_ONLY，保存到ZHY并核对SHA。报告/索引/current state与小清单按scoped审计后同步Git；以实际发布回执为准。没有代码、实验或自动化状态变更，未新增模型测试。
