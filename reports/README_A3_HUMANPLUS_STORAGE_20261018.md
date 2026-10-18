# A3 humanplus 淘汰权重存储管理

记录日期：2026-10-18，北京时间18:10。S01清理已完成计划，尚未删除；执行及结果追加下文。用户明确允许删除过大或确认完全无用的训练权重，覆盖此前对这类淘汰产物必须先网盘备份的要求。关键输入/官方模型的保护与备份规范仍有效。

## 目标与范围

只处理humanplus:/home/humanplus/YUANQI_A3中HP12/HP13永久关闭且禁止续跑/早期效果的权重，以及HP14/HP18完成十项匹配评测后已拒绝的候选。所有来源/失败/指标/配置/日志/JSON sidecar保留，原终态不改。新存储清单成为权重实际存在状态，历史审计只证明当时测量，不再据此运行旧checked/验证/回放。

## 实施与证据

- 新代码script/project_tools/prune_humanplus_rejected_weights.py，精确白名单32路径；不使用通配删除，不删除目录/缓存/根外项目，不加载张量。
- 17常规路径含2个final8评测bundle硬链接，15个唯一权重inode；另15个last.pt符号链接。检查每文件大小SHA与sidecar一致、全部硬链接已列出，无未知链接；逻辑唯一3187189129字节，实际已分配3187286016字节，约2.97GiB。
- 逐项确认HP12/13/14 STOPPED，HP17/18 COMPLETE_MATCHED_EFFECT且独立审计verified=true/gate=false；活动或待执行计划若依赖选中文件则拒绝。
- 相关同UID非登录任务FD/maps元数据核对，不读取进程内存；未发现选中文件持有。只根据进程名称及systemd父进程关系排除sd-pam、ssh-agent和sshd登录/凭据服务，不读取其认证状态；其他UID进程不检查内存。最初预检因上述受限登录服务停止且无删除，sudo -n只读权限检查不可用；未索取认证、未扩大sudo/内核权限，未禁用系统保护。最终精确清单计划已生成。
- 7标准库文件系统控制Mac/新机通过，覆盖完整硬链接、未列硬链接拒绝、大小SHA改写拒绝、符号链接逃逸拒绝、活动输入保护和白名单/登录helper边界；不冒充实际删除验证。
- 计划data/manifests/a3_humanplus_weight_prune_20261018_S01_plan.json，含逐文件SHA/大小/inode/硬链接/eligible终态SHA/进程检查范围。

## 版本与命令

阶段前科学终态提交5acbe4c。清理实施与计划审diff/暂存、audit_git_payload.py后scoped提交；实际SHA以Git为准。`python3 -B -m unittest script.project_tools.test_prune_humanplus_rejected_weights`。`/usr/bin/python3 -B -m script.project_tools.prune_humanplus_rejected_weights plan`已完成、禁止覆盖；下一单次execute前再次核验终态SHA/文件/inode/FD/maps/活动输入。execute逐路径unlink+目录fsync并持续保存结果，不全局drop_caches或sync，不停止他人任务。

## 结果边界与保存

计划阶段0删除/张量加载/新策略forward/仿真/optimizer。预计释放按唯一inode st_blocks计算；文件系统可用空间前后差另记，可能含并行文件系统活动，不虚报等同本次释放。官方035及E11/E12 teacher必要输入、已迁原始数据、全部配置/指标/日志/小清单保留。未备份的淘汰权重按用户明确授权删除属于有意丢弃，不称网盘备份；大型必要工件仍LOCAL_ONLY。

GitHub LOCAL_COMMITTED_UNPUSHED，不重复同一publickey失败push。后续每轮终态先判断采用/继续用途，明确淘汰才登记清理；不能只因文件大就删除仍有用途模型。

## 下一步

提交本计划后单次execute；核对删除路径不存在、035SHA不变、实际空间变化和记录留存，立即追加README/索引/current state并提交。之后继续长期优化，只从保存失败证据/既有研究去重选择实质不同方向；HP12/13/14/18权重删除后不复验/恢复/重放。
