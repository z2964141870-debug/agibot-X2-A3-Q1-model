# A3 humanplus 淘汰权重存储管理

记录日期：2026-10-18。S01于18:11完成删除，18:12独立核对通过，释放3187286016字节；详细计划与结果见下文。用户明确允许删除过大或确认完全无用的训练权重，覆盖此前对这类淘汰产物必须先网盘备份的要求。关键输入/官方模型的保护与备份规范仍有效。

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

## 18:12 删除及独立核对完成

S01 execute于2026-10-18T10:11:25.820289Z至10:11:26.305331Z单次完成，COMPLETE_REJECTED_WEIGHTS_DELETED。17常规路径（15唯一权重inode）及15last符号链接共32路径均消失，释放实际分配3187286016字节（约3.19GB/2.97GiB）；文件系统可用空间从239912800256增至243100086272字节，增加与本次分配释放相同。各训练目录现仅184/236KiB配置与sidecar，未删除目录或数据。保留17个checkpoint sidecar；035402050931字节及E11/E12 teacher两PT均重新大小SHA通过，035 SHA9cf33be2...70f2d未变。0新张量反序列化/策略forward/仿真/optimizer/旧科学审计复跑。

计划SHA6d92bf77aef290bded4e53bd8cb46c9162a8d09691fb086fae2046f334a26653；结果SHA b3a88738925e036c0ba72eb30754b96887c0e4003c9674a5c655d97306672333。data/manifests/a3_humanplus_weight_prune_20261018_S01_{plan,result,audit}.json保存逐路径/大小SHA/硬链接/删除进度/独立核验。最终进程检查明确排除系统sd-pam/ssh-agent/sshd登录凭据服务，其他可见相关同UID任务FD/maps无选中文件；未检查其他UID进程内存，不宣称全系统句柄扫描。数据/配置/原始日志/指标/JSON均保留，未改历史终态；历史artifact SHA不代表被删权重现在仍存在。

用户存储授权已写AGENTS.md和README_STORAGE.md。HP12/HP13/HP14/HP18及HP16 bundle选中权重已永久弃用，不再运行旧checked链/复验/恢复/重放或为缺失文件重建。新科学计划仅纳入实际必要且存在的输入，不能继承历史审计全部artifact作为当前依赖。GitHub仍LOCAL_COMMITTED_UNPUSHED；删除未备份弃用权重不称备份，必要模型和数据仍LOCAL_ONLY。下一持续主线先保存数组/指标与既有研究去重选择实质不同方法，不重复已拒绝D01/D02配方。

## S02：HP19淘汰权重计划

HP19/D03已完成十项匹配效果且独立审计verified=true/gate=false，候选弃用，原终态提交b302e10。按同一用户存储授权，仅选择HP19/R25保存0/2/4/8四权重及最终8 bundle硬链接、四last符号链接，共9路径/4唯一inode。逐路径大小SHA及全部硬链接已核对，计划释放812695552分配字节（逻辑812666268字节）；不是网盘备份。

新独立prune_humanplus_route_weights复用S01文件系统保护语义，但S01源码/终态不改、不复跑。2项新增范围/完整拒绝门控制Mac和新机各通过；接受候选、未验或STOPPED均拒绝删除。计划固定035/E11/E12三必要输入和HP19的128非选中证据文件大小SHA，保留配置、全部原始轨迹/日志指标、sidecar及独立审计。148相关同UID非登录任务FD/maps无选中文件，系统登录凭据服务/其他UID内存不检查，未扩权。先scoped保存代码与S02_plan，再单次execute和audit；不加载任何张量或复跑科学审计。

## S02：18:51 删除与独立核对完成

2026-10-18T10:51:42.440056Z至10:51:42.582457Z单次execute完成，独立audit于10:51:42.964575Z verified=true。HP19的5常规路径/4唯一inode及4last符号链接共9路径已删除，释放812695552分配字节（约0.813GB/0.757GiB），唯一逻辑字节812666268。文件系统可用242000130048至242812821504字节，变化812691456字节，含潜在并行文件系统活动，不冒称严格等于本次释放。

035/E11/E12三必要输入及128非选中HP19证据文件大小SHA均未变，配置/日志/完整轨迹数组/指标/sidecar和旧终态保留。0张量加载或旧科学审计复跑；不再执行S02或加载、恢复、重建、重放HP19权重。历史final8 SHA f92ace494e64cdff842db7afd742b8310c1aa48dd77413cd32e1eb414ae77ae9仅代表删除前测量，不是可用模型。

计划SHA84624598d4fd4bcbfb1aa6ddd087d47b9f8b31e9905a9397287b588d6c3c87cf；结果SHA a29697988276d4ca2c5e009bc137fcddbdaa1e4e48e23960234d9e5d81803bca。data/manifests/a3_humanplus_weight_prune_20261018_S02_{plan,result,audit}.json为恢复入口。S01+S02累计释放4009981568分配字节，约4.01GB/3.73GiB，是有意丢弃淘汰产物，不称备份。GitHub仍LOCAL_COMMITTED_UNPUSHED。

下一遵循用户最新专项专家到统一学生方向：先核对已有自适应采样/目标/双模拟器接口，独立固定zzx整段专家合同，不重复D01/D02/D03配方，不改变已关闭实验或原统一策略验收门。
