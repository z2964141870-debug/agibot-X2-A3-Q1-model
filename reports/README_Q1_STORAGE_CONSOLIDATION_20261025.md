# Q1旧资产清理与ZHY统一目录

记录日期：2026-10-25，北京时间。阶段：Q1-STORAGE-C01/M01。状态：清理与统一目录迁移完成，环境与三份模型只读检查通过。

## 最终结果

唯一实体项目根已为`/opt/data/private/ZHY`。实际删除178938个过时文件或链接，文件块180.70GB；剩余归拢约299.58GB。10:52:47共享盘可用2141914271744字节，约2.142TB。未重新训练，旧失败/停止终态和当前Q1模型、原料、冻结输入保持。

| 原工作区 | 清理前GB | 清理后、移动前GB | 统一根内位置 |
| --- | ---: | ---: | --- |
| YuanQi_Q1 | 19.97 | 20.01 | 根目录四目录，审计文件增加约41MB |
| yuanqi_ops | 366.02 | 253.56 | data/logs/reports的yuanqi_ops命名空间，源码在script/yuanqi_ops |
| hechengzhu/ZHY | 94.38 | 26.00 | data/logs/reports的legacy_a3命名空间，源码在script/legacy_a3 |

GB均为十进制。旧根只是兼容软链接，不再占用另一份数据。source命名空间内部的data/logs/reports/envs也是指向四目录的链接。AMASS实际原料在`data/datasets/amass_bridge/amass_raw`；PHUMA、FGP原料和历史预览资源在`data/yuanqi_ops`；当前Q1基础环境在`data/legacy_a3/env`，入口在`data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python`。

## 目标与范围

用户授权淘汰旧A3的ZHY大文件、重复文件、无用途历史训练和已完整提取的压缩包；随后明确要求本项目全部文件统一到huanyu的`/opt/data/private/ZHY`，不再创建多个平级项目根。本次处理当前Q1、yuanqi_ops及hechengzhu/ZHY；其他用户/项目目录不属于本次范围。

沿用`data / logs / script / reports`。当前Q1成为统一根的主工作区；另外两处保留资源分别归入四目录内的`yuanqi_ops`、`legacy_a3`来源命名空间。旧根最终仅保留兼容软链接，避免既有绝对路径断开；新工作使用统一根。迁移采用同盘rename，不复制整套数据。

## 已完成核验与关键决定

- 10:12清理前共享盘可用1961206640640字节，约1.96TB。空间由外部恢复，不计入本轮清理释放量；此前09:42的0可用仅是历史读数。
- 当前Q1 Python虽然入口在yuanqi_ops/envs，实际基础环境是`hechengzhu/ZHY/data/env`，torch/numpy/mujoco/mink/smplx/scipy均从这里加载。约23.68GB基础环境必须保留，不能整体删除旧ZHY。
- 另外的`data/environments/isaac-runtime-raw`约22.90GB是独立旧环境副本，待依赖/进程核验后清理。
- 现有AMASS约180.26GB主要是原始NPZ，未找到AMASS遗留压缩包。PHUMA的data.zip不能因存在很小的extracted目录就删除；FGP包先完整SHA及解压成员比对。
- 当前Q1全部文件、516冻结输入、R02D退出权重、官方基线、SMPL-X、AMASS/PHUMA及有价值的小型配置/指标/日志保留。旧停止/失败实验终态不改写，不启动训练。
- 容器内可读进程已查未使用旧权重和raw环境；PID1/204/209部分proc字段权限不足，不能声称宿主机级完全排除。

## 版本、证据与执行

工作副本`/tmp/yuanqi-supermimic-q1-20261023`，分支Q1，阶段前HEAD为`3c0595e3d27079849459af33e6e14aa692544790`。此前盘点见[存储盘点](README_Q1_PRIVATE_STORAGE_INVENTORY_20261025.md)。本阶段逐项清理与迁移清单、命令、实际释放量、环境/冻结验证将在执行后补充。

R02D退出final.pt为205316947字节，SHA`8749a520441374b4c5818529090b9e33fa07ae70d6d304237bb5d3ef0a053d01`。旧路径严格身份检查包含绝对路径；迁移后需通过独立路径映射核验，不把路径变化冒充旧实验原位重跑通过，不改冻结收据或源码。

## 保存与下一步

本阶段记录、工具及小型清单保存到Q1分支，按标准流程审计、提交、推送并独立核远端，同次完整提交号由Git历史提供。大型保留资产仍为LOCAL_ONLY；删除是淘汰，不称备份。下面的执行进度是本轮历史过程，最终结论以上述结果及末节核验为准。

## 执行进度

删除入口`script/project_tools/cleanup_private_20261025.py`已部署，7项临时目录控制全部通过：精确身份、文件变化拒绝、越界拒绝、保护inode、软链接父目录拒绝、硬链接拒绝、当前进程打开文件识别。正在以流式SHA生成`data/storage/C01_20261025/deletion_plan.jsonl.gz`；计划生成完成前不删除。逐文件精确路径/大小/SHA/inode/nlink及理由保存在服务器压缩JSONL，Git只保存小型汇总及清单身份。

归拢入口`script/project_tools/consolidate_private_20261025.py`已部署，4项临时目录控制在Mac和服务器均通过：最长前缀映射、其他项目前缀不捕获、同盘rename保inode/旧路径可读、已有目标拒绝。本阶段使用新的存储审计目录，不修改516冻结文件。旧路径严格身份校验的路径部分不能在迁移后冒称原位通过；独立迁移核验将检查内容SHA及明确的新旧映射。

保留旧A3历史中M11D最终剂量参考、E55金区A0、E52/E53各一份机制参考，以及四个世界模型final；保留理由来自原README结论而不是仅凭文件名。其余关闭路线的大型中间权重待计划核验后淘汰。FGP的train_released.zip经目录表确认主要是acc/pose/joint等传感器训练数据，并非模型checkpoint，保留其唯一来源包。

首次C01计划在raw环境bin/infotocap的共享inode保护处退出，0删除。修订为仅接受所有链接名均在待淘汰环境内的完整硬链接组；若有环境外名字仍拒绝，并按inode只统计一次实际分配量。新增两项控制通过，删除控制共9项。C01B复用首次已完整SHA且inode/大小/mtime/ctime/nlink仍一致的记录，仅计算剩余文件；保留首次失败清单，不覆盖历史。执行前再次逐文件比对这些元数据，避免重复读取约180GB。

AMASS原数据此前实际位于ops/script/amass_bridge/data，归拢时会将这一数据目录移到`ZHY/data/datasets/amass_bridge`，原桥接代码位置只保留兼容链接，避免180GB原料继续混在script里。

C01B仍被共享inode保护拒绝，0删除；进一步lstat确认是16组各3个名字的硬链接软链接，均只位于raw及两个incomplete环境内，非基础环境常规文件。此前所有权枚举仅纳入常规文件，现将软链接inode纳入同一完整组审计，以O_PATH/NOFOLLOW持有链接本身，删除链接不会打开或删除其目标。新增对应控制通过，删除控制共10项。C01C继续复用C01B的稳定哈希前缀；不放宽环境外共享inode拒绝规则。

## C01C实际清理完成

原生apply退出0，COMPLETED，实际删除178938个文件/链接，逻辑字节180263029563，按inode去重的文件实际分配量180703694848字节，即180.70GB。各组文件分配量：ops训练104.72GB、ops旧SuperMimic6.92GB、旧ZHY训练13.64GB、旧ZHY实验31.74GB、废弃环境22.87GB、两个重复压缩包0.812GB。16组环境内部硬链接软链接完整核验后删除；当前Q1基础环境保留。

共享盘可用量从1961100525568增至2141903618048字节，同期增加180803092480字节；此df差值包括目录块、审计文件及其他任务活动，不能替代逐文件释放统计。此前约1.96TB的恢复来自外部，不算本轮清理。

最终计划SHA为`003d6a1feddf6faaebc091d8831eb877283b7f08002d3e7c8acc88c12cdd5aeb`，逐项删除日志SHA为`566bc6530d91d9ced38070b0c48d2c716d79073b28b8829ae79e63c80f5b6aa9`，服务器原件在`data/storage/C01C_20261025/`。保留8个精选A3权重合计约2.16GB、所有配置/小日志/指标/来源代码；历史终态不改写。下一核对实际缺失与保留项，再以M01新清单完成同盘归拢及516冻结/退出权重SHA核验。

清理后独立逐项lexists检查178938项全部不存在，8个保留权重大小/SHA全部保持，见[独立结果](../data/manifests/q1_storage_C01C_result_20261025.json)。M01迁移前清单已实际完成：280057个常规文件、3703个链接、41364个目录；516冻结和5项关键文件含退出权重均在移动前核验通过。三树剩余299575971840字节，即299.58GB；[迁移清单摘要](../data/manifests/q1_storage_M01_inventory_20261025.json)，逐项元数据gzip在服务器data/storage/M01_20261025，SHA`d1c751f41c47a80c4933e9a9ffe62f044ec89c5ca6ba94f1d5b79c41fff17994`。

## M01实际归拢与核验

apply原生退出0，PASSED，0复制操作。同盘rename后逐项检查280057常规文件的inode/大小/mtime、3703个链接和41364个目录全部通过；0失败、0活动日志变化。修复138处原相对链接，使目标保持同一文件。516冻结及spec/退出final/三个停止状态共5项关键文件大小/SHA全部保持；退出final仍为原SHA，未改写旧收据。[归拢结果](../data/manifests/q1_storage_M01_result_20261025.json)、[逐项移动与链接修复](../data/manifests/q1_storage_M01_moves_20261025.jsonl)。这是文档更新之前的迁移时点检查，之后正常更新README/AGENTS不冒充旧文件mtime未变。

[新入口依赖导入](../data/manifests/q1_storage_M01_runtime_20261025.json)原生退出0，torch/numpy/mujoco/mink/smplx/scipy均从统一根内成功加载，CPU张量求和为3。pyvenv.cfg的base_prefix字符串仍指向旧兼容路径，real_executable已在新根，故当前不删兼容链接、不改venv冻结来源。[三份Q1模型检查](../data/manifests/q1_storage_M01_models_20261025.json)加载及mj_forward全部通过，nq/nv/nu/body/geom维度匹配，qpos/qacc有限，0积分步/0新训练。首个只读模型wrapper误用ngeoms属性失败，修正为MuJoCo的ngeom后完成实测；不是模型或迁移故障。

本轮工具控制共10项删除+4项归拢全部通过。[执行记录](../logs/session_records/q1_storage_C01_controls_20261025.log)。这些是一次性运维入口，不在已完成目录重复apply。已实际执行的历史命令如下，输出现均通过旧根兼容链接映射到统一根：

~~~bash
python3 /opt/data/private/YuanQi_Q1/script/project_tools/cleanup_private_20261025.py plan --output /opt/data/private/YuanQi_Q1/data/storage/C01C_20261025 --reuse-plan /opt/data/private/YuanQi_Q1/data/storage/C01B_20261025/deletion_plan.jsonl.gz
python3 /opt/data/private/YuanQi_Q1/script/project_tools/cleanup_private_20261025.py apply --output /opt/data/private/YuanQi_Q1/data/storage/C01C_20261025
python3 /opt/data/private/YuanQi_Q1/script/project_tools/consolidate_private_20261025.py inventory --output /opt/data/private/YuanQi_Q1/data/storage/M01_20261025
python3 /opt/data/private/YuanQi_Q1/script/project_tools/consolidate_private_20261025.py apply --output /opt/data/private/YuanQi_Q1/data/storage/M01_20261025
~~~

代码版本为本阶段Q1提交中的四个运维/控制文件；精确内容及发布号由Git历史恢复。原大规模逐文件计划/删除日志/迁移前清单保留在`ZHY/data/storage/`，Git只保存本节引用的小摘要与移动日志；没有在Mac复制模型或原始数据。未读取或保存连接凭据。

## 下一步与边界

所有新工作从`/opt/data/private/ZHY`开始，不再开平级根。原R02D仍为STOPPED_STORAGE/STOPPED_SIGNAL，盘余恢复不自动重启；下一科学执行需显式注册新根的路径绑定，再做P15正式八组和退出权重的独立动力学配对，不能绕过旧绝对路径严格身份检查。原训练源码/收据不为迁移修改，516冻结内容保持。BONES仍由师兄处理；不恢复获取。

AMASS/PHUMA/SMPL-X、官方基线、FGP唯一来源包和历史humanplus预览保留。未发现AMASS残留压缩包；两个重复FGP/CLOTHO zip已在SHA完全相同后删除冗余份。当前Q1旧权重还承担输入/基线/诊断链，未判定无用，不做盲删。容器proc权限边界仍存在；本次不处理其他项目目录，不声称对共享宿主机的全部写入者完成归因。
