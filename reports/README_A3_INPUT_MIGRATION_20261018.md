# A3已验输入迁移到humanplus

10/18北京时间10:20最新：用户明确“直接迁移，慢一点也没关系”，覆盖本次旧手动交接等待与120秒慢传截止。HP04直接迁移COMPLETE_VERIFIED，11文件613,119,744字节单次112.032秒完成；035 PT已到位。连同HP03共63文件628,994,284字节独立大小/SHA通过，关键训练输入10/10。当前INPUTS_VERIFIED_WAIT_RUNTIME_VALIDATION，未启动模型/训练。下方01:16及更早缺失/等待状态均为历史。

## HP04直接迁移与独立验收

用户在本次对话明确由代理搬迁剩余文件，允许慢传，不再要求用户手动搬迁。此前源socket已失效、BatchMode连接拒绝；随后使用用户当前会话提供的认证，通过交互SSH恢复临时master，ControlPersist为12小时。密码没有写入脚本、日志、Git或自动任务，也未复制私钥或修改认证配置。主机校验保持StrictHostKeyChecking=yes。

新增script/project_tools/humanplus_large_transfer.py，6项Mac标准库文件/路径控制通过。传输前固定data/manifests/a3_humanplus_large_transfer_plan_20261018_HP04.json，包括代码SHA、原11文件路径/大小/SHA、单文件六小时网络上限和0自动失败重试；不再使用120秒慢传截止。此HP04是用户明确再授权后的直接迁移，不改HP02停止结果、不复用其部分归档，也不复跑HP03。

使用系统OpenSSH SFTP，humanplus直接从59读取，Mac只存小代码/清单。先在源逐文件核验大小/SHA，再逐文件下载到data/migrations/a3_humanplus_large_inputs_20261018_HP04/{00..10}.part，核验普通文件/路径无软链接/大小/SHA后以不覆盖硬链接放入工程，最终路径再次核验。共11文件，0既有文件覆盖、0源文件删除；没有下载整套环境或启动模型。大型暂存/模型/轨迹均LOCAL_ONLY，尚无网盘备份核验证据。

10:19:26传输完成，结果data/manifests/a3_humanplus_large_transfer_result_20261018_HP04.json，耗时112.032秒。035 PT为402,050,931字节，SHA256 9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d。10:19:57另用独立标准库流式SHA重新核验HP03的52文件和HP04的11文件，共63/63、628,994,284字节通过。小清单data/manifests/a3_humanplus_large_transfer_verification_20261018_HP04.json；原始源校验/SFTP/逐文件收据/独立验收在logs/a3_humanplus_20261018_HP04/。

执行命令为在/home/humanplus/YUANQI_A3运行/usr/bin/python3 script/project_tools/humanplus_large_transfer.py register，再execute；两者均已单次执行完成，禁止将命令当作复跑指示。started.json与最终result已存在，脚本拒绝再次执行有效载荷传输。

独立验收同时只读核验新机同boot afb6796b-3883-4df4-99ea-bc381c52ea13，可用22,611,034,112字节；GPU1396MiB/0%/43°C、无GPU计算应用/可见计算，journal1410条、0FATAL/新增OOM。没有清缓存或干预其他工程。该快照及文件通过不等于三次启动验收、ABI/PhysX/fused CUDA或完整监督器通过。历史轨迹仅来源证据，新机035匹配基线仍须重测，0张量/模型/仿真/optimizer。

阶段代码/清单/README/索引/current state按scoped审计本地提交；既有GitHub写权限未恢复，不循环重试相同失败push，仍LOCAL_COMMITTED_UNPUSHED。旧HP02/HP03计划、结果和partial保留。zhy-a3沿用原id/30分钟，提示更新为输入已齐与下一步独立新机运行验证，不再等待用户搬文件。下一步先独立登记监督器集成和环境小验证，再重测035六衣服/四官方完整匹配基线，最后原预算8更新短训练及最终checkpoint匹配评测；本轮只完成迁移。

## HP02/HP03历史记录

10/18北京时间01:16接续：01:09外部EgoLocate计算已退出，035 PT/10历史轨迹仍未到位，转WAIT_USER_LARGE_TRANSFER；后续启动重新核验资源。[只读遥测适配](README_A3_HUMANPLUS_TELEMETRY_20261018.md)完成标准库代码/14控制及两采样元数据探针，不代表完整运行保护或环境通过。下文00:33及更早占用是历史。

记录：2026-10-18。HP02批量传输STOPPED_TRANSFER_TIMEOUT_NO_RETRY；独立HP03小文件迁移COMPLETE_SMALL_FILES_ONLY，52项大小/SHA通过且已放入工程。等待大文件交接，未启动训练。

## 目标与范围

将59服务器ZHY的固定035、teacher锚点、四段MotionLib、六段合格参考和必要来源证据选择性迁往humanplus。源为root@59.77.15.15:25115的/opt/data/private/hechengzhu/ZHY，目标为/home/humanplus/YUANQI_A3。原hp3090不参与训练，Mac只存小清单和报告，源文件保留。

## 完成内容

用户重新提供认证后，已从humanplus交互登录59并建立/tmp/yuanqi-zhy-source.sock复用连接。首次连接在提交认证后关闭；一次明确限制为password认证的连接成功。没有保存密码、复制私钥或修改authorized_keys。首次在用户指定端点接受ED25519主机键，指纹SHA256:JiB72vZg5qqp85lPhM6Ns2NMh3MldA05ZK729E3l1PI，随后StrictHostKeyChecking=yes匹配；这是首次信任记录，不是独立带外身份核验。

只读确认源HEAD仍48449c427cd37f247f7918a624b0eb5b99d212f3，boot仍d994e71f-bae0-46ba-a5f7-6c2163771c6c；E51目录不存在，E45 STOPPED、E50 COMPLETE。没有在源启动模型、仿真、训练或恢复旧实验。源shell已有/generate_container_user缺失警告，未改shell配置。

源清单63项、628,994,284字节。10项直接训练输入与humanplus已有固定E51计划的SHA一致；六份reference.csv核对E23合格记录，保留原分区，zzx仍只作DEVELOPMENT_TRAIN。四官方CSV与固定vendor一致，单列为已存在资产而未重复搬迁。历史轨迹限定为六衣服/四官方，旧日志和其他动作未全目录复制。

## 版本、命令与证据

- 计划：data/manifests/a3_humanplus_migration_plan_20261018.json，登记于传输前，SHA256 0f45897b059c1e36edabe8ea34959e2cfbb4cf12cbfe123c8479c32fe95656de。
- 原始清单：新机logs/a3_humanplus_20261017/source_transfer_inventory.json。
- 传输收据：新机logs/a3_humanplus_20261017/input_transfer_receipt.json。
- 独立暂存：新机data/migrations/a3_humanplus_inputs_20261018_HP02/payload.tar.part。
- 源未安装rsync，未新增安装；使用系统tar，--null/--verbatim-files-from按固定文件表传输，不扫描整个目录。
- SSH复用：先进入humanplus，再用ssh -S /tmp/yuanqi-zhy-source.sock -o BatchMode=yes -o StrictHostKeyChecking=yes -p 25115 root@59.77.15.15执行只读源操作。socket失效不能从旧成功记录推定可继续认证。

## 结果与边界

固定120秒传输预算于120.101秒触发停止，收到333,199,360字节，状态STOPPED_TRANSFER_TIMEOUT_NO_RETRY。归档未完成，0文件放入目标工程、0文件被宣称SHA验收通过；不从部分归档运行模型或自动续传。它是传输预算停止，不是计算失败或机器故障。

待迁11个大文件共613,119,744字节：官方model_step_200000.pt为402,050,931字节，另10份历史timeseries为12.0–62.8MB。按用户已定慢大文件规则，不持续强传，逐文件源/目标/大小/SHA已写data/manifests/a3_humanplus_large_transfer_handoff_20261018.json；历史轨迹不能作为新主机效果基线。HP02状态不改，部分归档不复用。

HP03另固定52个小文件/15,874,540字节、60秒单次预算，不含以上11个慢传大文件。计划data/manifests/a3_humanplus_small_migration_plan_20261018.json，SHA256 ca04ec7d47c647ed1fe2c999b3cd8607446878761a462e912f7f0957c5e7b311。一次传输7.115秒完成，归档15,943,680字节。先在新的data/migrations/a3_humanplus_small_inputs_20261018_HP03/verified_files内逐成员核对相对路径、唯一性、普通文件类型、大小及SHA，52/52通过；再完成全部目标不存在及父目录非软链接检查，最后硬链接放入工程并再次SHA校验。0既有文件覆盖，0旧账本状态复制为新实验。结果为data/manifests/a3_humanplus_small_migration_result_20261018.json；四官方CSV在固定vendor原位额外SHA通过。10项关键训练输入中9项已齐，035 PT仍缺；teacher E11/E12、四MotionLib、六份合格reference.csv及所选小型来源合同均已验证落地。源机器的绝对路径保留在历史合同内，未来新机独立登记须显式重建本地路径，不能直接调用旧ZHY登记器。

新机00:00只读快照：同boot afb6796b-3883-4df4-99ea-bc381c52ea13，可用内存19,549,663,232字节；EgoLocate仍占3278MiB计算显存，不能启动本工程计算。新机当前会话及可见cgroup祖先memory.max/high均max，memory.events各项0；尚未实现新的连续资源门。普通用户journalctl -k -b -o json可读当前boot的1410条内核记录，包含约2.10秒起的启动记录；仅核实读取入口，未把它当连续可靠监控验收。没有全局清缓存、调整资源阈值或干预他人任务。

迁移后00:33再次只读核验：同boot，可用19,429,961,728字节，GPU4599MiB/6%/50°C，EgoLocate计算显存仍3278MiB；journal当前boot仍1410条，按项目原FATAL表达式未匹配严重事件。原始证据logs/a3_humanplus_20261017/after_migration_inventory.json。不是连续监控或运行验收，无确认机器故障。

## 保存与同步

源模型/数据不删。暂存归档、原始日志和迁入工件LOCAL_ONLY；不声称网盘备份。计划、小清单和本报告按scoped审计本地提交，既有GitHub SSH写权限未恢复，不能称已推送；提交身份与完整hash由新机Git历史给出。

同一zhy-a3 heartbeat已更新为HP02停止/HP03完成及大文件交接恢复入口，仍ACTIVE、原30分钟节奏和原聊天，未新建任务；提示不含凭据。旧yuanqi-a3/hp3090与每日yuanqi未改。状态无变化时静默，不重复迁移、催问密码或把环境准备称作策略提升。

## 下一步

HP02不续传、HP03不复跑。优先按大文件交接清单迁入035 PT并核对402,050,931字节/SHA256 9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d，其余10历史轨迹也单独核验。源保留所有原件。输入齐全且其他计算结束后，独立登记新机运行验证，重测035的六衣服/四官方匹配基线，再做原预算8更新训练。可独立完成cgroup v2与journal监控的代码适配，但运行验收仍须另登记，不把读取元数据算通过。035仍baseline，没有策略效果提升证据。
