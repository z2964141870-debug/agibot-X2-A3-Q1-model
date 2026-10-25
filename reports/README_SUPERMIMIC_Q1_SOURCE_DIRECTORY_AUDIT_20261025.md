# PHUMA全库命名目录复核与后续排除名单

记录日期：2026-10-25，北京时间。阶段P16。状态：08:37只读全625片命名目录复核、08:40排除影响计算完成；没有创建新库或执行训练、积分、动力学评测。

## 目标与范围

BONES获取已交给师兄，当前使用已有PHUMA、AMASS和SMPL-X。共享盘0可用期间，利用既有manifest/check小型元数据推进来源隔离核对；不修改原R02/R02D、P08B或held-out划分，不绕过100GiB保护。P15五控制已过，但正式八组仍未执行。

## 完成内容与关键发现

完整连接P08B-V2合格manifest和独立检查的625片（train612/dev6/test7），逐clip核split/source一致、准入名单覆盖及输入前后大小/SHA。相同命名source_group和相同原动作文件SHA的跨split均为0；这不等于受试者、语义或跨集合隔离认证。

11个命名空间全部完成本次名称统计：HumanML207、GRAB61、idea400185、game_motion9、fitness17、music60、humman7、kungfu25、haa50017、EgoBody23、perform14。原HumanML别名审计仍保留26未解及13集合更名候选；本阶段不以名称复核取代它。

新增两个保守排除依据：

| 目录候选 | 对应held-out | 当前train片数 |
| --- | --- | --- |
| GRAB/s1 | development_upper_426100d304963e76 | 15 |
| GRAB/s10 | test_upper_36ef37b389783134 | 4 |

目录sN在这里只是命名候选，未核原始人的生物身份。应在下一独立训练视图保守排除这些train参考，不能用“原动作不相同”证明该测试受试者独立。完整19片clip/source/SHA见[排除影响JSON](../data/manifests/supermimic_q1_source_P16_exclusion_impact_20261025.json)。另game_motion/subset_0026包含开发动作和两个训练动作；通用subset目录没有受试者语义证据，只登记复核，不自动扩张为受试者排除规则。

## 原训练与扩库影响

读取原P05库completion（与原R02登记同SHA）及其manifest/check，确认原353片全部被P08B逐来源/输出身份保留，340/6/7不变。19个候选中11片属于原R02 train；R02D继承同340输入，因此其后续结果只能按原协议报告，不能宣称受试者独立泛化。本阶段没有更改旧输入或终态，也没有把目录候选称为已证实的相同原动作泄漏。

按未变的独立check.duration_s求和：19片共75.24秒；新train视图候选为593片、2116.10秒，即35.2683分钟，仍超过30分钟工程时长。dev6/test7逐clip保持。这里只是影响算术，没有重读NPZ/保存时轴、重跑几何门或trainer准入，不能把593片说成已创建并验收的新库。

各held-out原PHUMA来源的official_group_splits字段保留：若包含train/test同名来源，不代表本项目新划分已有相同动作字节泄漏；本阶段以本项目actual split/source/SHA进行上述独立统计。原三个test_locomotion仍是fitness动作名称，不是新的平地步行测试认证。

## 版本、可复现方法与证据

执行代码基线`15941e6f259ce15cf291a5a1e151aa44a0f8c82b`，权威Git`/tmp/yuanqi-supermimic-q1-20261023`/Q1；只读访问`huanyu:/opt/data/private/YuanQi_Q1`。仅读JSON/哈希和资源，使用系统Python的`python3 -B -c`，没有部署代码或写服务器输出。

[完整可复现源码及原生stdout](../logs/session_records/supermimic_q1_source_P16_readonly_20261025.log)包含A/B两次命令的完整Python源码，均原生退出0。A的[结构化stdout JSON](../data/manifests/supermimic_q1_source_P16_directory_review_20261025.json)绑定P08B manifest SHA`8e3b9768...`/check SHA`24fbc53e...`，B进一步绑定原P05 completion SHA`e5e9d54c...`及原库manifest/check；完整SHA以JSON为准，所有绑定在读取前后相同。

08:37共享内存13534208000字节/OOM14、盘余0。0新参考、0积分/optimizer、0服务器文件。不是MuJoCo控制、参考新准入或策略效果评测。本阶段没有新增生产代码；数据复核用覆盖、来源连接、固定哈希及前后不变断言验证。

Mac另以Node内置JSON/assert独立连接两份stdout：输入绑定、625片划分/命名空间覆盖、19片source集合、11片原训练标记、75.24秒与593/2116.10秒算术、未认证/未建库标记、原生stdout保存和README证据链接共十类检查通过，源码及输出续存在同一日志。两JSON分别7286字节/SHA`47bdc52fbe1ac6ae906e60d3d1584d320373753b4ffbe0ad1eab7836064b7cf9`、7656字节/SHA`56eca5bcfb23cea004fdb4f142c62f7ceb0658603b7ef80771b070dad1dca1e9`；这是本机stdout副本身份，不冒称服务器曾写同名文件。

## 决定、保存与下一步

保留`source_isolation_certified=false`。下一独立视图在原HumanML排除上再保守排除上述19片；实现时还须重新核完整19门继承、NPZ/保存时轴和trainer准入、P10预留目录排除，不能只拿这个名单启动训练。是否真有共享受试者或跨集合重复，仍须原始元数据支持。

优先事项仍是恢复至少100GiB正式盘余后执行新P15八组和独立审计，再完成已保存退出权重的配对动力学评测/视频；不恢复原STOPPED/FAILED。现有资源协调问题保持等待，不追加重复问题。目标尚未达成。

08:44:43最终轻量复核仍为盘余0，OOM14、内存13493379072字节，原三停止终态/进程身份保持、P15正式目录不存在。源码及原生输出附在同一P16小日志D段，未启动新的正式计算。

本文、两个stdout JSON、完整源码/小日志、索引及当前状态审查并通过payload audit后提交推送Q1、独立核远端，再机械同步用户恢复镜像。同次提交号由Git历史提供。大型工件仍服务器LOCAL_ONLY，未新增网盘备份。
