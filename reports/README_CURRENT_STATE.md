# YUANQI 当前状态与下一步

当前进行：[M02/DEV-B/P15R](README_SUPERMIMIC_Q1_RELOCATED_DIAGNOSTICS_20261025.md)。11:31已完成统一根显式迁移适配及原5000父/累计10282退出策略真实开发配对、两次独立重算；两者6/6完整、0/6严格跟随，移动0.07825至0.07258m/参考0.52117m，未改善步行。原516冻结和七退出审计输入保持，实验4.09MB、盘余2.118TB/OOM14。下一执行P15R四train×两限速八组，旧训练及失败/停止终态不恢复。

更新：2026-10-25 10:52[清理与统一目录完成](README_Q1_STORAGE_CONSOLIDATION_20261025.md)。删除178938项/文件块180.70GB，剩余约299.58GB全部位于huanyu唯一实体根`/opt/data/private/ZHY`，旧三个根仅兼容软链接。280057常规文件与516冻结/5关键文件核验保持，依赖导入及三份Q1模型加载/forward通过；盘余约2.142TB，存储阻塞已解除但未新训练。下一显式绑定新根后继续P15与独立评测；SuperMimic for Q1目标尚未达成，旧A3/STOPPED/FAILED不恢复。BONES由师兄处理，等通知，不再查询下载或处理凭据。

## 唯一训练与恢复

[PHUMA R02D](README_SUPERMIMIC_Q1_R02D_BOUNDED_20261025.md)保持340 train/dev6/test7、seed20261025、64环境/4槽/H32，从原R02已审计step5000继承三网络/三Adam/两采样器，固定新增7000轮，不混612新库。真实与想象统一clip20，坏梯度写入前拒绝；训练源码固定`39397087493088605e080494d7728ff458d69939`，516冻结不能修改。

08:05:45原guard[STOPPED_STORAGE](../data/manifests/supermimic_q1_R02D_storage_terminal_20261025.json)，余盘106970607616字节低于100GiB；[controller](../data/manifests/supermimic_q1_R02D_controller_terminal_20261025.json)/[train](../data/manifests/supermimic_q1_R02D_train_terminal_20261025.json)终态STOPPED_SIGNAL，实际5282/7000新轮/338048转移、累计10282轮、WM2062组/actor2012次。原controller/trainer已消失、guard只余匹配原身份的Z，不自动恢复。退出final.pt205316947字节/SHA8749a520441374b4c5818529090b9e33fa07ae70d6d304237bb5d3ef0a053d01，08:19[只读十五门](../data/manifests/supermimic_q1_R02D_exit_save_readonly_audit_20261025.json)全过，三网络/三Adam/两采样器有限、计数40110/3911/78220、516冻结保持。最终动力学配对未执行，不能称预算完成或步行改善。

07:56[保存5000独立审计](../data/manifests/supermimic_q1_R02D_intermediate5000_audit_20261025.json)十三门全过，三Adam计数38980/3798/75960及原输入/516冻结保持，执行前实际5050新轮/RUNNING。结束OOM14/盘余152.34GB、实验分配2.259GB，尚未最终配对，原预算/100GiB停止线不变。

主机`huanyu:/opt/data/private/ZHY`；Python`/opt/data/private/ZHY/data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python`，基础环境实体为`ZHY/data/legacy_a3/env`，旧绝对路径只经兼容链接访问。当前可经既有18864 SSH转发访问，不写入连接凭据。04:56启动身份：boot`d994e71f-bae0-46ba-a5f7-6c2163771c6c`；guard839210/start_ticks3616742247，controller839211/3616742257，trainer839214/3616742577。恢复须重新核`/proc`非Z、boot/ticks/cmdline/组及以下三状态，不能凭旧PID认定活跃。

实验：`data/supermimic_for_q1/PHUMA_R02D_bounded_partial5000_extend7000_20261025/`；终态读取`storage_guard.json`、`controller_status.json`、`run/status.json`，stdout在同名`logs/`。原七阶段未完成；退出保存审计已过，下一安全资源条件下独立评测，不能重启旧controller或改写原终态。

## 当前数据与发现

- [P17目录排除新入口](README_SUPERMIMIC_Q1_DIRECTORY_EXCLUSION_VIEW_20261025.md)实现/本地五控制/真实父库只读预览通过；发布49bef24代码仅内存执行未部署，625父参考/207别名/P10十五预留目录核验通过，两份各2885行仅19条train资格变化，保留旧来源链/十九门/held-out。候选593片/2116.10秒，正式新库未建/时轴未重读/未新准入；[原生预览JSON](../data/manifests/supermimic_q1_P17_readonly_preview_20261025.json)，0新服务器文件/积分/optimizer。
- [P16全625片命名目录复核](README_SUPERMIMIC_Q1_SOURCE_DIRECTORY_AUDIT_20261025.md)：相同source_group/原动作SHA跨split0，但GRAB/s1与s10新增19片train目录候选与held-out重叠，原R02 train含11片；不宣称原训练受试者独立。下一新视图保守排除后候选593片/2116.10秒=35.27分钟，未建库/重读NPZ/准入，旧输入/终态不改。通用game_motion/subset_0026重叠只登记，不据此推断身份。全源隔离仍false。
- [PHUMA P08B-V2](README_SUPERMIMIC_Q1_PHUMA_LIBRARY_20261025.md)：612 train/2191.34秒=36.52分钟、位移候选103段/363.06秒，dev6/test7不变；[P09注册入口](README_SUPERMIMIC_Q1_SOURCE_VIEW_REGISTRATION_20261025.md)真实预检通过，未训练。HumanML原事件/目录候选已保守排除；26未解、13集合更名候选及其他命名空间未全核，`source_isolation_certified=false`。
- [P10/P10B补充步行来源](README_SUPERMIMIC_Q1_WALKING_HOLDOUT_20261025.md)：52未用块/17事件候选/15目录候选固定选15片，原source split保留、另行预留仅供补充test。转换/十九门实际完成，0/15准入：全部支撑比例失败，14最大间隙/6自穿透失败。原P10导入错误的wrapper失败与成功P10B补检均保留。15目录继续预留，新扩库须排除；没有可用的新walking动力学测试。
- [P11训练步行限速归因](README_SUPERMIMIC_Q1_WALKING_RETARGET_20261025.md)：排全部已知held-out/P10后固定四个train目录，十二组2/4/8rad/s保持root/任务/原速。2rad/s贡献解算失真，但提到4/8仍支撑不足、两片约9mm自穿透，0准入。服务器三控制、真实worker退出0，保存数组188项独立重算误差0。下一解决train接触/映射一致性，不只提高限速、不重复旧逐帧根高投影。
- [P12/P12B训练步行接触对照](README_SUPERMIMIC_Q1_WALKING_CONTACT_20261025.md)：两轮共48组、两次各8940项独立核验/8436新inverse完成。速度条件减轻锁脚失真，但最高83.42%近地仍伴22.13mm穿地/91.96%滑动，全部修正无准入；两轮加避碰共16组自穿透0。下一须原生足地/可达性约束，不再只靠软脚目标；不改R02D/P10测试。
- 同README的P12C控制D十项通过后，07:02实际24组及独立8940核验/8436 fresh inverse完成；16修正变体无穿地，但最低脚底仍悬空8.01至33.88mm、近地比例0至15.08%，0准入。加避碰7/8自穿透0，一组仍2.39mm。下一量化腿部可达性与根/足目标兼容，不只增权重、不重复旧逐帧根高投影；原训练不改。
- [P13可达性诊断](README_SUPERMIMIC_Q1_WALKING_REACH_20261025.md)24组4314帧/独立6068核验完成，修正目标有腿长外球冲突。[P14](README_SUPERMIMIC_Q1_WALKING_ROOT_OFFSET_20261025.md)四控制/八组1438帧/独立3188核验及2812新inverse完成，恒定降根52至107mm后全部穿地20至63mm、0准入。16状态只读确认原生足地不等式允许已有穿地零步；另发现运行时floor移动未清sameframe优化标记，使actual位置仍0。新P15修正及五控制已过，正式八组等盘余；旧失败原件保持，不能改写为成功。训练资产floor=0，不涉及该运行时移动操作。
- [AMASS和SMPL-X](README_SUPERMIMIC_Q1_SMPLX_INTAKE_20261024.md)：17265标准SMPL-X动作/120Hz/16betas原始库与身体模型到位；[P04/P06](README_SUPERMIMIC_Q1_AMASS_LOCAL_OFFSET_20261025.md)16片初版/4片局部偏移均0准入，失败配置不接库。

原R02 step5000开发对照完整5→6/6、跟随0/6，移动0.078251/参考0.521165m；R02C新增1810时数值FAILED保留。现有三个test_locomotion是fitness名字，位移标签不能当步行认证。源足底、映射及浮脚缺口见[D02C](README_SUPERMIMIC_Q1_PHUMA_SUPPORT_MAPPING_20261024.md)和[D04](README_SUPERMIMIC_Q1_PHUMA_REFERENCE_DYNAMICS_20261025.md)，不重复相同样本诊断，不据此声称唯一失败原因。

## 资源与下一步

10:52最新状态：[清理与归拢](README_Q1_STORAGE_CONSOLIDATION_20261025.md)完成，唯一实体根`/opt/data/private/ZHY`、四目录、约299.58GB。AMASS原料在`data/datasets/amass_bridge/amass_raw`；PHUMA/FGP/预览资源在`data/yuanqi_ops`，依赖与重要旧A3参考保留。原三根无新副本，迁移0复制，138相对链接修复；逐文件/冻结/环境及三XML均通过。后续新任务先完成新根路径登记，原严格路径身份检查不绕过。以下09时盘余0与未清理条目保留为历史现场证据，不能当当前状态。

09:34至09:42[完整private盘点](README_Q1_PRIVATE_STORAGE_INVENTORY_20261025.md)：五个一级目录合计831.41GB，两次相同；Q1为19.97GB、yuanqi_ops366.02GB、旧ZHY94.38GB，相关合计480.36GB，纠正仅Q1约20GB的范围。AMASS原库180.26GB/历史training106.73GB为主要项；两组旧A3 checkpoint独立文件SHA相同，环境也有重复，尚未清理或完成可删依赖审计。整盘同期已用增27.73GB至7.56TB、可用0，private仅绑定其中子目录；约6.73TB差额不能从此容器归属，具体写入者未定位，需平台侧核查。[原生日志](../logs/session_records/q1_private_storage_inventory_20261025.log)及[本机整理汇总](../data/manifests/q1_private_storage_inventory_20261025.json)已保存，0服务器新增/删除/移动及训练。

09:06:56最终原生复核盘余仍0/OOM14、内存13.51GB，原三停止终态/进程身份保持，P15目录未建；[P17完整日志](../logs/session_records/supermimic_q1_P17_real_preview_20261025.log)。09:02只读du确认Q1 data目录实际约19.95GB、R02D2.465GB，不跟随外部数据链接；[整理收据](../data/manifests/supermimic_q1_P17_owned_storage_summary_20261025.json)仅限本树、不归因全盘。即使清空此目录也不足恢复100GiB，正式科学执行须等待外部空间，资源协调问题已提出；本轮没有删除资产或改正式存储。已完成可用的准备和真实只读核验，不重复已过控制代替动力学结果。

08:26/08:32[共享盘0可用及停止身份](README_SUPERMIMIC_Q1_STORAGE_FULL_20261025.md)复核，08:44的[P16日志D](../logs/session_records/supermimic_q1_source_P16_readonly_20261025.log)再读盘余仍0、内存13.49GB/OOM14。原三终态保持，controller/trainer不存在、guard仅Z，P15正式目录仍未创建。正式转换/评测和新训练等待至少100GiB盘余；资源协调问题已提出，不重复索取。R02D自身约2.259GB加退出保存约205MB，不把整盘变化归本任务；本阶段只读元数据，0服务器新文件，没有迁移到overlay/tmp。所有大型原件LOCAL_ONLY，不删未知文件、不干扰其他任务。

下一保持R02D真实终态，盘余恢复后新独立评测。[P15](README_SUPERMIMIC_Q1_FLOOR_RECOVERY_20261025.md)A/B失败保留，C五控制全部通过/0跳过：floor.sameframe缓存修正和实际坐标拒绝检查、真实mesh+sphere有符号导数、穿地恢复QP。源码发布af72dd1、生产者SHA d411fedc2c2ff03ba17f04f2700ba458668b8fb6f7c4bffcfa1b8ce96a4188c0，控制SHA135fc332eee140aad7b25c82b99702e9b88fc1ed17e75a7f82293a973fde2a75。正式八组0启动/目录未建；恢复100GiB盘余后重新核资源/绑定再运行新P15并独立复算，不热改P12C/P14。不使用P10调参或启动第二训练，目标尚未达。

Git权威副本`/tmp/yuanqi-supermimic-q1-20261023`，分支`Q1`；远端`git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`。重要阶段立即README/索引、小收据、payload audit、提交推送并独立核远端，再同步用户恢复镜像。Mac不新增大工件。唯一未跟踪旧BONES下载脚本不执行/提交/删除。

历史快照完整保存在[当前状态历史归档](README_CURRENT_STATE_ARCHIVE_20261025.md)；其他项目与教学的既有入口可由[报告索引](README.md)恢复，不能用历史运行条目替代现场核验。

- 每日研究最近检查10/25：43请求均成功，六份arXiv列表逐条未变；SMR仅项目页标识/CSS，111条去重与H001保持，静默、不新增简报。最新有内容仍为[10/24](research/2026-10-24/README.md)；见[检查](README_RESEARCH_CHECK_20261025.md)与[同步回执](../data/manifests/research_watch_sync_20261025.json)，main/Q1文档已推送，09:21:54独立内容核验通过。当前Q1工程以自身最新状态为准，日报不调度计算。
