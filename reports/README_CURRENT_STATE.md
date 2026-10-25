# YUANQI 当前状态与下一步

更新：2026-10-25 08:44来源复核完成及盘余0/原停止身份复核，北京时间。当前目标是SuperMimic for Q1的世界模型、想象BPTT和critic复现，最终必须由未见步行动作的真实MuJoCo跟随支持；尚未达成。BONES获取交由师兄处理，等用户通知；不再查询下载或处理凭据。现有PHUMA、AMASS和SMPL-X继续使用，旧A3/STOPPED/FAILED不恢复。

## 唯一训练与恢复

[PHUMA R02D](README_SUPERMIMIC_Q1_R02D_BOUNDED_20261025.md)保持340 train/dev6/test7、seed20261025、64环境/4槽/H32，从原R02已审计step5000继承三网络/三Adam/两采样器，固定新增7000轮，不混612新库。真实与想象统一clip20，坏梯度写入前拒绝；训练源码固定`39397087493088605e080494d7728ff458d69939`，516冻结不能修改。

08:05:45原guard[STOPPED_STORAGE](../data/manifests/supermimic_q1_R02D_storage_terminal_20261025.json)，余盘106970607616字节低于100GiB；[controller](../data/manifests/supermimic_q1_R02D_controller_terminal_20261025.json)/[train](../data/manifests/supermimic_q1_R02D_train_terminal_20261025.json)终态STOPPED_SIGNAL，实际5282/7000新轮/338048转移、累计10282轮、WM2062组/actor2012次。原controller/trainer已消失、guard只余匹配原身份的Z，不自动恢复。退出final.pt205316947字节/SHA8749a520441374b4c5818529090b9e33fa07ae70d6d304237bb5d3ef0a053d01，08:19[只读十五门](../data/manifests/supermimic_q1_R02D_exit_save_readonly_audit_20261025.json)全过，三网络/三Adam/两采样器有限、计数40110/3911/78220、516冻结保持。最终动力学配对未执行，不能称预算完成或步行改善。

07:56[保存5000独立审计](../data/manifests/supermimic_q1_R02D_intermediate5000_audit_20261025.json)十三门全过，三Adam计数38980/3798/75960及原输入/516冻结保持，执行前实际5050新轮/RUNNING。结束OOM14/盘余152.34GB、实验分配2.259GB，尚未最终配对，原预算/100GiB停止线不变。

主机`huanyu:/opt/data/private/YuanQi_Q1`；Python`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。当前可经既有18864 SSH转发访问，不写入连接凭据。04:56启动身份：boot`d994e71f-bae0-46ba-a5f7-6c2163771c6c`；guard839210/start_ticks3616742247，controller839211/3616742257，trainer839214/3616742577。恢复须重新核`/proc`非Z、boot/ticks/cmdline/组及以下三状态，不能凭旧PID认定活跃。

实验：`data/supermimic_for_q1/PHUMA_R02D_bounded_partial5000_extend7000_20261025/`；终态读取`storage_guard.json`、`controller_status.json`、`run/status.json`，stdout在同名`logs/`。原七阶段未完成；退出保存审计已过，下一安全资源条件下独立评测，不能重启旧controller或改写原终态。

## 当前数据与发现

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

08:26/08:32[共享盘0可用及停止身份](README_SUPERMIMIC_Q1_STORAGE_FULL_20261025.md)复核，08:44的[P16日志D](../logs/session_records/supermimic_q1_source_P16_readonly_20261025.log)再读盘余仍0、内存13.49GB/OOM14。原三终态保持，controller/trainer不存在、guard仅Z，P15正式目录仍未创建。正式转换/评测和新训练等待至少100GiB盘余；资源协调问题已提出，不重复索取。R02D自身约2.259GB加退出保存约205MB，不把整盘变化归本任务；本阶段只读元数据，0服务器新文件，没有迁移到overlay/tmp。所有大型原件LOCAL_ONLY，不删未知文件、不干扰其他任务。

下一保持R02D真实终态，盘余恢复后新独立评测。[P15](README_SUPERMIMIC_Q1_FLOOR_RECOVERY_20261025.md)A/B失败保留，C五控制全部通过/0跳过：floor.sameframe缓存修正和实际坐标拒绝检查、真实mesh+sphere有符号导数、穿地恢复QP。源码发布af72dd1、生产者SHA d411fedc2c2ff03ba17f04f2700ba458668b8fb6f7c4bffcfa1b8ce96a4188c0，控制SHA135fc332eee140aad7b25c82b99702e9b88fc1ed17e75a7f82293a973fde2a75。正式八组0启动/目录未建；恢复100GiB盘余后重新核资源/绑定再运行新P15并独立复算，不热改P12C/P14。不使用P10调参或启动第二训练，目标尚未达。

Git权威副本`/tmp/yuanqi-supermimic-q1-20261023`，分支`Q1`；远端`git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`。重要阶段立即README/索引、小收据、payload audit、提交推送并独立核远端，再同步用户恢复镜像。Mac不新增大工件。唯一未跟踪旧BONES下载脚本不执行/提交/删除。

历史快照完整保存在[当前状态历史归档](README_CURRENT_STATE_ARCHIVE_20261025.md)；其他项目与教学的既有入口可由[报告索引](README.md)恢复，不能用历史运行条目替代现场核验。
