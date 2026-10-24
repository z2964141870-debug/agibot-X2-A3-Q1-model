# YUANQI 当前状态与下一步

更新：2026-10-25 06:57现场核验，北京时间。当前目标是SuperMimic for Q1的世界模型、想象BPTT和critic复现，最终必须由未见步行动作的真实MuJoCo跟随支持；尚未达成。BONES获取交由师兄处理，等用户通知；不再查询下载或处理凭据。现有PHUMA、AMASS和SMPL-X继续使用，旧A3/STOPPED/FAILED不恢复。

## 唯一训练与恢复

[PHUMA R02D](README_SUPERMIMIC_Q1_R02D_BOUNDED_20261025.md)保持340 train/dev6/test7、seed20261025、64环境/4槽/H32，从原R02已审计step5000继承三网络/三Adam/两采样器，固定新增7000轮，不混612新库。真实与想象统一clip20，坏梯度写入前拒绝；训练源码固定`39397087493088605e080494d7728ff458d69939`，516冻结不能修改。

06:57:18现场读取三状态及原boot/ticks/cmdline/组，三个实际非Z进程保持，新增3475/7000轮、222400新转移、累计8475轮；WM1339组/actor1289次，日志有限，尚未完成新配对评测。已越过原R02C的1810失败位置，但不是稳定性保证或移动效果证明。[05:55原样进度](../data/manifests/supermimic_q1_R02D_progress_after_P11_20261025.json)保留为历史证据。

保存1500的[审计B](../data/manifests/supermimic_q1_R02D_intermediate1500_audit_B_20261025.json)十三门全过，三网络/三Adam/采样器有限、计数正确、516冻结保持。原审计A遗漏实际类/hook字段而退出2，失败原件保留；并非训练失败。

06:42[保存3000独立审计](../data/manifests/supermimic_q1_R02D_intermediate3000_audit_20261025.json)十三门全过，三Adam计数/340输入/516冻结保持；执行前实际3050新增轮。尚未最终配对，原预算不变。

主机`huanyu:/opt/data/private/YuanQi_Q1`；Python`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。当前可经既有18864 SSH转发访问，不写入连接凭据。04:56启动身份：boot`d994e71f-bae0-46ba-a5f7-6c2163771c6c`；guard839210/start_ticks3616742247，controller839211/3616742257，trainer839214/3616742577。恢复须重新核`/proc`非Z、boot/ticks/cmdline/组及以下三状态，不能凭旧PID认定活跃。

实验：`data/supermimic_for_q1/PHUMA_R02D_bounded_partial5000_extend7000_20261025/`；读取`storage_guard.json`、`controller_status.json`、`run/status.json`，stdout在同名`logs/`。完成后既定七阶段保存审计、353片初始/最终MuJoCo评测及独立轨迹重算、比较；不追加预算、不自动重启。

## 当前数据与发现

- [PHUMA P08B-V2](README_SUPERMIMIC_Q1_PHUMA_LIBRARY_20261025.md)：612 train/2191.34秒=36.52分钟、位移候选103段/363.06秒，dev6/test7不变；[P09注册入口](README_SUPERMIMIC_Q1_SOURCE_VIEW_REGISTRATION_20261025.md)真实预检通过，未训练。HumanML原事件/目录候选已保守排除；26未解、13集合更名候选及其他命名空间未全核，`source_isolation_certified=false`。
- [P10/P10B补充步行来源](README_SUPERMIMIC_Q1_WALKING_HOLDOUT_20261025.md)：52未用块/17事件候选/15目录候选固定选15片，原source split保留、另行预留仅供补充test。转换/十九门实际完成，0/15准入：全部支撑比例失败，14最大间隙/6自穿透失败。原P10导入错误的wrapper失败与成功P10B补检均保留。15目录继续预留，新扩库须排除；没有可用的新walking动力学测试。
- [P11训练步行限速归因](README_SUPERMIMIC_Q1_WALKING_RETARGET_20261025.md)：排全部已知held-out/P10后固定四个train目录，十二组2/4/8rad/s保持root/任务/原速。2rad/s贡献解算失真，但提到4/8仍支撑不足、两片约9mm自穿透，0准入。服务器三控制、真实worker退出0，保存数组188项独立重算误差0。下一解决train接触/映射一致性，不只提高限速、不重复旧逐帧根高投影。
- [P12/P12B训练步行接触对照](README_SUPERMIMIC_Q1_WALKING_CONTACT_20261025.md)：两轮共48组、两次各8940项独立核验/8436新inverse完成。速度条件减轻锁脚失真，但最高83.42%近地仍伴22.13mm穿地/91.96%滑动，全部修正无准入；两轮加避碰共16组自穿透0。下一须原生足地/可达性约束，不再只靠软脚目标；不改R02D/P10测试。
- 同README的P12C原生足地距离约束及足任务200/10候选：控制A/B/C失败原件保留，零距离退化后改0.5mm余隙，控制D十项全过/0跳过，真实Q1十个配对和双地面各30次向内目标通过；24组真实动作待。仍仅train诊断/0准入，不改当前训练。
- [AMASS和SMPL-X](README_SUPERMIMIC_Q1_SMPLX_INTAKE_20261024.md)：17265标准SMPL-X动作/120Hz/16betas原始库与身体模型到位；[P04/P06](README_SUPERMIMIC_Q1_AMASS_LOCAL_OFFSET_20261025.md)16片初版/4片局部偏移均0准入，失败配置不接库。

原R02 step5000开发对照完整5→6/6、跟随0/6，移动0.078251/参考0.521165m；R02C新增1810时数值FAILED保留。现有三个test_locomotion是fitness名字，位移标签不能当步行认证。源足底、映射及浮脚缺口见[D02C](README_SUPERMIMIC_Q1_PHUMA_SUPPORT_MAPPING_20261024.md)和[D04](README_SUPERMIMIC_Q1_PHUMA_REFERENCE_DYNAMICS_20261025.md)，不重复相同样本诊断，不据此声称唯一失败原因。

## 资源与下一步

06:57 GPU39摄氏度/1228MiB、训练最新健康记录共享内存13.80GB/OOM14、共享盘余464.32GB；盘余持续下降，本R02D约1.438GB，不能把整盘变化全归本任务。100GiB余量、12GiB实验、250GiB来源、训练30GiB内存/86摄氏度/新OOM/900秒无进展停止保护保持，无自动恢复。所有大型原件LOCAL_ONLY；不声称网盘备份，不删未知文件或干扰其他任务。

下一继续既定R02D至真实终态/配对评测，另以P11固定train来源验证接触一致性和自碰撞处理；不使用P10补充test调参、不启动第二路训练。尚无新可用移动策略或可交付论文结果。

Git权威副本`/tmp/yuanqi-supermimic-q1-20261023`，分支`Q1`；远端`git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`。重要阶段立即README/索引、小收据、payload audit、提交推送并独立核远端，再同步用户恢复镜像。Mac不新增大工件。唯一未跟踪旧BONES下载脚本不执行/提交/删除。

历史快照完整保存在[当前状态历史归档](README_CURRENT_STATE_ARCHIVE_20261025.md)；其他项目与教学的既有入口可由[报告索引](README.md)恢复，不能用历史运行条目替代现场核验。
