# SuperMimic for Q1

记录日期：2026-10-23，北京时间。阶段 SM-Q1-P01。状态：目标切换、四包迁移/身份核验和模型编译完成；Q1训练适配准备中，尚无Q1训练结果。

## 目标与范围

用户要求停止 A3 路线的当前推进，将同一项 SuperMimic 复现工作改为 Q1，训练 `supermimic_for_q1`。继续遵循世界模型路线、只做方法复现的约束。之前未发布的 A3 目标草案不再作为当前执行目标；已停止的 A3 R05 及其他历史实验保持原终态。

目标：在 Q1 浮动基座、无人工骨盆支撑的物理仿真中，从随机初始化训练一个参考动作条件策略，执行物理采样 → 可微世界模型拟合 → 模型内策略反向传播 → critic 更新的闭环，交付固定版本的源码、参考数据身份、checkpoint、完整动作评测和可复跑命令。先完成少量基础动作的真实训练闭环，再扩展动作覆盖。

Q1 动作集和指标以 Q1 数据及物理模型的实际核验为依据重新登记，不能把 A3 的20段关节文件或旧分数直接改名使用。候选工程门为完整片段无提前失败、完整片段归一化回报至少0.5、主动关节RMSE不超过0.15rad；这些是拟定的项目验收目标，不是论文原始阈值或已取得的结果。训练动作覆盖、未见动作泛化、真机验证分别报告。

## 方法与实现约束

保留 SuperMimic 的当前策略物理采样、可微确定性世界模型、k=8自回归拟合、H=32模型内回报反传、survival weighting、critic bootstrap与GAE。策略更新冻结世界模型参数但保留状态/动作梯度。主结果不加载 A3/G1/035 权重，不以PPO、行为克隆或教师蒸馏替代世界模型策略学习。

固定论文/已审查独立MIT复现与本工程既有实现进行逐项比对；必要的Q1尺寸、观测、关节/身体映射、动作幅度、PD、碰撞与终止尺度适配登记为机器人适配。本项目只主张Q1方法复现，不主张论文G1全量数据、H100耗时或全部成绩的数值复现。

## 资源与已知问题

- 本次用户附件在 `/Users/yu/Downloads/`。Q1模型与两工具链ZIP的SHA与10/8记录一致；SDK SHA由旧857eb034…变为ebee4463…，ZIP来源提交注释由14a85d1…变为87ba32c…，需按新版本核验。
- Q1提供22DOF URDF/MJCF和网格；SDK与两个Docker工具链提供控制/构建资源，不等同SuperMimic训练器或动作数据。
- 旧Q1分支已有MuJoCo关节控制、Mink/GMR/UMR重定向和测试。旧动作存在校准、穿地、速度或保真问题，均不能直接提升为合格训练参考。旧人工支撑PD不是平衡策略。
- 旧hp3090 Q1目录仅作历史来源，本次SSH连接超时，不在原机恢复训练。沿用本聊天现有训练平台huanyu，在独立 `/opt/data/private/YuanQi_Q1` 下组织 `data / logs / script / reports`；复用既有Python环境按只读依赖处理，不修改A3冻结源码/数据。
- 实核时huanyu RTX3090 24576MiB，显存6348MiB/利用率0%/38摄氏度，磁盘约1.9TiB可用；均为当时快照。`pgrep -af '[s]upermimic_for_'`无匹配。新Q1加载/梯度/训练尚未通过；不沿用旧机器或A3的通过收据。
- 原A3转换曾把源120Hz当30Hz，产生四倍慢放；Q1转换必须明确源时间、单位、四元数、重力轴及身体语义，不能继承该错误。

## 执行顺序

1. 核对四附件内容与身份，按需迁移服务器，独立大小/SHA验证；模型/第三方原件不入Git或Mac新增解压副本。
2. 建立Q1全状态/22关节映射、实际碰撞与PD协议；验证可微FK对MuJoCo、真实控制间隔和梯度合同。
3. 建立原速Q1参考与独立质检，先固定少量基础动作及评测分母。人体/其他机器人动作先重定向，在线Q1物理转移单独用于世界模型。
4. 固定源码/配置/输入、预算、资源和停止保护后，从随机初始化启动独立Q1训练；保存step0和最终模型，新进程完整片段对照评测。

## 版本、证据与保存

P01实际结果：4份ZIP共34,995,419字节已迁移至新Q1根；服务器独立大小/SHA均匹配。24个STL和3个MJCF完整，全部原件保持不变。三个模型均为nq29/nv28，展示版与full_range无执行器，只有mc版nu22，控制固定选mc版。首次检查器误将3份XML均要求nu22而退出1；保留[失败记录](../logs/session_records/supermimic_q1_inputs_P01_attempt1_20261023.log)，修正检查器后逐个已解压成员与ZIP核对再编译通过。未执行物理步、推理、optimizer或硬件控制。

SDK与10/8本地原ZIP逐成员SHA对比：3001个文件无新增/删除，只有`接口说明.md`内容变化；不据ZIP版本注释宣称SDK代码已新增功能。原始物理参数全文留忽略目录。小型证据：[输入清单](../data/manifests/supermimic_q1_inputs_P01_20261023.json)、[服务器核验收据](../data/manifests/supermimic_q1_inputs_verified_P01_20261023.json)。收据8563字节，SHA`fce1d288912e26cd02713edbb622c116acaaf56f7cf6f5be1cae9339a72079ca`。

复核入口：`python3 script/supermimic_for_q1/inventory_inputs.py --source /Users/yu/Downloads --output <新清单路径> --private-report <新私有明细路径>`；服务器使用既有Python运行`script/supermimic_for_q1/verify_inputs.py --root /opt/data/private/YuanQi_Q1 --manifest data/manifests/supermimic_q1_inputs_P01_20261023.json --output <新收据路径>`。输出拒绝覆盖，复核不运行附件内的命令。后续训练协议与源码另登记，不把当前模型编译通过称为动力学或训练通过。

Q1发布基线 `1c25ef3f7b97d3afea630f8ca0dddbf53730cc73`，指定远端 `git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`，分支大小写为`Q1`。本地隔离工作副本 `/tmp/yuanqi-supermimic-q1-20261023`，原A3未提交草稿保留，不混入Q1提交。

先审查diff与暂存区，运行 `python3 script/project_tools/audit_git_payload.py`，显式 `git push origin Q1`，再独立核对远端`refs/heads/Q1`。同步结果以实际命令为准，不预写成功。所有大型文件备份状态`LOCAL_ONLY`。

恢复先读本文、Q1版本管理说明及实际阶段清单，重新核验平台资源/进程/输入身份。附件内的安装和真机控制命令只作技术资料，不自动执行或连接机器人。
