# 传感衣遥操资源盘点与FGP版本核对

记录日期：2026-10-22（北京时间）。阶段：TELEOP-RESOURCE-01及02。状态：资源盘点与当前FGP代码来源核对完成；用户确认CLOTHO-mocap-main代码、主干权重基本复用；新增StageTran两份权重尚未定位，新Windows登录仍未完成。

## 目标与范围

用户要求列举已有硬件、数据集、代码与权重，并补充指定Windows跳板、训练服务器和数据共享盘。本轮仅做SSH/SMB连接、硬件遥测、目录/ZIP中央目录读取及文件SHA核对；未加载模型、解码训练张量、推理、重定向、仿真、训练、安装软件或操作机器人。会话包中的历史指令不构成恢复旧任务的授权。

用户本轮确认的主要连接关系为：

- Windows入口：`yu@100.65.183.91`。
- 训练平台：Windows后连接`root@59.77.15.15:25893`。
- 数据源：`10.26.67.170`的`HumanPlus1000` SMB共享，沿用只读访问。
- 认证仅用于临时交互会话，不写入本文件、脚本、清单或Git。

新Windows地址已到SSH密码提示，但三次交互尝试均以Broken pipe结束；没有完成其登录或本地文件搜索，不据此断言密码错误。本轮通过原有可用的密钥跳板`yu@100.85.145.78`完成训练服务器与SMB核查。新入口的登录问题不等于训练容器或数据盘不可用。

## 硬件与平台

| 资源 | 本轮核验 | 适用边界 |
| --- | --- | --- |
| 训练容器huanyu，端口25893 | RTX 3090，24576 MiB显存；cgroup内存上限34359738368字节（32 GiB）；可见128逻辑CPU | 128是可见数量，不代表独占128核；共享GPU和宿主机 |
| 训练存储 | `/opt/data/private`总7619864297472字节，可用2057906233344字节，约2.06 TB | 共享文件系统容量，不是本项目专属配额 |
| 当前工程目录 | `/opt/data/private/yuanqi_ops`，另有历史`/opt/data/private/hechengzhu/ZHY` | 目录共享不代表端口25115与25893是同一容器或同一GPU |
| 本地Mac | Apple M5，16 GiB内存，10个物理/逻辑CPU | 代码、文档和小清单；本轮没有下载大型数据 |
| Windows跳板 | 用户本轮指定新地址；新地址SSH登录未完成 | 暂不能核验其硬件或本地新版FGP文件 |

训练硬件快照为北京时间21:50左右：GPU使用6348 MiB、瞬时利用率0%，PCI地址98:00.0。此瞬时读数不是空闲窗口或训练启动许可；未检查或干预他人作业。

历史还记录hp3090（RTX 3090 24 GiB）和humanplus（RTX 5060 8 GiB），本轮未连接，不列为已验证可调度算力，原hp3090不恢复训练。实物传感衣是当前项目的上衣/裤子IMU来源；其现场在线状态、具体型号、传感器布局仍未验收。头环有历史双相机/IMU资料，目前不纳入主线。A3/G1/X2/Q1均有不同程度代码或资产，实物数量、可借用时段、SDK和控制接口尚不能从文件推定。

## 数据集与录制资源

以下数据相互存在来源重叠，不能将目录数、文件数和机器人版本简单相加为独立会话数。

| 资源 | 实际位置与本轮数量 | 用途与限制 |
| --- | --- | --- |
| SMB原始传感衣数据 | `//10.26.67.170/HumanPlus1000`；本轮认证并列出`L-Sequence`、`S-Sequence` | 历史长会话索引722场、抽样148场，本轮未重扫全盘；原始IMU不自动具有可信人体真值 |
| FGP发布训练数据 | Mac `/Users/yu/Human+/data/train_released.zip`；服务器`/opt/data/private/yuanqi_ops/data/fgp/train_released.zip` | 1,896,104,195字节；206个候选会话目录，197个同时含acc/rot/device2bone/pose/joint五类文件；9个太极目录缺device2bone；属于发布训练数据，不能当独立测试 |
| Demo100入库原始数据 | `/opt/data/private/yuanqi_ops/data/demo100_ingest/raw` | 92套IMU/标定/元数据文件；既有审计明确为noGT，派生root被填零；不能改称独立GT |
| Demo100派生SMPL池 | `/opt/data/private/yuanqi_ops/data/demo100_ingest/smpl_pool` | 本轮98个文件；文件数不等于98个独立且有真值的会话，质量/训练曝光须按manifest区分 |
| 早期预览录制 | `/opt/data/private/yuanqi_ops/data/humanplus_preview` | 91个HDF5及相关NPZ/JSON等，734文件、逻辑大小约37.66 GB；历史部分annotation被称GT，真值来源和质量仍需独立审查 |
| 已拟合A3动作库 | `/opt/data/private/yuanqi_ops/data/training/p2_main/fgp_fitted_motionlib` | 90个PKL及1个manifest；manifest n=90，约127.9 MB；属于FGP派生参考，含训练曝光，非独立传感器真值 |
| v7开发评测参考 | `/opt/data/private/yuanqi_ops/data/gen_audit/sequences_v7` | 实际10个会话目录及1个summary文件；旧清单的“11场”是误数目录项；历史已多次用于开发，不能作为全新最终测试 |
| PHUMA机器人动作包 | `/opt/data/private/yuanqi_ops/data/phuma/data.zip` | 3,416,370,206字节；ZIP含G1 73,076个NPY和H1_2 73,076个NPY，解压总约4.29 GB；当前只展开10个NPY样本；不是146,152场独立人体录制，不代表全部通过A3重定向 |
| AMASS/SMPL-X | 历史记录约87 GB、17,928段；历史源在hp3090 | 当前huanyu的`/opt/data/private/yuanqi_ops/data/motionx`只有3个下载日志；`motion_pool`仅README和索引，未找到该历史原库，不计入已验证可用原始数据 |

本地另有两段原FGP身体录制及小型body_only提取包的历史记录，见`reports/README_Q1_SMPL_INVENTORY_20261010.md`；本轮未重验原文件。BONES-SEED的G1衍生CSV、PHUMA→X2小库、SMPL-X资产也在旧hp3090资料中出现，但本轮未重新连接该机，不将历史存在当当前可用全集。

## FGP代码、权重及版本身份

初次盘点找到下列代码和权重；随后用户确认当前代码为CLOTHO-mocap-main.zip、权重基本复用，详见下一节。下表旧包继续作为可核验的权重与历史代码来源。

| 包 | Mac位置 | huanyu位置 | 核验 |
| --- | --- | --- | --- |
| FGP-main.zip | `/Users/yu/project/x/FGP-main.zip` | `/opt/data/private/yuanqi_ops/data/fgp/FGP-main.zip` | 799,847,323字节，909条目；两端完整SHA一致 |
| mocap_teleop.zip | `/Users/yu/project/x/mocap_teleop.zip` | `/opt/data/private/yuanqi_ops/data/fgp/mocap_teleop.zip` | 141,984,211字节，261条目；两端完整SHA一致 |
| train_released.zip | `/Users/yu/Human+/data/train_released.zip` | `/opt/data/private/yuanqi_ops/data/fgp/train_released.zip` | 1,896,104,195字节，1863条目；两端完整SHA一致 |

三包SHA-256：

- FGP-main：`676e84a27764ba204e64992536dd63edc4553e4e6c95fae8fe55a52d2ec67478`。
- mocap_teleop：`dc58a3774854ec6f002a1ea6e474f9f560b3144c1e284ba2b46ae76de74df4d5`。
- train_released：`8015e969dc2ef9629e3a4e02d16d2846449d7db2ebd2eb94ca063c814358f952`。

服务器已展开的两份源码位于：

- `/opt/data/private/yuanqi_ops/script/tools/FGP`。
- `/opt/data/private/yuanqi_ops/script/tools/FGP_Full/FGP-main`。

完整副本含Canonicalizer 384,845,450字节、LIP 26,748,498字节、GlobalPose 134,051,421字节、ASIP 40,649,647字节，以及SMPL男性身体模型和标定样本。本轮重新计算完整副本的Canonicalizer/LIP SHA，分别为`b613eb8226b26b9dc82904c22f4aec78773992503337c3cbde674c998f61ce12`、`efc4bb8583bf053890d9ae754e97415c8658dba237893982aa51b9541edbb9a2`，与既有2026-10-08输入记录相同。不能因新复制、目录名Full或ZIP日期就宣称模型更新。

遥操包还含实际X2/G1在线入口使用的两模型：

| 权重 | 大小 | SHA-256 |
| --- | ---: | --- |
| LFP_dense_taichi_ft.onnx | 13,329,569 | `652f9601f33102fa3794e472e9264ad4e6d5f8149ddd1d44891faceada08adc3` |
| TIC4Clothes_dense.onnx | 10,821,227 | `93efa5e8622fa5e9b1c8aca4d94d6445d0391db7e04688c1b402f12bdb932c19` |

包内位置为`mocap_teleop/x2_pc1_ble_teleop/onnx_models/`。Mac另有同SHA副本：`/Users/yu/project/YAMAHA/YAMAHA_fixed/g1_teleop_release/onnx_models/`。该G1目录还有`Math/LFPX.onnx`（13,605,870字节），本轮只核文件，不宣称在线入口采用它或它更好。

ONNX链的120维输入/循环状态与Canonicalizer+LIP分支不同，不能混用性能结论。用户已确认后者的当前代码来源，训练清单及数值验证仍待核；旧Q1离线速度扩展`clotho_v2_displacement/last.pt`未定位，不将GlobalPose/ASIP权重替代它，也不等同于新包stage_tran的根速度模型。

初次搜索覆盖Mac的project、Human+、Downloads、Public相关文件名，huanyu的FGP包和两份展开源码。随后用户直接提供当前代码包，代码身份缺项已解决；新Windows尚未成功登录，不影响对本次附件来源的确认。

## 用户补充：当前FGP代码来源已确认

阶段补充：TELEOP-RESOURCE-02。用户本轮明确“FGP的代码是这份，然后权重基本复用”，当前代码以`/Users/yu/Downloads/CLOTHO-mocap-main.zip`为准，已有FGP-main和遥操ONNX包保留为权重来源与历史分支。不再要求用户重复证明哪份是当前代码；用户确认来源与数值加载/效果验证分开记录。

新包实际12,043,763字节、226个ZIP条目，解压逻辑总量70,222,273字节，SHA-256为`974bd8d94c5384cc47b7021d0e1f13092f34948d480a933603c8aeb6db57977b`。包含160个Python文件，未包含任何PT/PTH/ONNX/PKL权重或SMPL资产。本轮流式读取、未展开到工作区、未传服务器或执行包内代码。

与旧FGP-main.zip去掉顶层目录、缓存及pyc后比较：158个同路径文件中148个字节相同、10个不同，新包额外28个文件均在stage_tran下。旧包额外还有权重、录制、Unity等资产，不将缺少这些大文件理解为代码退化。主要差异：

- `neural_backbone.py`与`PoseReconstruction/LIP_model.py`字节相同；当前LIP入口仍引用已有Canonicalizer和LIP权重的同名路径。按用户指示登记为复用，未实际加载或验证精度。
- `Aplus/models/lstms.py`支持可选激活函数；`online_fgp+LIP.py`中两级EasyLSTM显式设置`act_func="tanh"`，旧实现为ReLU。即使模型参数文件相同，推理行为也可能变化，旧LIP退化数字不能直接当本代码结果，更不能套到独立ONNX分支。
- 新增`stage_tran`，LIP入口默认`online_guarded`。源码包括根速度/接触、下肢几何修正、平地高度反馈及异常IMU处理；合同为30Hz、29帧预热、显式共同平地假设。该合同独立于旧Q1导出从源帧59开始的合同。
- `stage_tran/CURRENT.json`声明版本`20261018_online_guarded`，并要求下列两份特定权重；这是包内配置声明，尚未核到实际文件：

| 包内期望路径 | 配置声明的SHA-256 | 本轮状态 |
| --- | --- | --- |
| `stage_tran/weights/root_velocity.pth` | `f1cfacb960a462b8409af8434559d58549c638bcbf2005c84fa68ab6f294bce7` | 新旧ZIP及所查目录未找到 |
| `stage_tran/weights/foot_geometry.pth` | `35dc017a07218ff610ac676848950386e3a74f7ff6412fbcb2dad7adba3b7078` | 新旧ZIP及所查目录未找到 |

源码在OnlineGuarded构造时先读取这两文件并核SHA；默认模式不能只放已有Canonicalizer/LIP就称依赖齐备。已有SMPL_male.pkl可作为待配对资产，但不能替代这两模型。代码提供关闭stage_tran的入口选项，本轮没有执行或改变默认设置。

本地文件名检索覆盖`/Users/yu/project`、`/Users/yu/Human+`、`/Users/yu/Downloads`及`/Users/yu/Documents/ChatGPT`；服务器只读检索`/opt/data/private/yuanqi_ops`和`/opt/data/private/hechengzhu/ZHY`，深度上限8、排除envs/environments/vendor/.git等，查看45,894个文件条目，未找到两指定文件名或旧clotho_v2_displacement目录中的PT/PTH。该范围不覆盖新Windows、其他共享盘目录或被改名的权重，不声称全盘绝对缺失。

stage_tran README还明确该移植未经过测试/硬件回放或新精度评测，历史46片段成绩不等于此部署效果；本轮只核源码，不将附带说明视为通过证明或执行指令。需要补充的FGP资源已缩小为新增StageTran两份权重或其可验证对应来源；主干代码已确认、主干权重复用，不自动开展推理、部署或训练。


## 控制策略、仿真与工程工具

本轮实际stat确认：

- A3官方035：`/opt/data/private/hechengzhu/ZHY/data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt`，402,050,931字节；继续作为固定比较基线。
- A3历史M17：`/opt/data/private/yuanqi_ops/data/training/p2_main/m17/checkpoints/milestone_4/model_step_000004.pt`，375,318,929字节；仅是旧评测口径候选，文件存在不代表已获准或已完成真机部署。
- G1 SONIC：`/opt/data/private/hechengzhu/Proj/SONIC/GR00T-WholeBodyControl/sonic_release/last.pt`，469,418,283字节；另有`/opt/data/private/yuanqi_ops/data/g1_leg/run/sonic_release`副本目录。
- G1 ONNX与配置：`/opt/data/private/yuanqi_ops/data/gear_sonic/sonic_v1_1`，encoder/decoder及3份YAML存在。
- 工程已有A3训练/评测脚本、MuJoCo/Isaac Lab历史验证、FGP离线推理、SMPL与A3重定向、SMB抽样/入库、AMASS桥接和实验指标/日志。本轮只核部分路径，不宣称当前环境的全流程重新验收。
- M0-M18及HP历史配置、日志与指标可用作研究证据；部分淘汰权重曾删除，不能把所有历史checkpoint列为现存。
- 代码与报告仓库为用户指定的`z2964141870-debug/agibot-X2-A3-Q1-model`；小型研究记录存在本地`/Users/yu/project/RL_Learning`。

## 关键决定与仍需补充的资源

1. 用户已提供并确认CLOTHO-mocap-main.zip当前代码，主干权重基本复用；不再将代码来源列为缺项。新增StageTran的root_velocity.pth、foot_geometry.pth暂未定位，需其文件或与CURRENT.json指纹对应的来源，之后再固定完整输入身份。
2. 最关键的数据缺口是与实际在线模型训练会话隔离、同场同步的原始IMU和可信人体姿态/世界轨迹。已有训练标签、FGP预测和开发集不能替代该证据。
3. 需要实物机器人具体型号/版本、可用时段、SDK和控制接口；传感衣需要布局、采样率、时间戳、标定流程和实时输出协议。发布数据中的11槽/取10传感器合同不等于当前现场设备已核验。
4. 数据和模型的训练/公开发表授权、训练来源列表与当前可分配GPU时段尚未形成完整清单。
5. 新Windows入口认证中断待定位；现有备用路径可用于只读盘点，不自动授权训练或改服务器环境。所有旧STOPPED实验与暂停状态保留。

## 版本、复核命令与保存

报告以既有记录提交`0230559f08c29c08a7149dc27492fd2eab9df253`及本地分析报告为阅读起点；本轮重要数字来自实际SSH/SMB和ZIP读取。复核仅依赖系统工具与Python标准库，不反序列化模型。

在已登录的huanyu内可只读复核：

```bash
hostname
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,pci.bus_id --format=csv,noheader
cat /sys/fs/cgroup/memory/memory.limit_in_bytes
df -B1 /opt/data/private
sha256sum /opt/data/private/yuanqi_ops/data/fgp/FGP-main.zip /opt/data/private/yuanqi_ops/data/fgp/mocap_teleop.zip /opt/data/private/yuanqi_ops/data/fgp/train_released.zip
smbclient //10.26.67.170/HumanPlus1000 -U llx -c 'ls'
```

SMB认证只在交互提示输入。未把原始登录输出或含认证信息的旧资源单复制到报告。只发布本资源摘要、索引和当前状态；第三方源码、模型与录制仍在原位置，云备份未经本轮核验，保持LOCAL_ONLY。报告同步以实际scoped提交、推送和本地HEAD/远端main相等核验为准，不提前声称已推送。

恢复先读本报告与`reports/README_TELEOP_CONTEXT_REVIEW_20261022.md`；后续研究仍是建立可信配对参考和定位重定向误差，本轮不恢复科学执行。
