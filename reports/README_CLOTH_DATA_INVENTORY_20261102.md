# 衣服原始数据源与现有数据盘点

记录日期：2026-11-02，北京时间。阶段：CLOTH-DATA-INVENTORY。状态：原始共享登录与现有副本元数据复核完成；未开始新的衣服测试或训练。

## 目标与用户约束

用户补充最原始衣服数据在 `10.26.67.170`，用户名 `llx`，通过 Windows 访问。此来源只允许查看、读取和复制；禁止编辑、删除、移动、重命名、改权限或进行写入测试。新转换、报告、缓存和测试输出必须放在项目工作区，不能写回原始共享。密码只用于临时认证，不保存到项目文档、脚本、配置、日志或 Git。

这次精确路径先从既有记录恢复，再现场复查；不能把旧记录称为一直记得的实时位置。所有数量分清本次实际核验和历史审计，重叠来源不相加。

## 原始共享：现场已确认

经 `ssh hp_han` 登录 Windows 的 `DESKTOP-3C1DPPR`，以临时 SMB 会话完成认证。2026-11-02 00:01:49 至 00:08:55 北京时间仅列目录和文件元数据，没有复制或打开原始数据内容，也没有源端写操作。

实际路径是：

```text
\\10.26.67.170\HumanPlus1000\common\L-Sequence
\\10.26.67.170\HumanPlus1000\common\S-Sequence
```

旧记录省略了 `common` 层；后续以本次实际路径为准。

- `L-Sequence` 现场统计为 49 个六位日期目录，保存前 40 个日期名作为有界样本。未递归统计录制总数；旧“722 长会话 / 抽样 148”只保留为历史证据，不能当当前总量。
- `S-Sequence` 直接包含 `Calibration`、`Demo100`、`human plus weave`、`jiajun`。
- `Demo100` 下有 `BigHouse`、`EastGate`、`Factory`、`HaiCang`、`Home`、`School` 六类目录。
- 最早日期的首层抽样只见 `Pi1` 子目录，未往下读取；设备目录不计为录制会话。

只使用临时 PSDrive 和 `Get-ChildItem` / `Test-Path`。结束后解除临时映射；解除映射不删除源文件。

## 当前有哪些数据

当前用户25386唯一根为 `/opt/data/private/ZHY`。以下服务器结果在 2026-11-01 23:54:23 北京时间实际核了目录、文件数量和大小；没有加载模型或大张量。旧 `/opt/data/private/yuanqi_ops/data` 已归拢到当前 `data/yuanqi_ops`，不能继续拿旧路径当新目标。

| 数据 | 当前位置 | 本次核验与用途边界 |
| --- | --- | --- |
| CLOTHO 发布衣服训练数据 | Mac `/Users/yu/Human+/data/train_released.zip`；服务器 `/opt/data/private/ZHY/data/yuanqi_ops/fgp/train_released.zip` | 两处均 1,896,104,195 B。ZIP 目录 1863 项，含 acc 的会话 206，pose+joint 206，acc/rot/pose/joint/device2bone 五类齐全 197；9 个太极目录缺 device2bone。含走路、跑步、单腿站、下蹲、热身等。用户此前已确认是自采衣服训练数据；发布训练集不能充当独立测试集。 |
| Demo100 原始副本 | `/opt/data/private/ZHY/data/yuanqi_ops/demo100_ingest/raw` | 368 文件：184 JSONL、92 CSV、92 JSON，共 1,880,778,961 B，对应历史 92 套。历史审计为 noGT，派生 root 填零。 |
| SMPL 池 | `/opt/data/private/ZHY/data/yuanqi_ops/demo100_ingest/smpl_pool` | 95 PKL + 3 JSON，共 222,971,411 B；95 文件不代表额外 95 场独立采集。 |
| 长会话预览与派生 | `/opt/data/private/ZHY/data/yuanqi_ops/humanplus_preview` | 91 HDF5；全部 734 文件、37,657,763,923 B。只核元数据，未读 HDF5 内容或重新跑前端。 |
| A3 机器人动作派生 | `/opt/data/private/ZHY/data/yuanqi_ops/training/p2_main/fgp_fitted_motionlib` | 90 PKL + 1 清单，总 127,921,296 B。旧配方 SEQ+shift3、drop_u4_l1st、device2bone identity；根来自标签、关节24-31补零，不能直接作为 Q1 参考。 |
| 既有选片审计 | `/opt/data/private/ZHY/data/yuanqi_ops/gen_audit/sequences_v7` | 10 个会话，31 文件 / 14,853,844 B；是选片派生，不加到原始录制数。 |
| T11 衣服前端开发诊断 | `/opt/data/private/ZHY/data/yuanqi_ops/experiments/teleop_20261022_T11_clotho_activation` | 走路 / 单腿站 / 热身 3 个 NPZ，全部 9 文件 / 2,823,911 B。结果是历史开发片诊断，本轮未重跑。 |
| E23 A3 参考 | `/opt/data/private/ZHY/data/legacy_a3/experiments/a3_zhy_smpl_hull_20261013_E23/sequences` | 7 个动作目录，28 文件 / 3,964,418 B；历史仅6条过运动学检查，下蹲被拒，非Q1合格资产。 |

SMB 原始池、Demo100、91 HDF5、95 SMPL 和90 A3 PKL存在来源重叠。这是不同阶段的数据形态，不是五批互不重复的新数据。

另有两段已有手跟随录制，Mac 实际文件仍在：

- `/Users/yu/Documents/ChatGPT/YAMAHA/packaging/original-cloth/data/hand_follow_20260923_191240/egolocate.jsonl`：199,376,995 B。
- `/Users/yu/Documents/ChatGPT/YAMAHA/packaging/original-cloth/data/hand_follow_20260924_150132/egolocate.jsonl`：187,213,915 B。
- 对应小包在 `/Users/yu/Documents/ChatGPT/元启/data/sync/q1_mocap_test_20261008/<同名会话>/body_only.npz`，分别 1,945,546 / 1,913,458 B。历史已解码1969 / 1923帧，本轮只 stat，未重新加载。
- 历史用户确认这两段基本原地、trans为0；可看静态上肢，不能验证真实移动根轨迹。
- 旧 E003 / E004 `body_only.npz` 在本次所测25386位置不存在；原历史位置属hp3090。本轮不访问旧主机，也不推断文件已全局删除。

## 前端效果应使用哪个旧结论

不能只沿用10月8日旧入口的37-39 cm误差。10月22日T11纠正 LIP 的 `tanh` 激活后，三个开发片根相对24点 MPJPE 为：

| 片段 | 既有 MPJPE |
| --- | ---: |
| `183926_zj_walk` | 2.974 cm |
| `193310_yy_oneLegStand` | 2.768 cm |
| `192333_zj_warmUp` | 3.096 cm |

每条仅前300源帧，评分区间 `[59,300)` 共241帧；使用标签辅助共同骨架且有训练曝光，不是独立GT、完整片或机器人步态效果。世界根位移未证、实时条件未过，两份新增StageTran权重仍属于历史未定位项；本次没扩大搜索或推断现在绝对缺失。详见 [T11诊断](README_TELEOP_CLOTHO_DIAGNOSTIC_20261022.md)。

## 版本、证据与现有训练

本阶段只改位置记忆、清单与报告，不修改算法或现训源码。开始时权威Git基线为 `9d37d2435d9936dadc2c0c220673c8858519fcba`；完整结构化元数据见 [盘点清单](../data/manifests/cloth_data_inventory_20261102.json)。旧来源见 [10月22日盘点](README_TELEOP_RESOURCE_INVENTORY_20261022.md)、[Q1 SMPL清单](README_Q1_SMPL_INVENTORY_20261010.md)、[存储归拢](README_Q1_STORAGE_CONSOLIDATION_20261025.md)。

原MorphMimic G1在 2026-11-01 23:54:23 北京时间读到 `RUNNING / 123376 / 160000`，非有限0；仅当时进度快照，未物理评估。原执行来源 `71859ca8abe238477aa18245a31d81204da9197f` 未改，没有另起GPU作业、衣服推理、仿真或训练，也未调用goal工具、agent或修改自动任务。

## 保存与下一步

原始共享只读策略已写入AGENTS。项目小记录按显式stage / payload audit / push origin Q1 / 独立40位远端 / Mac与25386逐大小SHA发布；实际完成证据以 [旁收据](../logs/session_records/cloth_data_inventory_publication_20261102.json) 为准，不预称未核同步完成。既有大文件保持原位置，本次没有从SMB复制新数据，也未为验盘重读大文件内容。

衣服后续测试优先先区分“人体标签参考”与“衣服前端预测”，核相同片段、实际帧率、坐标/标定与世界根来源，再在项目工作区制作独立Q1输入。此次用户要求先盘点，因此本轮止于盘点交付；现有90条A3数据不能直接代替Q1重定向。
