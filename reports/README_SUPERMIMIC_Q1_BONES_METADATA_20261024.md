# BONES-SEED元数据接收与来源组准备

记录日期：2026-10-24，北京时间。阶段：SM-Q1-BONES-P01。状态：三文件接收、全字段一致性/事件审计与候选来源划分完成；真实动作包和Q1准入待完成。

## 目标与范围

落实[BONES主线目标](README_SUPERMIMIC_Q1_BONES_GOAL_20261024.md)，先核metadata身份，再建立原动作/镜像/裁片不跨划分的输入协议。本阶段只接收和读取用户在Windows提供的数据，不将metadata条数视为Q1合格动作数。

## 完成内容与证据

三文件从`D:\tmp`经Windows scp直接迁入huanyu的`/opt/data/private/YuanQi_Q1/data/references/bones_seed/metadata/`。源端大小与修改时间稳定，源端完整SHA与服务器完整重算逐文件一致；服务器原生验证退出0。临时接收目录已安装后移除，原Windows文件保留，Mac未存原始payload。

| 文件 | 字节 | SHA-256 |
| --- | ---: | --- |
| seed_metadata_v004.parquet | 4499876 | 7cbe5624f91ae19af3feee69cff74b2cf553fb26367c8edc4b674c3d8ea97e7d |
| seed_metadata_v004.csv | 146411608 | 71d09422034946d17dbb0db027d569408c486545bdb12f7979d55e6ffd9bbc39 |
| seed_metadata_v002_temporal_labels.jsonl | 58447630 | 302da3c319bbed4f712fa22a4d71b5288493911140cd72579e29de2ac324d634 |

parquet真实142220行，含filename、move_duration_frames、move_g1_path、is_mirror、take_name/org_name/actor/date和动作/道具/姿态标签。动作路径是`g1/csv/<date>/<filename>.csv`；不能直接采用旧的`qpos_29dof.csv`加`time.csv`目录解析器。重复`(1).parquet`未迁入。

首对body_check原动作/镜像共享take与actor/date，虽然package为Locomotion，实际标签standing且horizontal_move=0。因此选移动不能只靠package，最终必须以真实轨迹检验。JSONL首条含filename及秒单位事件区间，完整覆盖/时轴一致性尚未核查。

收据：[supermimic_q1_bones_metadata_intake_20261024.json](../data/manifests/supermimic_q1_bones_metadata_intake_20261024.json)。20:18:50恢复检查时接收和验证已完成；不回填未经记录的精确传输完成时间。

## 版本与可复现核验

阶段前Q1 HEAD为`c634e1f79f5f425f3cab79008db34bb7efb10534`。本阶段配置与报告见同次提交；元数据留被忽略的data区，不提交第三方原始数据。

在huanyu项目根可重新检查：

```bash
sha256sum data/references/bones_seed/metadata/seed_metadata_v004.parquet data/references/bones_seed/metadata/seed_metadata_v004.csv data/references/bones_seed/metadata/seed_metadata_v002_temporal_labels.jsonl
/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python -c 'import pyarrow.parquet as pq; print(pq.ParquetFile("data/references/bones_seed/metadata/seed_metadata_v004.parquet").metadata.num_rows)'
```

## 结果与边界

本阶段证明三个实际输入文件可读且迁移未改变字节。尚无完整动作包、真实CSV schema/时间/关节顺序、匹配G1模型或BONES到Q1参考验证。0新Q1参考，0新训练。SOMA与G1按实际到位包分别处理，不凭人体模型已到位推断SOMA转换可用。

## P01完整审计与候选划分

20:30:22真实审计完成，原生退出0，36.61秒，OOM前后14。完整重算三文件SHA、CSV/parquet全部142220行和所有字段相等、142220个唯一G1路径。v002时序JSONL为142220个唯一动作/352703事件，完整覆盖v004、没有非法区间/重叠或数量不符；尚未与真实CSV时长比较。标签本轮不用于裁片或训练。

`inventory_bones.py`将同演员/日期下标准take与原capture别名作并集合并，另合并原文件/镜像身份，得到71117组：train56808、development7234、test7075。修正名和原始名不同52508行。15306行因组内原动作/镜像帧数不一致暂隔离（例2765/2764帧）；这不是宣告动作损坏，等待实际时间轴解释，未为配额放行。未来裁片必须继承冻结source_group，数值重复/跨数据源重叠仍待检查。

初始候选只选无道具、站姿、neutral、walking（可含turning/standing）、101至3001源帧，每组至多一个未镜像动作。稳定哈希80/10/10划分与独立排序种子固定为`Q1_BONES_G00_20261024_v1`。可选元数据行train4639/dev600/test568；本批固定train1024/dev128/test128，合计1280独立组，实际源帧train784237/dev96910/test95071。无符合严格条件的standing补充候选，选中全为移动候选；是否真实移动须由原包轨迹再证实。未知真实帧率前不换算训练秒数。

本机与服务器6项控制各通过，覆盖take别名传递、镜像/裁片、演员/日期隔离、帧数不符隔离、类别/道具过滤、顺序稳定和事件异常。另独立读取保存索引重算142220唯一身份、71117来源组、1280独立候选及各划分，确认来源组不跨划分、候选均非镜像/未隔离且对应索引内容。代码SHA为`982dcf7df3f68e3680e02b9368a0197355a92d97fc9a41b63fa84ec66c66cd18`，部署与本机一致。

证据：[完整审计收据](../data/manifests/supermimic_q1_bones_metadata_audit_20261024.json)、[独立重算与控制](../data/manifests/supermimic_q1_bones_metadata_independent_20261024.json)。原始输出`data/supermimic_for_q1/BONES_P01_metadata_20261024/`，全划分CSV24852439字节/SHA90805cb3…，候选JSONL3013245字节/SHA8afa5b6f…，完整身份见收据；执行日志`logs/BONES_P01_metadata_20261024.log`。

```bash
/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python script/supermimic_q1_sources/inventory_bones.py \
  --metadata-root data/references/bones_seed/metadata \
  --intake-receipt data/manifests/supermimic_q1_bones_metadata_intake_20261024.json \
  --output-dir data/supermimic_for_q1/BONES_P01_metadata_20261024 \
  --expected-oom-kill 14
```

命令已执行，该输出目录拒绝覆盖；复核读既有收据，需要重做时使用新的独立输出目录。20:31再查D:\tmp仍只有三metadata、SMPL-X包和四crdownload，无完整G1/SOMA包，已询问实际完成位置。P02和新训练未开始。

## 保存与下一步

原始数据和收据状态均LOCAL_ONLY，未做网盘备份。README、目标和索引同次保存；提交与远端验证以Git历史/发布收据为准。

下一实现并测试磁盘预算及worker异常退出清理，并准备真实动作包安全索引/选取入口；包完成后直接迁服务器、检查真实CSV/时间轴/关节顺序与匹配模型，再作Q1转换和独立准入。候选划分冻结在模型选型前，不能按策略效果重新抽容易样本。
