# BONES-SEED元数据接收与来源组准备

记录日期：2026-10-24，北京时间。阶段：SM-Q1-BONES-P01。状态：三文件接收与完整校验完成；语义审计、划分及动作包待完成。

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

## 保存与下一步

原始数据和收据状态均LOCAL_ONLY，未做网盘备份。README、目标和索引同次保存；提交与远端验证以Git历史/发布收据为准。

下一编写结构化metadata审计：CSV/parquet身份与关键字段一致性、事件覆盖和非法区间、原动作来源组、可用类别和确定性候选划分。完整动作包预计约20:30下载，须重新查目录并核验实际稳定文件，不能按估计时间认定完成。训练存储保护和有限worker清理仍待实现。
