# BONES G1列名勘误与V2原速接口

记录日期：2026-10-24，北京时间。阶段：SM-Q1-BONES-P02-F02。状态：文档协议修正、服务器11项控制及Mac语法检查通过；真实CSV/模型未验收。

## 问题与影响

复查BONES官方seed-viewer固定提交d9963c21bee413e7eb63c1b88ae01cdcb2cf3601发现，G1Animation.setFrame实际读取root_translate/rotate XYZ；models.ts从原始CSV保留列名，按完整关节名加_dof读关节。此前F01使用Hips和去掉_joint的列名属于误判，NVlabs转换器实际上跳过表头按位置读值，不支持那一命名结论。此前测试表头也由同一错误常量构造，因而9项通过并未发现这个接入错误。

V1没有接收真实BONES CSV、没有产生Q1参考或训练，不影响旧PHUMA/A3结果。保留F01收据及Git历史，在原README顶部明确勘误，不把旧控制记录改写成正确协议证明。

## 完成修正

新协议BONES_G1_NAMED_CSV_DEG_CM_EXPLICIT_NATIVE_HZ_V2使用root_translateX/Y/Z、root_rotateX/Y/Z及29个完整joint名加_dof，例如left_knee_joint_dof。严格列名解析、显式原生帧率、厘米/度转换、wxyz以及原速50Hz重采样保持；不默默兼容V1错误别名。新契约`script/supermimic_q1_sources/bones_g1_format_v2.json`替代V1，实际原包仍须读取核验。

测试改为独立写出的完整G1表头，不导入解析器ROOT_COLUMNS构造期望。新增拒绝Hips/短关节名回归，以及全部29关节在重排输入下逐槽验证。原9项单位、坐标、非交换旋转、120到50原速、Slerp、端点、有限/缺帧和边界控制继续通过。

## 版本与真实验证

21:29:51服务器原生退出0，11项0.121秒。Mac仅语法编译通过，数值测试未在Mac执行，仍未新增依赖。三部署文件SHA与本地对应值核对，见[精确收据](../data/manifests/supermimic_q1_bones_format_v2_controls_20261024.json)。阶段前已发布c0e77b725e804d2031cb2a704eab7094b6f488b2，代码/配置随本阶段后续提交保存。

```bash
/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python -m unittest discover \
  -s script/supermimic_q1_sources -p test_bones_g1_csv.py -v
```

日志`/opt/data/private/YuanQi_Q1/logs/BONES_P02_format_v2_controls_20261024.log`。来源是同提交g1_animation.ts与models.ts，以及ProtoMotions f69fdcaabff31846b4d2247207417b57dc788828的convert_g1_csv_to_proto.py；只记录协议证据，不复制第三方整段实现。公开可核验不等于真实原包一致。

## 结果与下一步

合成接口验证完成，新增真实Q1参考0、训练更新0；BONES主目标仍在进行。下一在真实原包到位后核完整SHA/gzip、原生Frame/列名和模型，并继续独立重定向/质检。P01来源划分及P02-A01按名单接入保持，旧实验不恢复。小记录同步Q1和两端工作区，大工件留服务器LOCAL_ONLY。
