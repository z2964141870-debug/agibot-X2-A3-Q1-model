# BONES G1独立格式与原速重采样准备

记录日期：2026-10-24，北京时间。阶段：SM-Q1-BONES-P02-F01。状态：公开源码格式核验、解析器实现及服务器9项控制完成；真实动作包尚未接收，未做Q1转换或训练。

## 目标与范围

在[BONES主目标](README_SUPERMIMIC_Q1_BONES_GOAL_20261024.md)下建立真实单CSV输入接口，避免沿用PHUMA30Hz/弧度或旧`qpos_29dof.csv`目录格式。此前P01来源划分保持冻结。本阶段为P02准备，不把合成检查当作真实数据准入。

## 公开来源核验

BONES官方seed-viewer固定提交`d9963c21bee413e7eb63c1b88ae01cdcb2cf3601`的`frontend/public/src/g1_animation.ts`使用120fps、厘米转米、角度转弧度及根Euler ZYX。NVlabs/ProtoMotions固定提交`f69fdcaabff31846b4d2247207417b57dc788828`的`data/scripts/convert_g1_csv_to_proto.py`给出Frame、Hips_translate/rotate XYZ及29个_dof命名列，并用SciPy外禀xyz生成根四元数。这与按存储X/Y/Z分量表达的内禀ZYX一致，已用非交换三轴矩阵独立验证。

精确来源URL、版本和约定保存在[bones_g1_format_v1.json](../script/supermimic_q1_sources/bones_g1_format_v1.json)。只按已读公开协议独立实现标准CSV/SciPy接口，未将第三方完整代码/模型加入Git，也未读取未完成下载或绕过数据集授权。

## 完成实现与关键决定

`bones_g1_csv.py`要求显式source_hz、精确命名列及连续整数Frame，不默认30/50/120Hz，不把缺帧重编号。按列名映射29关节模型顺序，保持原世界位置；厘米到米、度到弧度、根Euler转换为MuJoCo wxyz。可核metadata预期帧数，输入与输出均有大小/帧数上限。未知/重复列、非有限值和帧号异常拒绝。

目标50Hz使用显式原生时间、位置/关节线性插值和SciPy Slerp，保持time_scale=1。120/50不是整数抽帧比例，不能直接按2帧一取再标50Hz；本入口不靠改帧率减慢动作。只保留不超过源结束时刻的50Hz格点，端点截断不足一个控制周期，不补帧或拉伸整片。

另用三metadata实际记录核查事件时长：142220条末事件均为正；按(N-1)/120与末事件结束差的中位0.06667秒、P99 0.51667秒、最大10.98333秒，仅12282条在20ms内，30/50Hz均0条。不能把事件时段当精确CSV边界或从其推导每条真实fps。JSONL本阶段不用于训练或裁片；原包到位仍须读取实际Frame/行数并解释metadata偏差。

## 实际验证与命令

huanyu原有训练Python执行9项控制全部通过、原生退出0、0.174秒：单位/原点/wxyz、重排列、非交换Euler矩阵、120→50时长与恒速、±179度跨界Slerp、非整端点不补帧、列约束、非有限/缺帧/帧号检查、显式fps/帧数/大小边界。三部署文件SHA与Mac逐项一致。

Mac默认Python缺scipy，第一次unittest在导入时失败、9项未实际执行；随后仅做语法编译通过，未安装新包，不记作本机数值验证通过。实际数值证据来自服务器同版代码。

```bash
/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python -m unittest discover \
  -s script/supermimic_q1_sources -p test_bones_g1_csv.py -v
```

收据：[supermimic_q1_bones_format_controls_20261024.json](../data/manifests/supermimic_q1_bones_format_controls_20261024.json)，含代码SHA、原生控制输出和时序标签统计。日志`logs/BONES_P02_format_controls_20261024.log`。阶段前Q1已发布`e18ce215083d99cfd53fd3e6bb87a774530a00d5`；当前代码/报告由本次后续提交保存。

## 结果、存储与下一步

20:51复查D:\tmp仍只有未完成动作下载，最大crdownload2720542770字节；不视为可用原包或据此猜百分比。huanyu无动作归档，同boot/OOM14、GPU39/空闲17788MiB、盘余1965380382720字节。0新BONES Q1参考，0训练，目标未达。

下一补原包流式索引和按冻结名单提取，原包完成后核全文件大小/SHA、完整gzip与成员目录，再验证真实CSV/帧数/模型几何并重定向。P01候选的时长仍不是Q1合格时长。Mac只增小代码和报告，大文件留服务器、LOCAL_ONLY。
