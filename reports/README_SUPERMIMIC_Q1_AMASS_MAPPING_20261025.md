# AMASS P05 映射与足底诊断

记录日期：2026-10-25。阶段P05，状态：A因错误前提失败；B真实诊断完成。

## 目标与范围

使用已接收AMASS/SMPL-X继续SuperMimic for Q1。固定P04 B001四段未准入候选，分开定位肘腕残差和足底目标/解后几何；不改原参考、质量门或正在运行的PHUMA R02。BONES获取已交师兄，不继续处理。

## 完成内容与决定

新增`script/supermimic_q1_sources/diagnose_amass_mapping.py`。准备A沿用交接中“肘腕共用elbow_link”的错误判断，未先核对实际映射。01:32服务器执行前提检查失败；随后核对父提交及服务器冻结映射，均已是elbow→elbow_link、wrist→wrist_yaw_link，SHA`dd18c57aeac6d6fb398e5960033dc3a03f339a790c3635c0f8099f3715d1f7df`。该假设撤回，不据此修改正确的映射。

修正B改为核对实际肘腕固定连杆，量化目标间距是否与真实固定长度兼容。固定长度L的两点，等权位置残差平方和下界为`(||p1-p2||-L)²/2`；同时保持各部位残差及足底诊断。这不预设上肢或足底是唯一原因。

按真实Q1足部碰撞几何重算映射目标与解后最低脚底，分别比较原整体落地偏移和整段常量落地后的曲线。逐帧重现P04支撑比例、最大间隙、全局残差，并核对刚体几何查询与MuJoCo。此步骤0次新IK/物理积分/optimizer；人体原始足底接触不在本阶段验证范围。

## 版本与验证

权威工作区`/tmp/yuanqi-supermimic-q1-20261023`，分支Q1，父提交`524d3d697aae264f6e52c622686cddae81403a04`。

- 修正B诊断SHA：`d84b9dd5c1afad3829802a8348c63a6521a049699e6f08e3d43098971dadb3f2`。
- 修正B控制SHA：`a7946826e7bce39e0d86d99de7e58b390f97d0f93e7ed08fcd3fe472536773be`。
- 复用几何helper SHA：`8b445549ce41fc850600ca872f0f689a8602fd6fa0763363ac35afa13fa648c3`，已与服务器一致，不调用其历史analyze入口。
- Mac四项控制通过：L=0中点/偏移、真实固定长度精确拟合、拉伸目标的下界；B服务器控制和真实诊断待执行。A两端两项数学控制通过没有验证实际映射前提，不能作为实际诊断通过。

执行前要求OOM计数仍14、共享内存低于28.75GiB，至少100GiB磁盘余量。独立guard每0.25秒检查，输出512MiB上限，最多900秒无进展，仅回收自身worker；不放宽原训练30GiB保护。

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python \
  -m unittest discover -s script/supermimic_q1_sources -p test_diagnose_amass_mapping.py -v

env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python -u \
  script/supermimic_q1_sources/diagnose_amass_mapping.py \
  --root /opt/data/private/YuanQi_Q1 \
  --motions /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/AMASS_P04_Q1_B001_20261025/references \
  --checks /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/AMASS_P04_check_B001_20261025 \
  --output /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/AMASS_P05B_mapping_diagnosis_20261025 \
  --expected-oom-kill 14
```

## 结果、保存与下一步

01:27读取原PHUMA R02仍RUNNING：4175/12000轮、267200交互、8347.11秒，三个原PID活跃，GPU38–39摄氏度/OOM14，盘余1965428817920字节；本次未重复全量冻结审计，最近00:56全量469项通过。原有限训练和自动配对评测继续，不能据在线日志声称效果成功。

准备A发布`5f7a9b998df68b9548c5e46329d1f2b84835da48`已推送且独立核验Q1。A真实运行01:32:20–25，guard831025/worker831026，原生退出1、worker已回收，OOM14；没有完成诊断收据/改变参考。失败原件保留`AMASS_P05_mapping_diagnosis_20261025/guard.json`及`logs/AMASS_P05_mapping_diagnosis_20261025.log`，[guard小副本](../data/manifests/supermimic_q1_amass_P05A_failed_20261025.json)保存。本记录明确修正前文准备判断，历史提交不改写。

B新输出预定`AMASS_P05B_mapping_diagnosis_20261025/guard.json`及`diagnosis/receipt.json`，数组留服务器，stdout为`logs/AMASS_P05B_mapping_diagnosis_20261025.log`。原件LOCAL_ONLY。修正B待本阶段提交推送并独立核验；不复用A失败目录。

下一实际部署并运行控制/诊断；只有数值支持后才新建独立映射试验。P04仍0/16准入，AMASS来源别名与受试者划分未完成，不能混入R02或最终测试。30分钟/1小时合格库及Q1移动策略目标仍未完成。

## P05B真实结果

修正发布`0a386022f372bbf37215c3b16be248a670475a80`已独立核远端，服务器逐SHA一致、四项控制通过。01:38:02–11，guard831289/worker831293原生退出0并回收，四段共804帧全部五项对账通过（20项）。刚体足底与MuJoCo最大差`3.747e-16m`，原P04足底比例/最大间隙/残差精确重现；OOM14，结束共享内存29276663808字节，输出实际分配176128字节。

| 来源 | 目标整段落地后足底跨度 | 目标/解后15mm内比例 | 目标/解后曲线RMSE |
| --- | --- | --- | --- |
| KIT/10 Circle01 | 26.763mm | 41.79% / 9.45% | 19.479mm |
| KIT/10 Circle10 | 25.881mm | 51.74% / 10.95% | 19.066mm |
| KIT/314 slow06 | 14.582mm | 100% / 98.51% | 3.354mm |
| KIT/4 Circle02 | 20.228mm | 75.62% / 23.88% | 7.847mm |

目标本身已有支撑起伏，前三段圆走还存在额外IK偏差；本表三段圆走指两段KIT/10与一段KIT/4。仅整段平移不能消除起伏，不能把修腕部映射或增加迭代当成足底问题已解决。此为映射目标和Q1几何归因，没有验证人体原始足底接触。

实际肘腕固定距离两侧均0.126982m，但八个侧别/动作的目标距离P95为0.163126–0.250819m；最小两点残差平方和下界的帧均值为0.000406–0.002256m²。各片腕部P95为33.32–88.00mm；两段KIT/10左踝P95为100.25/102.79mm，故问题不限于上肢。证据支持现有全局缩放加校准位移目标存在刚性长度不兼容；没有据此认定当前合格PHUMA库或策略失败唯一主因。

[诊断原样小收据](../data/manifests/supermimic_q1_amass_P05B_diagnosis_20261025.json)SHA`1df0393ef4c141edf7b63a265f5478479953c886d659a6baa4a3e92f2c2dbc2b`；[guard](../data/manifests/supermimic_q1_amass_P05B_guard_20261025.json)SHA`2db582b90e48eea52841d9f436132948369979470137005771c7e17e9ca8004f`。服务器原件/四份数组留预定B目录，LOCAL_ONLY；0新IK/积分/optimizer，0参考修改/训练准入。

下一采用独立小样本改进目标尺度/校准，并保留原质量门和来源标记。先查现成GMR适配与当前差异，不盲目扩大失败映射；同时完成HumanML原动作/裁片/受试者对照，R02不变。
