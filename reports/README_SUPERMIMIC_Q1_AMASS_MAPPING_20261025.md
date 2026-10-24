# AMASS P05 映射与足底诊断

记录日期：2026-10-25。阶段P05，状态：准备完成，真实诊断待运行。

## 目标与范围

使用已接收AMASS/SMPL-X继续SuperMimic for Q1。固定P04 B001四段未准入候选，分开定位肘腕残差和足底目标/解后几何；不改原参考、质量门或正在运行的PHUMA R02。BONES获取已交师兄，不继续处理。

## 完成内容与决定

新增`script/supermimic_q1_sources/diagnose_amass_mapping.py`及两项数学控制。实读Q1模型存在独立`left/right_wrist_yaw_link`，但冻结映射将elbow/wrist指向同一elbow_link。两个不同等权位置目标指向同一点有残差平方和下界`||p1-p2||²/2`；本阶段量化实际冲突，尚不认定它解释全部失败。

按真实Q1足部碰撞几何重算映射目标与解后最低脚底，分别比较原整体落地偏移和整段常量落地后的曲线。逐帧重现P04支撑比例、最大间隙、全局残差，并核对刚体几何查询与MuJoCo。此步骤0次新IK/物理积分/optimizer；人体原始足底接触不在本阶段验证范围。

## 版本与验证

权威工作区`/tmp/yuanqi-supermimic-q1-20261023`，分支Q1，父提交`524d3d697aae264f6e52c622686cddae81403a04`。

- 新诊断SHA：`90681779e0c051936c63cae41e3958d37a37f615477ae83057f05d3ecf6abc9b`。
- 控制SHA：`e9466878d161a6a41913cdf663a54f713df9978afa34a1558b87c18d9098ca65`。
- 复用几何helper SHA：`8b445549ce41fc850600ca872f0f689a8602fd6fa0763363ac35afa13fa648c3`，已与服务器一致，不调用其历史analyze入口。
- Mac两项控制通过：中点达到下界、离开中点增加两倍偏移平方；服务器控制和真实诊断待执行。

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
  --output /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/AMASS_P05_mapping_diagnosis_20261025 \
  --expected-oom-kill 14
```

## 结果、保存与下一步

01:27读取原PHUMA R02仍RUNNING：4175/12000轮、267200交互、8347.11秒，三个原PID活跃，GPU38–39摄氏度/OOM14，盘余1965428817920字节；本次未重复全量冻结审计，最近00:56全量469项通过。原有限训练和自动配对评测继续，不能据在线日志声称效果成功。

真实输出预定`AMASS_P05_mapping_diagnosis_20261025/guard.json`及`diagnosis/receipt.json`，数组留服务器，stdout为`logs/AMASS_P05_mapping_diagnosis_20261025.log`。原件LOCAL_ONLY。准备记录待本阶段提交推送并独立核验远端；实际发布证据由Git历史提供。

下一实际部署并运行控制/诊断；只有数值支持后才新建独立映射试验。P04仍0/16准入，AMASS来源别名与受试者划分未完成，不能混入R02或最终测试。30分钟/1小时合格库及Q1移动策略目标仍未完成。
