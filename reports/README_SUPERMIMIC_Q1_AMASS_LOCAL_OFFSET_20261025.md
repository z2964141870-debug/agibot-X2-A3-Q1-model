# AMASS P06 局部校准偏移对照

记录日期：2026-10-25。阶段P06，状态：实现/语法完成，服务器控制与真实四段对照待执行。

## 目标与决定

P05B已确认原AMASS映射有目标刚性长度不兼容和足底起伏。本轮只把校准位置偏移从随根部旋转改成随对应身体部位旋转；保持同四段P04 B001、16betas/120→50Hz原速、根位置/朝向、腿长scale、Mink/quadprog、权重20/1、限速2rad/s、4次/帧及原22项质量门。初始静态拟合沿用最多32次并匹配新目标。

依据已保存的Q1 GMR配置逻辑及官方固定提交`bb1bbe40774794fceb2a7c579a3464a28e68c844`的`general_motion_retargeting/motion_retarget.py::offset_human_data`，其偏移在更新后的部位姿态中旋转。2026-10-25再次只读核对官方源码246–260行。此次复用这一已公开位置偏移方式，仍使用现有求解器/固定根约束，不称完整GMR复现或算法创新；不下载模型、第三方动作或执行新vendor源码。

## 完成内容与版本

新增`amass_local_offsets.py`、`compare_amass_local_offsets.py`及三项映射控制，路径均`script/supermimic_q1_sources/`。为复用已有独立22门，仅给`verify_amass_reference.check`增加可选target_builder，默认保持原P04映射；单独标量循环重算新目标，不能以新生产目标自比代替独立核验。

父发布`f58299e7b822ef0278561a3b40c515fc403c891e`已核远端Q1。01:48:58修改前核对R02 spec：469文件全部大小/SHA一致，AMASS verifier不在冻结清单内；三个原进程boot/start_ticks一致，原R02仍4750/12000轮、304000交互、160.37分钟、OOM14/GPU40。未修改冻结模块。

Mac语法通过；因Mac缺SciPy/Mink，三项新数值控制与一项既有映射控制跳过，不能标数值通过。服务器使用原环境验证，无新安装。真实对照必须在两组控制全部通过后执行。

## 可复现命令与产物

项目根`huanyu:/opt/data/private/YuanQi_Q1`，Python沿用原环境。先用相同环境运行unittest discover，分别指定`test_amass_local_offsets.py`和`test_verify_amass_reference.py`。

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python -u \
  script/supermimic_q1_sources/compare_amass_local_offsets.py \
  --root /opt/data/private/YuanQi_Q1 \
  --baseline /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/AMASS_P04_Q1_B001_20261025/references \
  --baseline-checks /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/AMASS_P04_check_B001_20261025 \
  --output /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/AMASS_P06_local_offsets_B001_20261025 \
  --expected-oom-kill 14
```

guard沿用28.75GiB/0.25秒、OOM基准14、100GiB盘余、输出512MiB、900秒无进展保护，仅管理新worker。预定输出`guard.json`、`comparison/receipt.json`、四份新参考及diagnostics；stdout为`logs/AMASS_P06_local_offsets_B001_20261025.log`。数组和原件留服务器LOCAL_ONLY，Git仅代码/小收据/README。

## 边界与下一步

本轮是4段针对性诊断对照，不能作为最终测试；不加入R02、不变质量门、不变旧参考/旧失败终态。几何通过仍需来源隔离、训练划分及动力学验证。先看质量门的实际变化再决定余12段或其他尺度/接触适配；0新增合格时长的状态在真实结果前保持。原训练自动配对评测未到，不据在线loss宣称移动成功。

本阶段准备记录待提交推送并独立核验Q1；详细SHA与实际运行证据随真实完成更新。
