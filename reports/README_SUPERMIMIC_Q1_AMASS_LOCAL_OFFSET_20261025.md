# AMASS P06 局部校准偏移对照

记录日期：2026-10-25。阶段P06，状态：真实四段对照完成，0/4准入，当前配置不采用。

## 目标与决定

P05B已确认原AMASS映射有目标刚性长度不兼容和足底起伏。本轮只把校准位置偏移从随根部旋转改成随对应身体部位旋转；保持同四段P04 B001、16betas/120→50Hz原速、根位置/朝向、腿长scale、Mink/quadprog、权重20/1、限速2rad/s、4次/帧及原22项质量门。初始静态拟合沿用最多32次并匹配新目标。

依据已保存的Q1 GMR配置逻辑及官方固定提交`bb1bbe40774794fceb2a7c579a3464a28e68c844`的`general_motion_retargeting/motion_retarget.py::offset_human_data`，其偏移在更新后的部位姿态中旋转。此次复用这一已公开位置偏移方式，仍使用现有求解器/固定根约束，不称完整GMR复现或算法创新；不下载模型、第三方动作或执行新vendor源码。

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

## 真实对照与决定

准备发布`82af527223074aba3261e471318e905e66382790`已独立核远端Q1，四份代码逐SHA一致；服务器三项新控制和一项既有根轨迹映射控制全部通过。01:56:55–01:57:11原生退出0，guard832020/worker832030已回收；四份201帧参考及独立22门完成，0/4通过全部门，0新增合格时长/训练资格。

| 来源 | 原/新位置残差P95 | 原/新足底15mm内比例 | 新自碰撞门 |
| --- | --- | --- | --- |
| KIT/10 Circle01 | 57.123 / 52.141mm | 9.45% / 26.37% | 失败 |
| KIT/10 Circle10 | 69.823 / 54.692mm | 10.95% / 26.87% | 失败 |
| KIT/314 slow06 | 54.086 / 58.027mm | 98.51% / 26.87% | 通过 |
| KIT/4 Circle02 | 38.947 / 43.894mm | 23.88% / 20.90% | 通过 |

两片圆走的残差下降不代表整体映射改善；慢走支撑显著恶化，且两片新增自碰撞失败。新配置不采用、不扩大到B002、不为增量放宽门、不替换旧参考。失败计数重叠：残差3、支撑比例4、最大间隙3、自碰撞2、converter_gates4；其余门仍按同阈值独立核查。

本次最终共享内存29251301376字节/OOM14，磁盘可用1965348655104字节；输出实际分配2912256字节。完整[对照小收据](../data/manifests/supermimic_q1_amass_P06_B001_20261025.json)SHA`f02d7bc5604ab1a4361686fc757801852e7b04687e55dd8cfde68d4dce9e92d3`；[guard](../data/manifests/supermimic_q1_amass_P06_B001_guard_20261025.json)SHA`cab756c857432fdfe21d6e324a98b7df68cacfa90278f04e76b504190b46375f`。服务器原件和全部数组LOCAL_ONLY，0物理积分/optimizer，原R02冻结输入未改。

下一优先已有合格PHUMA资源与来源审计。并行PHUMA B017仍须遵循P06原内存保护失败后的暂停决定，不能只缩小count盲目重试；先查资源峰值或等原训练释放后从offset160新目录完整重做。AMASS后续需更完整的体段尺度/接触适配证据，本次没有证明直接局部偏移足够。原训练有限预算/配对评测继续，未出现新有效移动策略证据。
