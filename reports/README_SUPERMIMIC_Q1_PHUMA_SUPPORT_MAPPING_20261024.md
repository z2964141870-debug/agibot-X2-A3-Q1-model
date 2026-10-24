# PHUMA足底高度映射诊断

记录日期：2026-10-24，北京时间。阶段：SM-Q1-PHUMA-D02/D02B。状态：两端四项刚体几何控制通过；D02导入失败已修正，D02B通过真实依赖导入后因内存余量保护退出1，两次均未创建样本输出。八片分析未完成，移至原训练流水线结束后。PHUMA R01继续，源码/输入/3000轮预算不改。

## 目标与范围

[D01](README_SUPERMIMIC_Q1_PHUMA_DIAGNOSIS_20261024.md)确认112候选中85片支撑比例失败、44片自碰撞失败（重叠），但尚未区分PHUMA源轨迹本身、机器人目标映射和IK残差各自对支撑不足的影响。本阶段只读八个固定训练集样本，比较同一50Hz时间轴上的源G1碰撞足底、Q1目标脚刚体姿态和原始Q1 IK参考足底。development/test不用于本次诊断选型。

新入口位于 `script/supermimic_q1_analysis/analyze_phuma_support.py`，不在当前R01冻结集合内。直接复用固定PHUMA解码/重采样和独立足底查询函数；不执行IK、积分、策略推理、训练或修改任何参考。源ZIP不重复全包顺序读取，仅按已固定成员读取并逐份核验原始大小/SHA；固定源46文件和R01的121冻结输入重新比对。

## 方法与控制

三种足底曲线各保留原始世界高度。另用每条整片曲线一个固定平移比较时间变化；源曲线乘既有腿长比例表达为Q1米。该诊断平移不写回轨迹，不是逐帧根高投影，也不证明动力学可行。Q1目标脚是任务给定的刚体姿态，不保证双脚/全部关节能同时到达。

新刚体几何查询支持sphere/box/mesh，四项控制覆盖带局部平移的旋转球、旋转盒垂直投影、网格顶点/多几何最低点和未知/空几何拒绝。Mac和服务器均原生退出0、4项通过。真实分析拟逐帧把刚体查询与MuJoCo实际几何比较，同时复现原PHUMA源FK和既有支撑统计；两次真实尝试均在样本处理前失败，不能宣称逐帧检查通过。CPU进程禁用CUDA，至少1GiB初始余量，距原30GiB训练保护线不足256MiB时停止诊断，OOM增量亦拒绝。

## 固定样本与产物

固定训练样本为两片已合格移动、两片仅支撑比例失败的移动、两片仅自碰撞失败的移动、一片仅支撑失败的蹲和一片仅支撑失败的上肢。这是针对性诊断样本，不代表整个PHUMA或独立测试集统计：

- `train_locomotion_1bc87468bec5a472`。
- `train_locomotion_26832240824d6963`。
- `train_locomotion_fe9abef3042c3c39`。
- `train_locomotion_1f75cd47fbf61bb7`。
- `train_locomotion_6cafc84ec90e5186`。
- `train_locomotion_007047b45054fd5a`。
- `train_squat_04ebe3a321c47d2f`。
- `train_upper_a5678394f936531f`。

首次预定输出`PHUMA_D02_support_mapping_20261024/receipt.json`和修正后D02B目录均未创建；D02B日志包含真实stdout/stderr，已取回并核对大小/SHA。未来独立D02C的NPZ仍只留服务器，Mac仅取小型JSON/日志，不重复既有输出目录。

## 版本与复现

当前训练版本仍 `ed4217384e7abfc8a9cea0b23b95592d4a3e19b0`。本诊断代码提交由本次Q1 Git历史提供，发布并独立核验后才部署运行；运行收据另记实际诊断发布号。Python沿用服务器已有环境，不安装依赖。复现入口：

```bash
PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" \
  script/supermimic_q1_analysis/analyze_phuma_support.py \
  --root /opt/data/private/YuanQi_Q1 \
  --motions /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_P03_references_20261024 \
  --training-spec /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_R01_B128_3000_20261024/spec.json \
  --output /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_D02C_support_mapping_20261024 \
  --code-commit "$DIAGNOSTIC_COMMIT" \
  --clips train_locomotion_1bc87468bec5a472 train_locomotion_26832240824d6963 train_locomotion_fe9abef3042c3c39 train_locomotion_1f75cd47fbf61bb7 train_locomotion_6cafc84ec90e5186 train_locomotion_007047b45054fd5a train_squat_04ebe3a321c47d2f train_upper_a5678394f936531f
```

上述为训练流水线结束且重新核验内存余量后的D02C恢复入口，当前未执行。输出排他创建，不能重跑覆盖同一目录。先查原controller终态及真实依赖/资源门，再执行真实八片分析；保存结果/失败后更新本README和两个入口、审计推送Q1。当前先等待原训练与既定匹配评测，不把诊断准备记为跟随提升。

## 首次执行失败与修正

诊断发布`210969ad6b2ee5e438d50c86e5f2d80e61eae2a1`已推送并独立核验Q1远端，两份源码部署SHA一致，服务器四项刚体控制原生退出0。首次真实执行原生退出1：导入`verify_reference.foot_bottom`时ImportError，既有函数实际名为`lowest_foot`。错误发生在创建输出目录和读取真实样本之前；原始`logs/PHUMA_D02_support_mapping_20261024.log`仅tee stdout而为空，stderr由原生工具会话返回，已明确区分留档于[失败记录](../logs/session_records/supermimic_q1_phuma_D02_failed_20261024.log)。不能将单元几何控制当作真实端到端分析通过。

本修正只改变分析脚本三处函数引用，不改R01冻结的既有验证器、参考或训练入口。后续采用独立`PHUMA_D02B_support_mapping_20261024`输出目录和日志，先在服务器执行包含真实依赖的导入检查，再分析原定八片。首次失败的退出码与原日志必须保留。

13:47:38只读核查原controller799647/worker799650仍为同一父子进程、同一启动时间和boot；1625轮/26000控制转移/599组WM/549次actor。GPU40摄氏度、cgroup30,838,591,488字节、OOM仍14。该快照不代表最终效果。

## D02B资源保护停止

修正发布`248c2cf26ae179be50e989486e0763bd0adc2d77`已推送并独立核验Q1远端；服务器分析源码SHA`8b445549ce41fc850600ca872f0f689a8602fd6fa0763363ac35afa13fa648c3`与发布一致。13:52:03真实依赖导入、原boot/start_ticks/父子进程及至少1GiB余量检查通过：cgroup31,133,540,352字节，OOM14，训练1775轮。

D02B实际运行原生退出1，在源/目标MuJoCo模型创建后的`health()`因距训练30GiB保护线不足256MiB主动中止。它尚未创建输出目录、读取八个选定成员或形成样本结果；没有新IK、物理积分、策略调用、optimizer更新或参考修改。错误日志未打印触发时具体内存，因此不编造峰值。两次失败均保留，不覆盖重跑。

13:52:53立即复查原训练仍为同一controller/worker及启动身份，status RUNNING、1775轮/28400控制转移/659组WM/609次actor；cgroup已回落30,843,056,128字节，OOM仍14。D02和D02B两个预定输出目录均不存在。结构化[失败及训练核查](../data/manifests/supermimic_q1_phuma_D02B_memory_guard_20261024.json)、[原生失败日志](../logs/session_records/supermimic_q1_phuma_D02B_memory_guard_20261024.log)已保存，日志588字节/SHA`2dc224c05a9e28476af412b873b5837eb790d30ac8050e5926b91d8a94d45c46`与服务器一致。

决定：停止在本轮训练期间并行尝试这项模型诊断，不放宽训练/诊断保护、不干扰其他作业。等待原3000轮及匹配评测流水线结束，再用独立D02C目录和最新资源核验恢复。D01的支撑/碰撞筛选计数仍有效，但源轨迹、目标映射和IK残差的具体归因尚无新证据。本阶段代码和失败记录随Q1提交，完整提交号由Git历史提供；大型源数据/权重仍LOCAL_ONLY。
