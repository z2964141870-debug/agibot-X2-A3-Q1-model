# PHUMA P13步行目标可达性

记录日期：2026-10-25，北京时间。阶段P13。状态：四控制、正式24组/4314帧及独立6068核验/fresh FK全部完成；0新参考。

## 目标与范围

继续现有PHUMA/AMASS/SMPL-X及SuperMimic世界模型路线；BONES交师兄处理。[P12C](README_SUPERMIMIC_Q1_WALKING_CONTACT_20261025.md)加足地不等式后仍浮脚，本阶段仅用已完成、独立审计通过的四个train/24组保存数组，量化根与踝部目标的几何兼容。不读取P10补充test来选参数，不修改当前R02D的340/6/7输入、7000新轮或516冻结，不启动第二训练。

## 实现与决定

新入口`script/supermimic_q1_sources/diagnose_walking_reach.py`读取P12C原数组，核receipt/guard/独立审计和全部绑定输入前后SHA。沿Q1的hip_pitch到ankle_roll刚体链累加平移长度，作为忽略关节范围/足姿态的保守外球半径。必须先核各关节均为原点处hinge、hip直接接pelvis；实际FK脚踝距离不可超此半径。目标距离超过此上界，就至少产生超出量那么大的踝部位置误差；球内不代表实际可达。

本地三项解析控制分别原生退出0，两个新文件语法通过；Mac无MuJoCo，实际Q1控制仅安排在服务器执行，不把本地三项称四项全过。部署前复核NPZ字段，采用原生产者实际`stance_mask`；没有运行错误字段的真实诊断。

同时对原映射目标与接触修正目标分别计算。保持root XY/方向时，给出左右脚外球允许的根Z修正区间及交集，区分水平跨度无解、双脚竖直区间无交集和必要降低量。区间非空只是必要条件，不能直接生成参考或称动力学可行；此处不做逐帧根高投影、IK或积分。新控制覆盖解析距离/竖直区间、原点平移不变性、非法输入拒绝及真实Q1 256组关节限位内FK/固定hip前提。

07:10实读R02D仍3825新增轮/244800转移、原三个实际进程及三状态RUNNING，GPU39摄氏度/OOM14、盘余393.39GB；未宣称移动有效。初步只读探查提示修正目标可能超腿长，正式绑定结果待；不以聊天探针代替本阶段收据。

## 版本、命令与产物

权威Git`/tmp/yuanqi-supermimic-q1-20261023`，Q1分支；服务器`huanyu:/opt/data/private/YuanQi_Q1`。先控制/审查/提交推送/独立核远端，逐SHA部署并运行服务器全部控制后，执行：

```bash
ROOT=/opt/data/private/YuanQi_Q1
PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -u script/supermimic_q1_sources/diagnose_walking_reach.py \
  --root "$ROOT" \
  --output "$ROOT/data/supermimic_for_q1/PHUMA_P13_walking_reach_20261025" \
  --expected-oom-kill 14
```

单CPU/nice10/无CUDA；共享内存28.75GiB/OOM14、盘余100GiB、输出32MiB、无进展900秒停止，只回收自身worker。输出`diagnosis/{receipt.json,*.npz}`和`guard.json`，stdout同名logs/；0新IK/积分/optimizer/训练参考。正式结果、独立核验及诊断后的方案选择待，不提前称任何门通过。

## 保存与下一步

大型原件保留服务器LOCAL_ONLY，Mac仅收小证据；代码、README、索引、当前状态及小收据审计后按Q1规范发布并核远端，再同步恢复镜像。当前目标未达、训练最终配对待。优先完成正式可达性证据，再决定参考映射或根高度的独立有限试验；不重复单纯增加脚任务权重。

## 服务器控制通过

准备发布`024a68f9cd96788dd24be7b569369b641bffe3eb`已独立核Q1远端。首次Git推送因本机6234代理连接关闭而原生失败128；原会话结束后，用当前SSH配置/10秒连接超时重新推送退出0，独立ls-remote完整号一致，未强推或改远端。两新文件部署前均不存在，部署后SHA：producer`a6dd647a36fb10130e0f8d1fb6d5419d04b6e2678accd76f99009db8ad41d223`、test`1e9ad7a7e9d370105beef8c45c42b0ebce5550a4f23fa02928d452bc9106dca5`，与发布一致。

服务器四控制原生退出0、0跳过；真实Q1 256组关节限位内姿态的双侧外球上界及hip中心固定前提通过。原样[控制日志](../logs/session_records/supermimic_q1_P13_reach_controls_20261025.log)SHA`28cfbe5ff7d15fd67984332a1432e8aa0a5f310f5f2ee0c0817c9c0dff2e87c8`，服务器`logs/PHUMA_P13_reach_controls_20261025.log`。这没有读取正式24组数组或改变参考，正式P13待执行。

## 首次启动预检失败

正式入口首次原生退出1：新main在输出目录创建前调用完整`budget.check()`，而现有StorageBudget的allocated_tree要求目录已存在。错误发生在mkdir/worker/正式数组读取前，没有创建预定实验目录，也无guard或服务器日志。[终端回执节录](../logs/session_records/supermimic_q1_P13_bootstrap_failed_20261025.txt)保留，不能称真实P13已运行或四控制覆盖了启动顺序。

修正对齐既有P12包装器：建目录前仅检查内存，建目录后由未改guard完整核100GiB盘余/32MiB输出及共享内存再启动worker；worker继续独立完整检查。未放宽任何停止线，当前R02D未变。修正发布/逐SHA部署后才运行正式P13。

## 07:25正式24组完成

修正发布`3ba499978d8d6d5323e6914fdb1b87d60ae9acdb`已独立核Q1远端；22端口连续连接超时后用同账号/同host-key别名的443端口推送，未更改仓库、账号、凭据或主机密钥校验。部署前确认旧producer SHA且预定实验目录不存在；新SHA`8febe3606a26854aaee159efabca10393431c26adec7fef715624bc91855857d`与发布一致。07:25:06至12，guard846120/worker846122原生退出0并回收，共24组4314帧/8628次实际FK外球检查完成；输入前后SHA保持。

Q1每侧hip_pitch到ankle_roll的保守最大半径为408.270844mm。全部原映射目标均未超外球；全部16组接触修正变体每帧至少一踝目标超球，最大超出101.98mm，证明这组固定根+修正踝目标无法同时精确满足。它没有证明原目标实际可达、当前训练参考全部不可达，或这是策略失败的唯一原因。

各帧水平跨度均未超球，左右脚必要根Z区间逐帧都有交集；因此仅这个必要条件尚未排除降低根Z的方向，仍不能由此宣称降低后足姿态/关节范围/速度或动力学可行。8rad/s加避碰四片的必要降低量如下；2rad/s对应最大必要降低57.98至102.14mm。

| 固定train事件 | 修正目标帧外球冲突 | 必要降低中位数 | 必要降低P95 | 必要降低最大 |
| --- | ---: | ---: | ---: | ---: |
| Circle06 | 100% | 32.93mm | 73.70mm | 83.42mm |
| walking_fast02 | 100% | 18.31mm | 42.12mm | 47.17mm |
| Circle02 | 100% | 21.54mm | 46.93mm | 51.19mm |
| walking_slow02 | 100% | 30.68mm | 64.67mm | 83.23mm |

原样[receipt](../data/manifests/supermimic_q1_P13_reach_receipt_20261025.json)SHA`527dd2fb9473182c14fb04a30b239313273acb98cf70370492dea8bd968cf3d4`、[guard](../data/manifests/supermimic_q1_P13_reach_guard_20261025.json)SHA`9c41bf67b83caee2efa50238a6f030780748dfcb7d6d0a5ce1374e939e41c656`。服务器输出`PHUMA_P13_walking_reach_20261025/diagnosis`，stdout同名logs/；实验分配884736字节，结束共享内存13820329984/OOM14、盘余317262221312字节。0新IK/积分/optimizer/准入。

新增独立`audit_walking_reach.py`不导入P13计算函数：从原模型显式链和根变换重算半径/hip中心，fresh FK重算实际脚踝，再从P12C原目标重算外球距离、Z区间和全部关键统计，并检查输入前后SHA。发布部署后结果待。下一先独立核验，再探索整段统一常数高度适配，保留root XY/方向/时轴；不重复逐帧根高硬投影，不以质量门降低换通过。

## 07:29独立复算完成

发布`3e454d5843e371d41f1f1cb20be2bfd2644bfa60`已显式推送Q1并独立核远端；新auditor部署前不存在，SHA`32a6b621ea9f24291334b43a28b796f5710ad37a5d93d566b46507784e0b7259`与发布相同。07:29:50独立CPU进程原生退出0，6068项数值核验通过、4314帧fresh FK，解析根变换得到的hip中心、显式链半径、两类目标距离、根Z区间和全部关键统计与生产者一致；原输入前后SHA保持。0新IK/积分/策略/optimizer，没有把必要几何条件当作充分可达证明。

原样[独立审计](../data/manifests/supermimic_q1_P13_reach_audit_20261025.json)SHA`d9bb69d6c9226c933e6795a70a3a8d20b936d3586f77c4187d6f43c4a189aef9`；服务器原件实验根`independent_audit.json`，stdout`logs/PHUMA_P13_reach_audit_20261025.log`。结束共享内存14501044224/OOM14、盘余293260779520字节。P13原producer、控制、数组与失败回执均保留，之后不在原P13入口上热改。

## P14候选与恢复步骤

下一独立P14只复用相同四train、2/8rad/s、P12C脚目标/原生足地约束和腕部避碰，不使用P10预留test。先从已核P13的逐帧必要Z区间求整段共同交集；候选常数偏移`delta=min(0, min(upper)-0.005)`，同时要求`delta>=max(lower)+0.005`且绝对值不超过0.12m，否则该组拒绝，不自动改阈值或改成逐帧投影。5mm仅为该保守外球候选的预留量，不是实际关节/硬件安全保证。

统一偏移原根Z与所有非足位置目标，保留root XY/方向、竖直运动形状、原速、非足相对根目标及足部世界锚点。用独立IK/输出保存实际偏移及旧/新根和目标，后检原位置/速度、足底、滑动、自穿透与inverse；它是改变参考适配的新诊断，不能再宣称原root全量不变。不放宽原十九门、8rad/s仍无原准入资格、0新训练。真正可用仍须新协议登记、独立测试和MuJoCo跟随。

07:31:41[原生训练/资源恢复记录](../logs/session_records/supermimic_q1_R02D_after_P13_20261025.log)核实huanyu/原boot/start_ticks/组/cmdline、三个实际非Z进程及三状态RUNNING：4375新增轮/280000转移、累计9375轮，WM1699组/actor1649次、最新数值有限。GPU39摄氏度/1228MiB、共享内存13805584384/OOM14、共享盘余283457986560字节；R02D约1.849GB、P13不足1MB。100GiB停止线保持，共享盘持续下降，不把整盘写入归本任务，不删除未知文件。固定7000新轮/516冻结和最终七阶段配对不变；若保护先触发，保留真实终态、不自动重启。
