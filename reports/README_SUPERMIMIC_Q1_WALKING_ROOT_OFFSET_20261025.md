# PHUMA P14整段常数根高度适配

记录日期：2026-10-25，北京时间。阶段P14。状态：服务器四控制、真实八组/1438帧、独立3188核验/2812次fresh inverse及穿地约束只读核验完成；八组均拒绝，0新训练参考。

## 目标与范围

用户将BONES获取交给师兄，等待通知，当前继续现有PHUMA/AMASS/SMPL-X和SuperMimic世界模型路线。[P13](README_SUPERMIMIC_Q1_WALKING_REACH_20261025.md)证明固定根与P12C修正脚目标存在保守腿长外球冲突；本阶段只用同四train及2/8rad/s的八个stance_collision组，检查整段常数根高度适配能否改善。原P11/P12/P12B/P12C/P13生产者、helper、控制和输出不改。

不使用P10预留test调参，不修改R02D的340/6/7、516冻结输入、固定7000新轮，不启动第二训练。新IK仅为诊断，不自动合库或准入，不放宽原十九门，8rad/s无原速度门资格。

## 实现与决定

新`probe_walking_root_offset.py`绑定P12C/P13完成收据、guard、独立审计、原数组与源码SHA。对每组P13必要区间求整段交集，固定`delta=min(0,min(upper)-0.005)`，须`delta>=max(lower)+0.005`且`abs(delta)<=0.12m`；不满足就拒绝，不自动改阈值或做逐帧投影。5mm是保守几何候选的预留量，不是动力学或真机安全保证。

同移根Z与所有非足位置目标，保留根XY/方向/竖直曲线形状、原速、非足相对根位置和脚部世界锚点；脚姿态、推断stance标签保持。复用已控P12C原生足地0.5mm约束及腕部避碰，独立求解，保存原/新根与目标及实际delta。遵循原保存坐标：原qpos先减公共origin shift供IK，floor取负shift，求解后加回shift；分析资产仍floor=0。

新`audit_walking_root_offset.py`不调用生产者的区间/变换/统计函数，从真实根变换和显式链重算区间及delta，再核脚锚点、非足位移、根/时间/姿态、限位、速度，fresh FK、足底/滑动和两种差分inverse。原contact auditor要求root不变，因此不复用或修改它来掩盖本阶段有意改变的根高度。

## 版本、命令与产物

权威Git`/tmp/yuanqi-supermimic-q1-20261023`，分支Q1；服务器`huanyu:/opt/data/private/YuanQi_Q1`。本地三项解析控制原生退出0、三文件语法检查退出0；本地没有执行MuJoCo控制。服务器完整四控制通过后才运行：

```bash
ROOT=/opt/data/private/YuanQi_Q1
PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -u script/supermimic_q1_sources/probe_walking_root_offset.py \
  --root "$ROOT" \
  --output "$ROOT/data/supermimic_for_q1/PHUMA_P14_walking_root_offset_20261025" \
  --expected-oom-kill 14
```

单CPU/nice10/无CUDA，28.75GiB共享内存/OOM14、100GiB余盘、128MiB实验、900秒无进展保护，guard只回收自身worker。建目录前仅检查内存、创建后由guard完整检查，避免P13首次bootstrap错误。输出`probe/{receipt.json,*.npz,dynamics/*.npz}`与`guard.json`，stdout在同名logs；独立审计输出实验根`independent_audit.json`。

## 已核状态与边界

07:37:54现场原boot/ticks/cmdline/组与三进程非Z匹配，R02D三状态RUNNING、新增4550/7000轮/291200转移、累计9550轮，WM1769/actor1719次，最新数值有限。GPU39摄氏度/1228MiB，共享内存13.81GB/OOM14，盘余250436395008字节。共享盘继续下降，本训练当时约2.054GB，不能把全盘增长归因本项目；100GiB停止线保持，若触发则保留真实终态、不自动恢复、不删未知文件。

P14真实结果见下方；几何必要区间通过不等于实际可达，支撑/碰撞四项子集通过也不等于十九门准入、动力学跟随或论文结果。当前目标尚未达成。

## 保存与下一步

代码、阶段README、索引和当前状态先审计、提交、推送Q1并独立核远端，再逐SHA部署。大型原件均服务器LOCAL_ONLY，Mac仅小证据。下一完成服务器四控制、真实八组及独立复核，按结果决定是否值得进一步完整准入；R02D仍按原预算及既定最终七阶段配对推进。

## 服务器控制与部署

准备发布`0f5a8d5a123f4953718361d9d6f8f73454e5ac2a`已显式推Q1、独立核远端完整号一致；三新文件部署前不存在，部署后逐SHA与发布一致。[控制记录](../logs/session_records/supermimic_q1_P14_root_offset_controls_20261025.log)保存原生四控制输出、命令和SHA。服务器原生退出0，四项全过/0跳过，其中真实Q1的32组限位内姿态确认降低根80mm使双hip同移而关节位置保持。它不证明目标可达或真实动作通过。

07:46:06 R02D4775新增轮/305600转移、三状态仍RUNNING，最新数值有限/OOM14，共享内存13803823104字节，盘余207749152768字节；P14目录当时不存在。原保护保持，真实P14待执行。

## 07:49真实八组及07:50独立审计

控制记录发布`a8e453abbc74fd4f3fb1e70cde95dbac5cc441d8`已推送Q1并独立核远端。07:49:31至46，guard847140/worker847141原生退出0、worker已回收；八组必要共同区间全部接受，八组IK实际完成，共1438帧。降根量52.17至107.14mm，保留原时轴、足部世界锚点、非足相对根位置和原位置/每组速度限位。

| train事件 | 上限rad/s | 降根mm | 近地<=15mm比例 | 最大穿地mm | 最大自穿透mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| Circle06 | 2 | 107.14 | 63.32% | 28.23 | 27.58 |
| Circle06 | 8 | 88.42 | 63.32% | 28.38 | 0 |
| walking_fast02 | 2 | 62.98 | 74.86% | 40.16 | 0 |
| walking_fast02 | 8 | 52.17 | 78.21% | 39.30 | 0 |
| Circle02 | 2 | 70.34 | 54.93% | 19.53 | 0 |
| Circle02 | 8 | 56.19 | 45.77% | 19.53 | 0 |
| walking_slow02 | 2 | 89.48 | 100% | 63.25 | 0 |
| walking_slow02 | 8 | 88.23 | 95.48% | 63.25 | 0 |

八组都不通过原支撑/碰撞子集，0准入；近地统计包括穿地帧，不能把比例上升称有效接触改善。近地1cm点的>0.05m/s滑动比例52.62至97.64%；两种差分均显示不良逆动力学，central3需超过现有effort cap的帧比例8.70至96.41%，root力中位0.999至164.49倍体重。它们是保存轨迹的MuJoCo软接触逆动力学诊断，不是硬件受力实测或策略积分。

07:50:11新独立auditor原生退出0，3188项数值核验、1438帧fresh FK、2812次fresh inverse通过；独立根变换/显式链重算区间与delta，检查脚锚点、非足位移、根/时间/姿态、限位、速度、足底、滑动、差分和inverse，除浮点舍入外误差0。全部原输入前后SHA保持，0策略/积分/optimizer，原P12C/P13未修改。

原样[receipt](../data/manifests/supermimic_q1_P14_root_offset_receipt_20261025.json)104258字节/SHA`fca8e09c4508125fdb1cde4ac175f3e0136862dfd5a6b6e139f79ac5a2bb10da`、[guard](../data/manifests/supermimic_q1_P14_root_offset_guard_20261025.json)SHA`4bc1621fb8f3713b15233386e52d2f86d2c6d0f2a78592cff11cb16fe780c78d`、[独立审计](../data/manifests/supermimic_q1_P14_root_offset_audit_20261025.json)SHA`e97331471a067d781597607edfd58c4a99ccc56326ca95f9a66b6b848b6300be`。服务器原件根`PHUMA_P14_walking_root_offset_20261025`，原stdout同名logs，结束分配约4.89MB/盘余186.50GB/OOM14；大型原件LOCAL_ONLY。

## 原生足地约束为何允许已有穿地

[只读探针与可复现源码](../logs/session_records/supermimic_q1_P14_native_floor_probe_20261025.log)重新加载每组原起始姿态并施加注册delta，同时读取保存的最深穿地帧，共16个状态。八个直接降根起始姿态均已穿地27.72至63.40mm；16个状态的原生足地不等式均允许零位移，最小h均为0。实际Mink源码`collision_avoidance_limit.py`SHA`53389b365ff9c09a4a92cbe23ce773480ebce9c80cb3acf23ba23d66be1763f9`在dist<=minimum时令h=bound_relaxation（本配置0），并不强制恢复正间隙。当前solver还在每帧IK前直接设置根，因此原生一阶约束不能保证修复外部置根后已有的穿地。

这明确否定了“加原生足地约束就必然无穿地”的泛化，也说明P12C只覆盖原本较高的根，未覆盖本阶段穿地初始化。它没有证明软目标未恢复间隙的唯一原因，或任何降根轨迹均不可能。第一次只读探针因analysis导入路径缺失退出1，发生在正式数组加载前；路径修正后的16状态检查退出0，失败一并记录，未新建或改写P14数组。

下一P15候选应先验证带恢复要求的有符号间隙不等式及可行初始化：在正/负间隙、不同dt、实际Q1关节及根变更后用距离有限差分核Jacobian符号，确认穿地时零步不再可行；逐步求解后必须检查实际几何，失败就拒绝，不通过软目标权重或放宽门换取准入。P15尚未实现/运行，不在已冻结P12C/P14入口上热改。

## 07:53原训练与资源

[原生现场记录](../logs/session_records/supermimic_q1_R02D_after_P14_20261025.log)重新核原boot/start_ticks/cmdline/组与三个实际非Z进程，三状态RUNNING，R02D4975/7000新轮/318400转移、累计9975轮，WM1939/actor1889次，最新数值有限。GPU38摄氏度/1228MiB、共享内存13810024448字节/OOM14、共享盘余168260894720字节；当时5000保存尚未生成。R02D约2.054GB、P14约4.9MB，不能把全盘下降归本任务。100GiB余盘停止保护及原固定预算保持，不自动重启、不删未知文件，最终353片七阶段配对仍待。
