# PHUMA P15足地间隙恢复

记录日期：2026-10-25，北京时间。阶段P15。状态：首轮服务器控制失败，四项中两通过/一错误/一失败；正式八组未启动。已发现地面sameframe缓存及错误球体前提，修正与补检待。

## 目标与范围

继续现有PHUMA/AMASS/SMPL-X与SuperMimic世界模型路线，BONES获取交由师兄，不查询下载或凭据。[P14](README_SUPERMIMIC_Q1_WALKING_ROOT_OFFSET_20261025.md)八组常数降根后均穿地，16状态确认原生Mink足地不等式允许已有穿地零步。本阶段先控制真正要求恢复间隙的不等式，再在相同四train/2和8rad/s/同delta、目标、时轴和初始化的八组上比较。

不修改P12C/P13/P14及其绑定helper、测试、数组，不改变R02D的340/6/7、516冻结或7000新轮；不恢复任何STOPPED/FAILED实验，不启动第二训练，不以P10预留test调参。

## 实现和决定

新增`probe_walking_floor_recovery.py`包装已控Mink有符号接触法线Jacobian：正间隙沿用原生界；低于0.5mm余隙时改为`h=gain*(distance-clearance)/dt`，`gain=0.85*dt`，使穿地时h为负，零位移不再满足。仍使用Mink/quadprog和MuJoCo几何，不改依赖包或物理引擎。

包装器在独立worker内临时替换P12C的floor_limit函数对象，复用P14原生solve和生产者；文件字节不变，退出finally恢复对象。新收据明确effective_floor_adapter、源/测试SHA、实际hook、创建limit数、调用数和恢复行数，并冻结新文件。保持原root、脚锚点、非足目标、静态32步初始化、原位置与每组速度限制。不预设QP有可行解，不放宽限制，所有输出仍诊断且0准入。

四项控制覆盖：纯解析正负间隙和dt；非法数值拒绝；双脚sphere/slide模型在正/负间隙、两地面、两dt下实际QP恢复；真实Q1的P14降根起始和最深穿地共16状态、每状态8方向，对原生G与独立球形足底高度的切空间中心差分核符号，并确认穿地零步被拒绝。只对有限激活行核导数；未激活的检测范围外行仍不施约束。

## 版本、命令与资源

权威Git`/tmp/yuanqi-supermimic-q1-20261023`/Q1；服务器`huanyu:/opt/data/private/YuanQi_Q1`。Mac只运行两项解析控制和语法检查，均原生退出0，不把它称MuJoCo控制通过。发布并独立核远端、逐SHA部署后运行服务器四控制；全部通过且资源允许才运行新`PHUMA_P15_walking_floor_recovery_20261025`。

```bash
ROOT=/opt/data/private/YuanQi_Q1
PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -m unittest discover -s "$ROOT/script/supermimic_q1_sources" \
  -p test_walking_floor_recovery.py -v
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -u "$ROOT/script/supermimic_q1_sources/probe_walking_floor_recovery.py" \
  --root "$ROOT" --output "$ROOT/data/supermimic_for_q1/PHUMA_P15_walking_floor_recovery_20261025" \
  --expected-oom-kill 14
```

正式对照维持28.75GiB共享内存/OOM14、100GiB余盘、128MiB输出和900秒无进展停止。控制只有小型内存计算/不写数据输出，也核共享内存/OOM。08:00现场原三个进程boot/ticks/cmdline/组保持；08:02:19又实读R02D5200新轮/332800转移、三状态RUNNING，盘余123965939712字节、OOM14。共享盘下降主要不是约2.259GB的本训练写入，不删未知文件或降低保护；若资源先拒绝正式P15，则保留准备及真实停止证据。

## 结果与保存

服务器控制和正式结果待，不能称恢复可行、支撑改善、十九门准入或策略成功。复用独立`audit_walking_root_offset.py`核相同P14变换不变量、fresh FK/滑动/两差分inverse，因本阶段根和目标协议不变，无需修改旧auditor。发生新失败仍保留真实结果，不自动热改旧记录。

关键代码、README、索引及小收据审计后提交推送Q1、独立核远端并同步用户镜像；大工件仍服务器LOCAL_ONLY。下一先实际四控制，再据资源和结果决定八组求解，不提前声明通过。

## 首轮服务器控制失败及新定位

源码发布`03642f0b18b7923ff18aac46dd60e0ff5b923522`已推Q1、独立核远端，两个新文件部署前不存在、部署SHA与发布一致。[原生A控制和QP跟踪](../logs/session_records/supermimic_q1_P15_controls_A_20261025.log)退出1：解析两项通过，toy实际QP发生NoSolutionFound，真实Q1导数检查因“十个foot geom全为sphere”断言失败。实际为每脚一个mesh和四个2mm sphere，旧控制错误假设不能用来证明恢复机制失败或可行。

toy跟踪显示第151次QP才失败：地面配置Z=-0.04时，模型实际仍在Z=0，foot恰好接触错误的平面，引起witness法线退化。新的[只读地面探针](../logs/session_records/supermimic_q1_P15_floor_position_20261025.log)在真实Q1中确认：仅改geom_pos使配置Z=-0.04但data.geom_xpos仍0、geom_sameframe=1；将sameframe显式置mjSAMEFRAME_NONE后实际Z才为-0.04。MuJoCo3.3.7最小控制还确认mj_setConst不会自动清这个标记。该问题同样存在于P12C/P14的运行时floor位置改写；其保存数组及失败结果仍是真实结果，不能重写为按预期floor求解成功。

下一只在新P15入口处理floor优化标记并显式核实际地面位置，按真实mesh+sphere处理足底导数。保持P12C/P14文件与结果字节；修正比较会包含地面坐标和恢复界两个变化，不能再称单变量因果对照。位置、速度、根/脚目标及原准入门不放宽。A失败完整保留，正式对照尚未运行。

## 训练存储停止

08:05:45 R02D共享盘低于100GiB保护，storage_guard终态STOPPED_STORAGE，controller/train为STOPPED_SIGNAL；新增5282/7000轮、338048转移、累计10282轮，未完成最终配对。保存final.pt205316947字节/SHA`8749a520441374b4c5818529090b9e33fa07ae70d6d304237bb5d3ef0a053d01`。原训练停止记录和下一步见[R02D](README_SUPERMIMIC_Q1_R02D_BOUNDED_20261025.md)，不自动恢复或改写终态。

后续df现场共享盘已余约73GiB，正式P15的100GiB保护不满足，因此不启动正式八组；只继续无大文件输出的内存控制，并已询问用户可用存储资源。大工件仍原地LOCAL_ONLY，不删除未知资产。

## B修正准备

仅修改未被正式数组收据绑定的新P15入口及控制；A版本保留于发布03642f0和上述原生失败日志。新增configure_floor显式清为mjSAMEFRAME_NONE，实际求解每个QP前都核data.geom_xpos与world geom_pos匹配，避免静默使用错误地面。修正后同时改变实际floor坐标和恢复界，明确两因素，不宣称单变量因果。

B新增第五控制，重现地面缓存不一致/要求显式拒绝，再验证正负地面位移实际生效；toy原八条件QP恢复不减少。真实Q1按两个mesh和八个sphere的实际形状计算独立支撑点高度，对有限激活行核有符号导数；mesh存在多个共面支撑点时，原生法线Jacobian的方向值须在独立正/反向差分的区间内，不能用错误全sphere前提或跳过mesh来通过。原16状态、每状态8方向保持。B本地两解析控制和语法检查后发布，服务器五控制待。
