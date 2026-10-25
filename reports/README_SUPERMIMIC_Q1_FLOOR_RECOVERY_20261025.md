# PHUMA P15足地间隙恢复

记录日期：2026-10-25，北京时间。阶段P15。状态：新适配器和控制实现、本地两项解析控制完成；服务器真实控制、八组对照及独立复核待。

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
