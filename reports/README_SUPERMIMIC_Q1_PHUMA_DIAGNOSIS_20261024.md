# PHUMA首轮保存核查与数据筛选诊断

记录日期：2026-10-24，北京时间。阶段：SM-Q1-PHUMA-D01。状态：第500轮部分保存审计、固定112候选筛选统计和独立重算完成；PHUMA R01训练继续，最终跟随效果未评测。

## 目标与范围

在保持R01已冻结源码、11训练片、4开发片、3测试片和3000轮预算的条件下，确认保存文件确有学习更新，并解释PHUMA首批候选为什么只有18片合格。原A3来源参考和旧权重仍不进入新实验。主线与完整输入身份见[PHUMA阶段](README_SUPERMIMIC_Q1_PHUMA_20261024.md)。

数值保存审计在huanyu `/opt/data/private/YuanQi_Q1` 的独立CPU进程执行，明确禁用CUDA且0新物理/策略调用/optimizer更新；筛选统计在Mac代码副本用标准库读取已逐字节核验的小清单。没有新增数据下载、重定向、仿真或训练作业，当前训练保持原进程。

## 第500轮保存审计

13:17:09完成，原生退出0、3.09秒。读取原始step0和step500，使用CPU mmap；按独立整数2/5更新节奏重算，500轮对应8000控制转移、149组世界模型更新、99次actor更新。

| 保存对象 | 实测Adam步数 | 预期步数 | 有更新的张量/保存张量 |
| --- | ---: | ---: | ---: |
| 世界模型 | 1490 | 1490 | 14/15 |
| actor | 99 | 99 | 20/22 |
| critic | 1980 | 1980 | 6/7 |

15项检查全部通过：step0本地计数及Adam状态为空，三网络参数变化，保存网络/optimizer张量有限，初始/当前manifest与准入一致，PHUMA训练/留出名单一致，随机初始化/发布版本一致，本地更新次数/20条日志完整，前后文件SHA一致，121冻结文件未变，无新OOM且未初始化CUDA。该项要求三网络均有真实参数变化，不以每个保存张量都变化作为判据。

step0为68,435,755字节/SHA `9e4d66cb26a1eab515745715f850bc93edb445bec53becba7dee978c6270abfc`；step500为205,233,191字节/SHA `bffc42838072fc6920c5c1acf32bc44afe348eebb449103c5260920e4a4437c7`，均留服务器。采样cgroup峰值31,219,302,400字节，低于原32,212,254,720字节保护线；OOM仍14。退出前只对两份本轮权重给只读缓存建议，不修改文件或全局缓存。完整预算审计/最终评测仍由原流水线执行；部分保存通过不等于策略可用。

结构化执行结果：[保存审计](../data/manifests/supermimic_q1_phuma_R01_checkpoint500_audit_20261024.json)、[规范化stdout记录](../logs/session_records/supermimic_q1_phuma_R01_checkpoint500_audit_20261024.log)、[实际执行源码](../logs/session_records/supermimic_q1_phuma_R01_checkpoint500_source_20261024.txt)。保存结果为本次远程stdout的结构化留档，不冒称已有最终审计收据。

## 固定候选的拒绝原因

新只读汇总入口 `script/supermimic_q1_analysis/analyze_phuma_screen.py` 校验输入manifest SHA/大小与独立检查绑定、112唯一来源组、逐片19布尔门和资格/划分一致性，再按split/动作类别汇总。独立第二种逐列计数与筛选行计数重算19门、12组、合格/拒绝名单及总数，全部一致。没有重新计算FK或动力学，不把统计复核叫作第二次物理验证。

| 未通过项 | 片数 |
| --- | ---: |
| 至少90%帧足底距地不超15mm | 85 |
| 支撑脚最大间隙不超30mm | 47 |
| 自碰撞穿透不超3mm | 44 |
| 转换器初门 | 30 |
| 身体位置残差P95不超5cm | 10 |
| 动作输出范围覆盖 | 3 |
| 固定落地平移范围 | 1 |

失败项重叠，不能相加当作拒绝总数；实际合格18、拒绝94。18片仅支撑比例失败，8片仅自碰撞失败，1片仅动作范围失败。仅一项失败不代表误差很小，也不能据此放行。训练移动候选32片仅2片合格，训练下蹲16片仅1片合格；development移动1/6，test移动0/6，因此当前结果仍不能证明独立移动测试泛化。

例如训练片 `train_locomotion_fe9abef3042c3c39` 参考水平位移1.26455m，位置残差P95为11.26mm、最大支撑间隙27.76mm、无自碰撞，但支撑比例仅42.28%，仍拒绝；`train_locomotion_6cafc84ec90e5186` 支撑比例100%、位移1.68903m，但自碰撞4.00mm超过3mm，也仍拒绝。这些是输入几何证据，不是学习后表现。

## 决定与边界

继续本轮PHUMA-only训练，不中途改变质量门、参考帧、时间倍率、控制参数或权重。数据规模偏小的直接瓶颈是Q1重定向后的支撑/碰撞合格率；该结论不证明这些因素就是当前策略失败的唯一原因。后续扩大候选时保持来源组隔离与预先固定质量门，重定向方案的选择先用train/development，test不用于挑选策略或参数。旧D05B根高逐帧投影已有负证据，不作为默认修复。

13:21:05实时核查：controller799647/worker799650与原boot/start_ticks及父子进程组身份一致，875/3000轮、14000控制转移、299组WM/249次actor，GPU40摄氏度/OOM14，cgroup30,845,796,352字节。完整原生读取见[进度记录](../logs/session_records/supermimic_q1_phuma_R01_progress_20261024_132105.log)。这些是带时间的快照，恢复须重核，不重新启动原流水线。

## 版本、复现与保存

训练发布为 `ed4217384e7abfc8a9cea0b23b95592d4a3e19b0`，本次只新增冻结集合之外的分析入口与小记录。保存审计源码记录可在服务器原Python环境以 `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY - < logs/session_records/supermimic_q1_phuma_R01_checkpoint500_source_20261024.txt` 重现；已完成，不需重复执行。

筛选统计实际命令：

```bash
python3 script/supermimic_q1_analysis/analyze_phuma_screen.py \
  --manifest data/manifests/supermimic_q1_phuma_P03_references_20261024.json \
  --checks data/manifests/supermimic_q1_phuma_P03_reference_check_20261024.json \
  --output data/manifests/supermimic_q1_phuma_P03_screen_diagnosis_20261024.json
```

输出用排他创建保存，已存在时拒绝覆盖。[统计清单](../data/manifests/supermimic_q1_phuma_P03_screen_diagnosis_20261024.json)、[独立重算](../logs/session_records/supermimic_q1_phuma_P03_screen_reconciliation_20261024.log)及[独立重算源码](../logs/session_records/supermimic_q1_phuma_P03_screen_reconciliation_source_20261024.txt)保存到本项目，随本阶段提交并核验Q1远端；文档不预写尚未执行的推送成功。大型权重/数据仍LOCAL_ONLY，未声称网盘备份。

下一步读取原controller/run状态等待3000轮，随后核查原流水线的保存审计、18片初始/最终完整动力学评测与独立重算；用真实根位移和全身误差决定下一独立实验。当前尚无final、完整配对成绩或真机结果，长期目标继续。
