# Q1相同执行前缀误差诊断

记录日期：2026-10-23，北京时间。阶段：SM-Q1-D03。状态：独立离线代码与3项本地控制通过；真实轨迹分析待执行，R02B仍训练中。

## 目标与范围

R02A最终走路只执行700/1652步；其0.346602rad关节误差来自42.373%前缀，而初始0.381450rad来自整段。两者直接相减不能证明同一段动作的改善。本阶段在独立`script/supermimic_q1_analysis/`加入相同执行前缀重算，不改R02冻结核心或E03续训源码集合，不推进物理、策略推理或优化器。

## 完成内容与决定

`compare_prefix.py`要求每份完整片段评测都有匹配的成功独立审计，逐项核对receipt、reference、trace的大小和SHA。比较集合必须覆盖相同片段、引用同一参考，验证50Hz/29qpos/57状态、执行步数与整片分母、有限值和关节RMSE。所有比较策略取共同实际执行前缀，使用float64重算全身/腿部/手臂关节RMSE和水平根位置RMSE，同时保留各自原执行时长、误差与失败标记。

该诊断不增加或放宽0.15rad等可用性门；共同前缀不是完整动作成功证明。后续选父臂仍先看完整率/失败和完整轨迹质量，再看同一时间段误差；出拳仍为已复用开发片段。

## 版本、命令与证据

代码副本`/tmp/yuanqi-supermimic-q1-20261023`，分支`Q1`。运行平台`huanyu:/opt/data/private/YuanQi_Q1`，Python`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。当前仅完成本地3项控制，0.007秒/退出0：截断引起的假改善、腿部与水平范数已知答案、审计/轨迹/参考篡改拒绝。命令：

```bash
python3 -m unittest discover -s script/supermimic_q1_analysis -p test_compare_prefix.py -v
```

`git diff 6b43cc0 -- script/supermimic_for_q1`为空，原训练代码不变。发布并核验GitHub Q1后拟部署分析目录，先运行服务器3项CPU控制，再执行下列真实只读分析：

```bash
$PY script/supermimic_q1_analysis/compare_prefix.py \
  --evaluation A_initial data/supermimic_for_q1/R02_cachefixed_pair_20261023/R02A_batch32_eval_initial/receipt.json data/supermimic_for_q1/R02_cachefixed_pair_20261023/R02A_batch32_eval_initial_audit.json \
  --evaluation A_final data/supermimic_for_q1/R02_cachefixed_pair_20261023/R02A_batch32_eval_final/receipt.json data/supermimic_for_q1/R02_cachefixed_pair_20261023/R02A_batch32_eval_final_audit.json \
  --output data/supermimic_for_q1/D03_R02A_prefix_20261023.json
```

命令尚未执行，真实原生退出码、结果和输入/输出身份随后补记；不把fixture当Q1效果。

## 结果与边界

当前支持可审计的相同前缀比较，尚未产生真实D03结果。22:21:24独立只读核验确认R02原controller751815/start3604769075与B756622/start3605356394、boot d994e71f-bae0-46ba-a5f7-6c2163771c6c匹配，冻结源码/输入/运行时通过。B1900轮/30400交互/709组WM/659次actor；GPU40摄氏度、容器31007768576字节、OOM仍14，B/pair无终态。本次连接使用既有Windows跳板临时SSH，认证不写文件。

## 保存与下一步

代码及本记录按Q1分支同步，具体发布提交由Git历史与随后实际结果给出；尚未声明部署或真实分析完成。大型模型与轨迹留服务器LOCAL_ONLY。先完成上述D03真实分析；B完成原有训练/评测/重算后，使用四份初始/最终receipt作匹配前缀比较，结合原comparison决定独立长预算，R03尚未登记。
