# Q1相同执行前缀误差诊断

记录日期：2026-10-23，北京时间。阶段：SM-Q1-D03。状态：独立离线代码、两端3项控制与A真实配对前缀分析完成；R02B仍训练中，四份配对分析待其原流程完成。

## 目标与范围

R02A最终走路只执行700/1652步；其0.346602rad关节误差来自42.373%前缀，而初始0.381450rad来自整段。两者直接相减不能证明同一段动作的改善。本阶段在独立`script/supermimic_q1_analysis/`加入相同执行前缀重算，不改R02冻结核心或E03续训源码集合，不推进物理、策略推理或优化器。

## 完成内容与决定

`compare_prefix.py`要求每份完整片段评测都有匹配的成功独立审计，逐项核对receipt、reference、trace的大小和SHA。比较集合必须覆盖相同片段、引用同一参考，验证50Hz/29qpos/57状态、执行步数与整片分母、有限值和关节RMSE。所有比较策略取共同实际执行前缀，使用float64重算全身/腿部/手臂关节RMSE和水平根位置RMSE，同时保留各自原执行时长、误差与失败标记。

该诊断不增加或放宽0.15rad等可用性门；共同前缀不是完整动作成功证明。后续选父臂仍先看完整率/失败和完整轨迹质量，再看同一时间段误差；出拳仍为已复用开发片段。

## 版本、命令与证据

代码副本`/tmp/yuanqi-supermimic-q1-20261023`，分支`Q1`。运行平台`huanyu:/opt/data/private/YuanQi_Q1`，Python`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。本地3项控制0.007秒/退出0，服务器同3项0.011秒/退出0：截断引起的假改善、腿部与水平范数已知答案、审计/轨迹/参考篡改拒绝。服务器运行先断言`countTestCases()==3`，避免空发现被误计通过。命令：

```bash
python3 -m unittest discover -s script/supermimic_q1_analysis -p test_compare_prefix.py -v
```

代码已发布`2ca9661140dd717163a4f5a7ba4be47f4b11e701`并独立核对GitHub Q1一致。服务器两份源码大小/SHA与发布副本逐项一致；`compare_prefix.py`6054字节/SHA`14f087355c132b7ccffcd7132b5232b9af9b3036bbbe419789643cea20c0d92c`。`git diff 6b43cc0 -- script/supermimic_for_q1`为空，原训练代码不变。下列真实只读分析已原生退出0：

```bash
$PY script/supermimic_q1_analysis/compare_prefix.py \
  --evaluation A_initial data/supermimic_for_q1/R02_cachefixed_pair_20261023/R02A_batch32_eval_initial/receipt.json data/supermimic_for_q1/R02_cachefixed_pair_20261023/R02A_batch32_eval_initial_audit.json \
  --evaluation A_final data/supermimic_for_q1/R02_cachefixed_pair_20261023/R02A_batch32_eval_final/receipt.json data/supermimic_for_q1/R02_cachefixed_pair_20261023/R02A_batch32_eval_final_audit.json \
  --output data/supermimic_for_q1/D03_R02A_prefix_20261023.json
```

输出`data/supermimic_for_q1/D03_R02A_prefix_20261023.json`，9147字节/SHA`a4c11deda34f1b4c10e83d24a70e07c5637d4c37c9b312e2be17c3279585cac9`；Git小副本`data/manifests/supermimic_q1_R02A_prefix_D03_20261023.json`。原始stdout为`logs/supermimic_q1_R02A_prefix_D03_20261023.log`，与JSON输出SHA相同。服务器控制日志`logs/supermimic_q1_prefix_controls_D03_20261023.log`及`logs/session_records/`小副本605字节/SHA`701bc57b272612f2b6865e4fdcfae80c9cfedefff88d0dfa2a117a6aca3fa5d2`。未下载模型/轨迹到Mac，只有小型JSON和控制日志。

## 结果与边界

| 片段与共同时间 | 全身关节RMSE初始→最终rad | 腿部RMSE初始→最终rad | 手臂RMSE初始→最终rad | 水平根RMSE初始→最终m |
| --- | --- | --- | --- | --- |
| 走路，前700/1652步、14.00秒 | 0.374801→0.346602 | 0.216716→0.237951 | 0.561632→0.487865 | 0.628437→0.561301 |
| 浅蹲，1157/1157步、23.14秒 | 0.413148→0.369294 | 0.295426→0.268847 | 0.581675→0.508787 | 0.022062→0.083814 |
| 出拳开发，678/678步、13.56秒 | 0.324217→0.300410 | 0.138062→0.157003 | 0.505014→0.446172 | 0.012564→0.072321 |

同一前14秒的走路全身关节及水平位置误差确有下降，腿部变差；随后最终策略提前失败，原完整率42.373%不变。浅蹲关节误差改善但水平根漂移增大；开发出拳也是手臂改善、腿部和位置误差变差。结果支持继续分肢和位移诊断，不支持称A已可用、仅靠回报/全身均值选父臂，或把失稳唯一归因于预算。

22:21:24初次恢复核验B1900轮，随后22:33:23新证据确认原controller751815/start3604769075与B756622/start3605356394、boot d994e71f-bae0-46ba-a5f7-6c2163771c6c匹配，均非僵尸且命令匹配；冻结源码/输入/运行时通过。B2225轮/35600交互/839组WM/789次actor；GPU40摄氏度、容器31023022080字节、OOM仍14，pair无终态。生成收据`D03_analysis_live_verification_20261023.json`，Git小副本`data/manifests/supermimic_q1_D03_live_20261023.json`，3470字节/SHA`371e24d628233dc040e45189660a39c4a43a7a186a547cb707c3dfd413b14fef`。本阶段0新物理/策略推理/optimizer；训练进程自身持续更新，不把诊断期间的推进算作新实验。连接使用既有Windows跳板临时SSH，认证不写文件。

## 保存与下一步

代码发布和部署已核验，当前补记真实结果；结果记录的具体提交由Git历史给出，推送完成后再独立核对远端。大型模型与轨迹留服务器LOCAL_ONLY。B完成原有训练/评测/重算后，用同脚本传入A/B各自初始与最终四组receipt/audit，以新输出名比较共同前缀，结合原comparison决定独立长预算；R03尚未登记。
