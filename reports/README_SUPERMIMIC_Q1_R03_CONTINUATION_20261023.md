# Q1从B128追加固定训练预算

记录日期：2026-10-23，北京时间。阶段：SM-Q1-R03。状态：父臂及固定预算已选择并记录；尚未登记或启动。

## 目标与范围

继续用户授权的SuperMimic for Q1世界模型路线，检验增加学习预算能否得到实际移动与全身动作跟随。保留[R02完整对照](README_SUPERMIMIC_Q1_BATCH_TRAINING_20261023.md)终态，在独立`data/supermimic_for_q1/R03_B128_extend12000_20261023/`继承第二组最终模型，固定追加12000轮。原hp3090/A3不恢复，不操作机器人。

## 选择及依据

父臂`R02_cachefixed_pair_20261023/R02B_batch128`：三片完整、腿部误差更低，浅蹲和复用开发出拳漂移较小；A走路仅42.373%后失败。B上肢/全身误差较大，走路仍近原地、根水平误差0.629024m，不能称已得到可用策略或batch128在所有指标胜出。D03相同前缀比较及23:07真实父状态/18阶段/比较重算均已核验；单种子与开发数据复用限制保留。

新增预算12000轮/192000新控制转移，累计15000轮/240000训练控制转移，只有有限追加预算，没有自动增量、无限循环或失败恢复。预计本阶段4749组WM/4699次actor，实际以保存审计为准。B原6110.817秒/3000轮，追加预算预计约7至8小时；这是估计，不是到期停止条件或完成证据。

B最后两个在线窗口真实关节误差尚未改善，模型k8误差仍下降；增加预算只是待检验假设，不能排除模型偏差、控制/参考动态可行性等因素。结束后以相同完整动力学评测及独立重算判断，不按在线回报宣布成功。

## 固定配置与初始化

继承父WM/actor/critic、三组Adam动量与计数、return-bin采样统计；保持16环境、batch128、WM64×10、k8/H32、学习率原cosine累计偏移3000、PD50/2、模型25%力矩、`Q1_FOOT_PRIORITY_V2`与相同走路/浅蹲训练素材。初始/final评测仍含出拳复用开发片段。

原检查点没有完整replay/仿真进程状态，本阶段显式新seed20261024、新replay/episode，并保留128采集轮和50组模型预热。step0在新episode重置之前保存父网络/Adam/bin状态；这不是逐步无缝续跑。只把模型内推演反传用于策略学习，不加入教师、行为克隆或A3权重。

## 版本、平台与命令

代码副本`/tmp/yuanqi-supermimic-q1-20261023`，分支`Q1`。服务器`huanyu:/opt/data/private/YuanQi_Q1`，Python`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。核心冻结`6b43cc0`；续训代码发布`d5f9476125f2c3c7ed2d7c1e6ac898e1f69d6429`及61项release已核验，后续D03分析代码位于独立目录，不改变这份发布清单。

父final205221971字节/SHA`b2990bc9af730127e5bda6d88fe1f3b214980e061e3badade555eedd02c51095`；release服务器`logs/supermimic_q1_extension_release_E03_20261023.json`，11363字节/SHA`c141ff85ecde79ac83bcde4edda7fe326e2f22cf5c8cf5932da718eb19386454`。当前主机boot d994e71f-bae0-46ba-a5f7-6c2163771c6c，已重查RTX3090 24576MiB/驱动570.153.02；23:07容器26910662656字节、空闲17788MiB、GPU39摄氏度、OOM14。旧训练无活跃进程；启动仍由登记入口重查输入、运行时、源码和资源。

以下命令待本记录发布后执行一次，尚未执行；不能重复现有输出：

```bash
$PY script/supermimic_q1_extension/pipeline.py register \
  --root "$PWD" --release logs/supermimic_q1_extension_release_E03_20261023.json \
  --pair data/supermimic_for_q1/R02_cachefixed_pair_20261023 \
  --parent-run data/supermimic_for_q1/R02_cachefixed_pair_20261023/R02B_batch128 \
  --output data/supermimic_for_q1/R03_B128_extend12000_20261023 \
  --steps 12000 --seed 20261024 \
  --reason 'B completes all three clips with lower leg error and less stationary-clip drift; extend budget to test learning while preserving the audited control and model contract.'
$PY -u script/supermimic_q1_extension/pipeline.py execute \
  --plan data/supermimic_for_q1/R03_B128_extend12000_20261023/plan.json
```

实际执行使用脱离SSH的`subprocess.Popen`，stdin为DEVNULL、stdout/stderr到该独立实验的controller.log；原生控制器依次执行训练、续训保存审计及step0/final三片整段物理评测/重算/分肢诊断。GPU86摄氏度、可用显存2GiB、容器30GiB、新OOM等保护继续；不改变其他六个Python进程。

## 结果、保存与下一步

当前只是已选择预算/父臂，不存在真实R03计划、进程或结果。启动后补实际plan大小/SHA、boot/PID/start_ticks、继承检查与首轮进度。结束后同时核对0.15rad等既有门、腿部与水平根轨迹、实际/参考位移和画面，不把流程成功或站住当成移动跟随。

代码/计划清单/小证据按Q1分支同步GitHub，先审查diff与payload；大型模型与轨迹仅服务器LOCAL_ONLY。恢复先读本记录与新目录status/started/日志，重新核对进程身份；无自动恢复，不重复注册或启动。
