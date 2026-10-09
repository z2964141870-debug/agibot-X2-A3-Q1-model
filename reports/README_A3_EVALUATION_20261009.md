# A3 模型效果评测

记录日期：2026-10-09，北京时间。实验：E01 MuJoCo step500 对 step2000。状态：两份模型各20动作评测完成，当前MuJoCo稳定动作跟随未通过。

## 实测结论

15:33:43启动评测，两个子进程各约110.5秒正常退出0，service inactive/success。两份输出覆盖相同20动作，无缺失/重复，每份24304 policy steps，各动作两模型的完整长度一致。最终模型的20个动作均触发上游跌倒启发式，不适合据此部署真机或宣称遥操成功。

| 指标 | step500 | step2000 |
| --- | --- | --- |
| 全程未触发跌倒的动作 | 0/20 | 0/20 |
| 首次跌倒时间中位数 | 1.90秒 | 2.40秒 |
| 首次跌倒时间范围 | 1.46–2.24秒 | 1.60–3.96秒 |
| 全程29关节角RMSE，按帧加权 | 0.3151 rad | 0.3148 rad |
| 保护采样CPU最高温度 | 79°C | 82°C |
| 保护采样GPU最高温度 | 37°C | 38°C |

具体例子：前向慢走参考长33.04秒，step500在2.16秒触发跌倒，step2000在1.96秒触发；浅蹲参考长23.10秒，1.90→2.62秒；双手前伸参考长17.08秒，1.90→2.38秒。因此中位时间稍长不代表每个动作改善。单模型、固定起点的确定性评测，不是多种子统计。

以上全程误差包括倒地后的帧，不能据0.3151→0.3148推断有效站立阶段跟踪改善。根位置误差也受倒地/参考继续移动影响。本轮确认“训练内MuJoCo稳定跟随失败”，尚不能确认是训练不足、参考/控制参数不一致或Isaac→MuJoCo差异；需要训练仿真中的同条件评测继续定位。

## 目标与范围

回答当前模型是否能执行参考动作。此前只完成checkpoint完整性和重载检查，以及step2的0.2秒短闭环，尚无最终模型效果评测。

使用固定官方MuJoCo闭链A3模型，比较同一训练血缘的500与2000 checkpoint，全部20个selected20 CSV、相同起始参考姿态、完整动作长度、a3_fast、50Hz策略、120Hz原CSV抽取4再按30Hz解释、零执行延迟。该集合参与训练，只能作为训练内sim2sim检查，不能代表未见动作泛化或真机效果。确定性Actor输出，不进行策略更新；不操作真机。

## 方法与关键决定

复用官方 `gear_sonic/scripts/sim2sim_a3_mujoco.py`，不修改上游控制链。`script/a3/evaluate_mujoco.py`串行运行两份模型，登记CSV/模型/配置hash与精确命令；复用已检查的温度监督函数。用户服务CPUQuota100%、MemoryMax10G、最长15分钟，CPU90/GPU85连续两次触发停止；不自动重试，不改变原R01预算或已暂停的巡检。

先检查完整动作数量、是否跌倒及首次跌倒时间，再检查关节和身体跟踪误差。上游MuJoCo跌倒启发式为根高度低于0.45m或roll/pitch超过60度；是该工具的失败判据，不能与Isaac的eval termination成功率混用。MuJoCo继续模拟跌倒后的帧，其全程误差可能包含倒地阶段，必须注明。

## 版本、命令与产物

主项目hp3090 `/media/yu/FAFF-E9771/YUANQI`，main起点 `06995ce9b827e0bd97bc1e2b0c77af90ea569590`，官方vendor `fe6868ba37034f89b912f0fb851bce19f120266d` 保持。新增包装脚本已通过py_compile和本次实际两模型/40动作覆盖核验，复用温度保护且未触发。本轮boot保持 `91918243-d1e8-4a3b-a49b-85b19b112134`，无新重启证据。

```bash
cd /media/yu/FAFF-E9771/YUANQI
systemd-run --user --unit=yuanqi-a3-mujoco-eval-20261009-e01 \
--property=WorkingDirectory=/media/yu/FAFF-E9771/YUANQI \
--property=CPUQuota=100% --property=MemoryMax=10G \
--property=RuntimeMaxSec=900 --property=KillMode=control-group \
--setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
/media/yu/FAFF-E9771/YUANQI/data/environments/a3-sonic/bin/python \
-m script.a3.evaluate_mujoco \
--output data/evaluation/a3_20261009/E01_mujoco_step500_vs2000 \
--logs logs/a3_evaluation_20261009/E01_mujoco_step500_vs2000
```

包装拒绝复用既有目录，复跑须改实验编号与unit。已有结果可用相同模块加 `--summarize-only` 重新聚合，该操作不启动仿真。模型分别在E005的step500和E006_auto_03_s1800的step2000，SHA分别为 `bfeecec8d4107f98d9a6676245da3d52f7ab58f294d8ff5c484ab85fbe74cad0` 与 `c19f3e760245a611918dd91552ef181ba07cb6d00962d22a84dd4557f8ac41fc`。各CSV与config哈希见manifest。

数据产物 `data/evaluation/a3_20261009/E01_mujoco_step500_vs2000/`：manifest、两份metrics和summary.json。Git保存小型逐动作摘要 `data/manifests/a3_mujoco_E01_20261009.json`；终端/温度日志 `logs/a3_evaluation_20261009/E01_mujoco_step500_vs2000/{step500,step2000}/`，聚合日志为上级summary.log。模型和大数据仍LOCAL_ONLY，不复制到Mac或Git。

## 结果、同步与下一步

仿真程序运行成功，策略效果失败，二者分别记录。报告、包装脚本与小型摘要按scoped审计后推送main并核验远端；实际提交号见Git历史与publication.log。没有启动新训练、解除R01预算或启用旧巡检。

若MuJoCo表现差，还需区分策略本身和Isaac→MuJoCo差异：之后同条件Isaac评测再定位，不能单靠这次结果认定训练算法或机器人资产有错误。本轮不启动新训练。
