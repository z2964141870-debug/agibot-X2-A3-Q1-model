# Q1 UMR 小样本接入实验

日期：2026-10-09。实验 E006/E007（初始准备失败保留）/E008/E009（接入与独立评测完成）。状态：两组 UMR 32 帧转换完成；E009 速度门限通过，但视觉网格穿地与动作保真未通过，尚不能用于全身控制。

## 目标与范围

用户允许尝试 GMR 之外的新重定向模型。本阶段检查 UMR 是否能接入现有 Q1 MJCF 与服务器已有 SMPL-X 数据。只尝试 256 表面点、50 epochs 的人体/机器人表面对应学习与不超过 32 帧的短动作重定向，不是强化学习控制 policy，也不是完整质量验收。没有真机操作或新大型动作/身体模型下载。

## 已完成准备

- [官方仓库](https://github.com/hanyang9/UMR) 已以 sparse clone 放在 `data/third_party/umr_q1_20261009`；固定版本 `d1cdaedaf67a827198da71c5fa831f48b2e51f91`，MIT LICENSE 已恢复且 SHA-256 为 `7976e004a50466e0ac0c03dac800ac78dd3d4aa483f0545a04e98deb9150b8ee`。必要 root/scripts/robot_configs/分区文件已实际核验，上游工作区干净；未取样本全集。一次 `ls-tree --long assets` 意外触发 partial clone 的 mesh 大小查询下载，发现后已停止；后续只查询明确文件、避免递归带 `--long`。这些 Git 对象仅留服务器，未迁往 Mac。
- 服务器当前 RTX 3090 24GB 可见，启动前约 96 MiB 占用、0% GPU 利用率；这只是本次运行前状态，不当永久状态。
- 在 `data/environments/umr_q1_overlay` 安装官方需求中的缺失小包：clarabel 0.11.1、cholespy 2.2.0、embreex 4.4.0、rtree 1.4.1，使用 `--no-deps --target`。未覆盖现有 x2-sonic-isaaclab 环境，也未重装 torch。约 22 MB wheel 下载直接在服务器完成，速度正常。运行时通过独立 PYTHONPATH 覆盖层加载。
- `prepare_umr_q1.py` 用 scipy 最小二乘与实际 Q1 FK/限位拟合双臂水平 T-pose，输出图、具名关节配置及身份清单；使用现有 MJCF、现有身体模型和 AMASS 前踢，不使用 G1 canonical pose。
- `run_umr_pilot.py` 逐阶段保存 stdout/stderr 和状态，build/train/retarget 超时分别 180/360/240 秒，任一步失败即停。默认不打开交互 viewer。

## 初始失败与恢复

E006/E007 已生成部分 T-pose/config，但 sparse checkout 的实际 pattern 为字面 `--cone`、`scripts`、`robot_configs`，根部 LICENSE/defaults 仍被排除，准备清单记录失败；旧目录保留。2026-10-09 14:27 恢复核验：E007 runner 同样在读取 LICENSE 时退出，未开始 build/train/retarget。此前“准备完成、运行中”已更正。较早的 build 导入曾缺 clarabel；四个小包已安装到覆盖层。下一次准备先检查必要文件，再建立独立 E008 输出目录。

E008 恢复成功后，T-pose 最大端点偏差 0.043881 m、关节越界 0；小图已检查，双臂仍有弯曲，属于近似 T-pose。attempt1 因根部 `view_smpl_mujoco.py` 缺失退出。恢复后 attempt2 实际 build/train 分别约 4 秒完成：SMPL-X/Q1 两表面样本、256 点，50 轮 loss 0.002030→0.000773。该 loss 仅对应学习目标，不证明动作或接触质量。retarget 读取前踢、betas/gender 并生成 32 帧人体表面后，因 `assets/smplx_parts_segm.pkl` 缺失退出。已保存旧日志和工件，继续仅补齐必要小型资产。

恢复 1,323,168 字节分区文件后，attempt3 复用已有采样/对应模型，实际动作转换约 6.6 秒完成，输出 `(32,29)` qpos。独立检查读取原始 120 Hz、stride4，输出 30 Hz、1.033333 秒，frame_ids 精确匹配；浮动根四元数/数值有效，导出 XML 与原 Q1 逐帧 FK 完全一致。原始限位越界 0，配置余量最大尾差 `6.68e-10 rad`。UMR 同采样帧人体右膝范围 84.43°、Q1 75.20°，但峰速 11.9866 rad/s，全部视觉网格最低 Z=-0.029799 m，现有 2 rad/s PD 参考门限失败。没有运行支撑 PD 或无支撑仿真。

源码核验还发现 UMR 固定使用前 10 个 betas，原 stageii 是 16 个；本次官方 pkl 重建与既有 16-beta npz 重建的关节最大绝对差 0.011788 m。因此不作为严格保形体的算法排名，也不与完整片段、2 rad/s 的 GMR 数字直接比较。256点未覆盖全部网格，采样约束与 LQR 后处理都不能代替全网格穿地检查。对应学习约 46 MB 工件留服务器，无 Mac 大文件迁移。

## E009 速度约束对照

保持同一源动作、同一32帧、256点、50epochs与随机种子；启用上游 `robot.dof_max_dq_box`，每个关节每次步长不超过 `2/30*(1-1e-5)=0.066666 rad`，移动帧仅迭代1次，首帧拟合15次。关闭轨迹 LQR 后处理以保持输出步长约束。因此主要变量是加入速度约束，后处理也随之关闭，不用于严格的单项算法排名。2 rad/s 是当前仿真参考配置的保守门限，**不是已核验的 Q1 真机额定速度**。

| 同采样帧指标 | E008 默认 | E009 速度约束 |
| --- | ---: | ---: |
| UMR 人体右膝几何范围 | 84.43° | 84.43° |
| Q1 右膝几何范围 | 75.20° | 41.71° |
| Q1 峰值关节速度 | 11.9866 rad/s | 1.999981 rad/s |
| 原始关节限位违例（>1e-6 rad） | 0 | 0 |
| 配置余量最大尾差 | 6.68e-10 rad | 6.68e-10 rad |
| 全视觉网格最低 Z | -2.98 cm | -2.89 cm |

E009 全部 32 帧仍有视觉网格低于地面超过 1 mm；没有据此推定 collision geometry 相同程度穿透，但全网格验收失败。右膝时间曲线有明显偏差，不能只按幅度比认定高保真。两个实验都是小样本接入，不代表 UMR 完整参数能力；也不能把失败全部归因于 Q1 尺寸。25项现有输入/重定向/仿真契约回归检查通过，日志已保存。GPU 前后空闲、GPU温度31–32°C，CPU所读传感器运行前40°C、结束后42°C；没有完整过程峰值监控或主机稳定性验收。

发布前更正检查清单布尔字段的语义：E009 只通过 2 rad/s 速度和 `1e-8 rad` 转换浮点预算；仍有 `6.68e-10 rad` 余量尾差，原始输出不通过严格 PD 参考限位检查，也未导出/执行参考。清单改为 `within_pd_conversion_numeric_budget` 与 `strict_pd_reference_limits_passed` 分开记录，数值、源输出与哈希未改。以后转换应在独立工件中只规范化预算内尾差、再调用严格校验，不能直接提高容差或掩盖真实越界。

## 版本、复现与产物

主机 `hp3090`，项目 `/media/yu/FAFF-E977/YuanQi_Q1`，Q1 分支。上一阶段 GMR 结果提交 `7960732b31aa4eca088e9de4738361acaa888346` 已推送并核验远端。

```bash
ssh hp3090
cd /media/yu/FAFF-E977/YuanQi_Q1
PY=/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python
"$PY" script/q1/prepare_umr_q1.py --output-dir data/experiments/q1_umr_20261009_E009 --reference-speed-limit 2
"$PY" script/q1/run_umr_pilot.py \
  --config data/experiments/q1_umr_20261009_E009/q1_pilot_config.json \
  --log-dir logs/q1_umr_20261009_E009 --attempt 1
PYTHONPATH="$PWD/data/environments/umr_q1_overlay" "$PY" script/q1/inspect_umr_pilot.py \
  --experiment data/experiments/q1_umr_20261009_E009 --attempt 1 \
  --manifest data/manifests/q1_umr_speed_20261009.json
```

以上是新建时的历史命令；已完成目录/清单不能覆盖。复现实验须换独立目录并相应重建配置。E008 不传 `--reference-speed-limit`。实际运行另设 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`，`--config` / `--log-dir` 使用绝对服务器路径。runner 超时会终止进程组，保留失败记录；尚未对超时注入做专项测试。

- E006/E007 部分准备原件保留；完成产物为 `data/experiments/q1_umr_20261009_E008/`、`data/experiments/q1_umr_20261009_E009/`，各约46MB。
- 实际配置快照：`script/q1/umr_pilot_E008_20261009.json`、`script/q1/umr_pilot_E009_20261009.json`；源码/配置/模型/日志与工件身份见 `data/manifests/q1_umr_20261009.json`、`data/manifests/q1_umr_speed_20261009.json`。
- 子进程日志：`logs/q1_umr_20261009_E008/`、`logs/q1_umr_20261009_E009/`；选定成功/失败/25项回归小日志另存 `logs/session_records/q1_umr_*_20261009.log` 进入 Git。
- 状态/当前阶段/日志：实验根 `pilot_status_attemptN.json`；文件存在不证明当前进程还活着，恢复先读状态和日志并检查进程。
- 全部 `TEST_ONLY`，工件备份 `LOCAL_ONLY`，没有 policy 或硬件验收。

## 保存与下一步

Q1 本阶段作业已结束，无控制 policy、支撑PD或无支撑动力学/真机测试；不据此判断同主机 A3 等其他任务是否运行。成果提交 `7ebf80f63d55f92b8542d5555141fb1e62924e7a` 已核验推送 Q1，回执为 `logs/session_records/q1_retarget_publication_20261009.json`；README、索引与当前状态已更新，既有 PredActor 暂存修改保留。大型工件仅服务器 `LOCAL_ONLY`，没有百度网盘上传证据。

下一步优先定位足底视觉网格/碰撞体/采样点的地面关系，再统一完整16-beta人体重建、参考时间与速度预算；随后扩大点数和片段，在相同目标、时间轴、速度限制及初始化下做 GMR/UMR 对照。当前源快速动作不能在保守门限下按原速度完整复现，可用减速回放隔离速度因素，但减速不等于实时全身遥操。重定向参考通过后才进入无支撑跟踪与控制策略训练。
