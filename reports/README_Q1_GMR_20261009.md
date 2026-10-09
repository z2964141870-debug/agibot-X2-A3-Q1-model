# Q1 官方 GMR 求解器对照与新方法选择

日期：2026-10-09。实验 E003（不限帧速度）/E004（2 rad/s 帧总位移约束）。状态：四段 AMASS 的两组 GMR 运动学实验完成；Q1 配置仍需改善，未验证全身平衡。

## 目标与范围

实际接入 GMR，判断当前 Q1 幅度压缩是否可通过另一种映射改善；用户同时允许比较 NMR 或其他新重定向方法。只做离线运动学及满足条件的固定骨盆 PD；没有策略训练或真机控制。新大型模型/数据未下载。

## 实现与来源

- 直接执行 YanjieZe/GMR 的 `GeneralMotionRetargeting`，固定公开 master 提交 `bb1bbe40774794fceb2a7c579a3464a28e68c844`；官方 `motion_retarget.py` / `params.py` / MIT LICENSE 原文件及 SHA 已核验，放在被忽略的 `data/third_party/gmr_bb1bbe407747/`，没有把第三方资产纳入 Git。
- 新 `run_gmr_comparison.py` 为 Q1 注册现有 22DOF MJCF 和实验配置；原人体输入采用同一 AMASS 身体参考，按腿长统一缩放；用合成标准站姿计算各目标的局部位置/姿态偏移。未声称这是官方 Q1 配置：公开 GMR 没有现成 Q1 表，本轮只用一个 match table、位置/姿态权重 20/1，尚未调优多体段尺度与两级目标。
- 官方 GMR 的调用方式把 limits 作为第六个位置参数；安装的 Mink 1.2 第六项是 safety_break。通过独立适配函数将 limits 显式传入关键字，官方源码不改，避免静默丢失限位。求解器使用已安装 quadprog。
- GMR 先用源首帧做 30 次初始姿态拟合，再运行 50 Hz 动作；原自写方法从机器人中立位启动。初始化方式、自由根部处理、目标偏移与解算迭代均不同，不能当严格单变量算法排名。
- E003 使用官方逐帧迭代，不限制帧间总关节位移；E004 将每帧关节范围限制到上一输出 ±2 rad/s×0.02 s，并与配置的 0.01 rad 限位余量相交。不是逐次 IK 速度限位的重复累加。

## 结果

| AMASS 样本 | 不限速 IK P95 | 不限速峰值速度 | 2 rad/s 组 IK P95 | 2 rad/s 组右膝几何范围 |
| --- | --- | --- | --- | --- |
| 站立 | 4.85 cm | 8.00 rad/s | 4.85 cm | 3.61° |
| 走路转向 | 5.30 cm | 14.15 rad/s | 5.23 cm | 32.27° |
| 前踢 | 12.55 cm | 23.66 rad/s | 12.55 cm | 40.49° |
| 摆臂 | 3.23 cm | 5.83 rad/s | 3.24 cm | 0.16° |

8/8 运动学处理完成，MJCF 原始关节范围违例均 0。不限速的前踢右膝范围 93.66°，但源几何范围 86.69°，且峰值速度 23.66 rad/s：范围更大不等于姿态准确或可执行。限速后范围 40.49°，当前自写合成站姿实验为 63.52°，但目标和初始化不同，不能宣称任何一种通用算法更差。

E004 走路/前踢通过精确参考合同并完成固定骨盆 PD，最大单关节 RMSE 为 0.0531 / 0.1977 rad；从已拟合第一帧关节开始，不能与原中立启动 PD 直接排名。站立/摆臂首次被严格余量比较拒绝，后续独立核验最大误差仅 3.47e-18 / 8.67e-18 rad。E005 用 `validate_gmr_reference.py` 在显式 1e-8 rad 预算内裁正尾差，实质越界仍拒绝；两段新回放通过，RMSE 0.0059 / 0.0105 rad，另外两段 SHA 核验后复用已有回放。四段均有合规参考与支撑回放证据，原始 E004 保留。不把程序完成等同保真通过。

两张 GMR mesh 抽帧图已生成，前踢图已视觉核验；沿用旧手腕峰值抽帧，不能代表膝峰值或完整步态。当前没有足底接触/穿地/打滑、自碰撞及无支撑跟踪验收。

## 版本、命令与产物

主机 `hp3090`，项目 `/media/yu/FAFF-E977/YuanQi_Q1`，分支 `Q1`。校准修复提交 `946e71f9bfcd3b6874e87aa11702b2eabbc90ccb` 已推送并核验远端；本阶段驱动/第三方源码精确 SHA 在每份 summary 中。E003 的原驱动在实验目录 `run_gmr_comparison_snapshot.py` 保留，避免 E004 新功能覆盖其复现版本。

```bash
ssh hp3090
cd /media/yu/FAFF-E977/YuanQi_Q1
PY=/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" script/q1/run_gmr_comparison.py \
  --output-dir data/experiments/q1_gmr_20261009_E004 --frame-speed-limit 2 --render
```

已有目录拒绝覆盖；复跑使用新实验目录。去掉 `--frame-speed-limit 2` 是不限速实验，不用于当前支撑参考合同。

- 工件：`data/experiments/q1_gmr_20261009_E003/`、`data/experiments/q1_gmr_20261009_E004/`、`data/experiments/q1_gmr_20261009_E005/`，含轨迹、Q1 映射表、部分图、summary 与尾差核验。
- Mac 小摘要：`data/sync/q1_gmr_20261009/`；公开清单 `data/manifests/q1_gmr_20261009.json` 登记两组实际结果、代码与来源。
- 小终端记录 `logs/session_records/q1_gmr_20261009.log`；第三方抓取遇到 raw.githubusercontent.com 超时，改从 Mac 获取共约 23 KB 源码后迁移。没有重传大型文件。

## 新方法核验与决定

### NMR：已发布推理，但直接输出 G1

官方仓库 `NJU3DV-HumanoidGroup/MakeTrackingEasy` 当前 README：2026-03-24 论文，03-26 HF demo，04 月推理/权重发布；CEPR 配对数据与训练代码的 TODO 仍未完成。公开推理固定 29 关节 G1，读取 140 维人体特征，输出 217 维 G1 特征。按 30 FPS、4 秒片段及重叠推理，默认还有低通处理，不是已验证实时 Q1 接口。

源码 `_ensure_large_files` 会自动下载约 518 MB checkpoint 和约 104 MB SMPL-X 模型；本轮未执行它，服务器已有合法来源身体模型。推理入口当前不使用输入 betas，`load_smpl_data` 返回形体值 None，并采用固定配置形体；不能据此当形体保持基准。Q1 需要自己的解码器/配对数据和训练实现，G1→Q1 再转换会重新引入本轮映射问题。最初检索只看到论文/项目页，后续已确认公开仓库，结论以此处核验为准。

### UMR：更适合优先尝试 Q1 的新方法

`hanyang9/UMR`，论文 `2609.02134`，2026 年 9 月；官方仓库 MIT。先在机器人与人体 canonical pose 上学习点云表面对应，再用该对应优化动作、接触与运动学约束。官方新机器人流程提供 MJCF + T-pose 配置，不依赖固定 G1 解码器，因此是 Q1 下一优先实验的依据，效果尚未验证。

官方默认 4096 个表面点、500 epochs 对应学习；这是表面对齐训练，不是控制 policy。支持 SMPL-X、BONES/SOMA、HiPHI 等，身体模型不随仓库分发。当前训练环境已有 torch/warp/trimesh，未找到 embreex/cholespy/clarabel/coacd/obj2mjcf/soma；新增环境和 Q1 T-pose/外表面采样需先验证。源码 README 推荐独立 Python 3.12/torch 2.4.1，不能直接覆盖现有训练环境。

### PHUMA / PhySINK：作为物理筛选对照

官方提供自定义 XML/URDF 与 SMPL+Sim 输入，含交互约束/地面和限速处理；原 Train/Test 人体 pose 因许可不发布，Video SMPL 仍待发布。Q1 支持需建立自己的映射和形体配置。它是后续物理可执行性路线之一，本轮未复现。

原始来源（2026-10-09 文档/源码核验）：

- GMR：https://github.com/YanjieZe/GMR
- NMR：https://github.com/NJU3DV-HumanoidGroup/MakeTrackingEasy ，https://arxiv.org/abs/2603.22201
- UMR：https://github.com/hanyang9/UMR ，https://arxiv.org/abs/2609.02134
- PHUMA：https://github.com/DAVIAN-Robotics/PHUMA

## 保存与下一步

本阶段报告与实验记录已保存；大型工件仅服务器 `LOCAL_ONLY`，全部 `TEST_ONLY`。下一步准备独立 UMR 小样本适配；没有将 GMR/NMR/UMR 标为 Q1 可部署方案。审核后仅本阶段路径提交到 Q1 并核验远端，既有 PredActor 暂存内容保留。
