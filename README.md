# HumanPlus Sonic X2 (branch: x2)

AgiBot X2 Ultra 上的 sonic_x2_v0.2 部署。本分支 = X2 侧唯一活跃入口。

## 当前状态：✅ LFP 穿戴衣服全身 SMPL 遥操真机行走/转身已通（2026-11-16）

- 在 2026-10-23 键盘行走 + `kplanner` 稳站基线上，完成 **LFP 衣服 → `x2_smpl_wholebody` → SONIC (`s1ft16000`)** 只读切换与带电闭环行走/370°+ 转身/俯仰恢复验证（连续稳定 `247.6 s`，航向误差均值 `5.2°`）。
- 三个板端生产修复（SMPL Y-up→Z-up 坐标、`time.monotonic()` 时钟对齐、退出断 BLE）及 3 终端前台入口见 `script/x2_sonic_live/`。

## 必读入口

| 文档 | 用途 |
|---|---|
| `skills/x2-garment-sonic-bringup/SKILL.md` | **衣服→SONIC 联调/排障/实机 Bringup Skill**（含症状排障表与参数清单）|
| `reports/HANDOFF_FGP_TO_MORPHMIMIC_20261116.md` | **学弟任务转接卡**（Phase 1 FGP 替换 LFP → Phase 2 MorphMimic 替换 SONIC）|
| `reports/README_X2_SONIC_LIVE_HANDOFF_20261116.md` | 2026-11-16 真人穿戴 SONIC 只读切换与三大生产 bug 定位报告 |
| `script/x2_sonic_live/README.md` | 3 终端实机入口、隔离只读影子测试、板端补丁安装脚本 |
| `script/project_tools/x2_field_deploy/FIELD_CARD.md` | **现场任务书 v3**（核心纪律 + 点火流程）|
| `script/project_tools/x2_field_deploy/x2_end_to_end_runbook.md` | 键盘行走全流程终端命令单 |
| `reports/X2_MISSING_JOINT_TOPICS_20261023.md` | joint 话题可见性排查（已结案，含三层根因）|

## 目录结构

```
skills/       独立 Skill 目录（与 data/ 同级：x2-garment-sonic-bringup）
deploy_ws/    部署 C++ 源码（x2_deploy_onnx_ref: SMPL 输入源、状态机、preflight）
script/
  x2_sonic_live/                   衣服→SONIC 3 终端入口、只读诊断、板端补丁与单测
  project_tools/x2_field_deploy/   现场工具箱（FIELD_CARD、键盘、runbook）
  project_tools/                   分析与实验工具
  sonic_x2_transfer_v2/            传输栈（gear_sonic_deploy: deploy_x2.sh 等）
  smartwear_runtime/               SMPL-X 转换（whole-body teleop 备用）
config/       运行配置
data/         工件台账与证据（manifests / evidence）
logs/         关键会话回执（session_records）
reports/      现行报告（2026-10-23 之后）与任务转接卡
```

## 下一步

按 [HANDOFF_FGP_TO_MORPHMIMIC_20261116.md](reports/HANDOFF_FGP_TO_MORPHMIMIC_20261116.md) 推进：
1. Phase 1：FGP 替换 LFP（先走 `script/x2_sonic_live/run_shadow_live.sh` 只读过关，再切真机）
2. Phase 2：MorphMimic 替换 SONIC（先解决右膝下限/单步跳变与左踝越界）

## 历史说明

2026-08 ~ 10-21 的交接/状态/计划文档已于 2026-10-24 清理（本分支此前提交中仍可追溯）。
