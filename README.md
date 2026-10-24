# HumanPlus Sonic X2 (branch: x2)

AgiBot X2 Ultra 上的 sonic_x2_v0.2 部署。本分支 = X2 侧唯一活跃入口。

## 当前状态：✅ 键盘行走已通（2026-10-23）

两轮完整点火（preflight → em stop_app → ~20ms 接管 → CONTROL 稳定），
18:0x 键盘 `w` 前进 0.35 m/s 实走成功。

## 必读入口

| 文档 | 用途 |
|---|---|
| `script/project_tools/x2_field_deploy/FIELD_CARD.md` | **现场任务书 v3**（核心纪律 + 点火流程）|
| `script/project_tools/x2_field_deploy/x2_end_to_end_runbook.md` | 全流程终端命令单（2 终端）|
| `reports/X2_MISSING_JOINT_TOPICS_20261023.md` | joint 话题可见性排查（已结案，含三层根因）|

## 目录结构

```
deploy_ws/    部署 C++ 源码（x2_deploy_onnx_ref: SMPL 输入源、状态机、preflight）
script/
  project_tools/x2_field_deploy/   现场工具箱（FIELD_CARD、键盘、runbook）
  project_tools/                   分析与实验工具
  sonic_x2_transfer_v2/            传输栈（gear_sonic_deploy: deploy_x2.sh 等）
  smartwear_runtime/               SMPL-X 转换（whole-body teleop 备用）
config/       运行配置
data/         工件台账与证据（manifests / evidence）
reports/      现行报告（2026-10-23 之后）
```

## 下一步

SMPL / whole-body teleop：token 模型与 SMPL tokenizer 已在机器人 `policies/`，
deploy 内置 SMPL 输入源（`smpl_intent_source`），上游文档
`logs/x2_round2_prep_20261020/upstream_docs/F03_pico_teleop.md`。

## 历史说明

2026-08 ~ 10-21 的交接/状态/计划文档已于 2026-10-24 清理（本分支此前提交中仍可追溯）。
