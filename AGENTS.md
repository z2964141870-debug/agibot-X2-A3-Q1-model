# HumanPlus Sonic X2 Agent Rules

## Read order

Before changing control code or touching the robot, read:

For the live keyboard-walk task (first walk achieved 2026-10-23), start with
`script/project_tools/x2_field_deploy/FIELD_CARD.md` (v3), then
`script/project_tools/x2_field_deploy/x2_end_to_end_runbook.md` and
`reports/X2_MISSING_JOINT_TOPICS_20261023.md` (closed).
These supersede all earlier live-state, model/tuning, and stage claims below.
Do not add runtime thresholds stricter than the pinned upstream Sonic-X2
configuration; the extra 100 ms runtime freshness abort was removed at the
user's request. This does not remove upstream protections, physical E-stop,
single-writer ownership, or the prohibition on killing a loaded writer.

Then read repository governance and the historical safety/experience sources:

1. `README.md`
2. `reports/README.md`
3. `script/sonic_x2_transfer_v2/GR00T_audit/AGENTS.md`
4. `script/sonic_x2_transfer_v2/GR00T_audit/docs/CURRENT_STATE.md`
5. `script/sonic_x2_transfer_v2/GR00T_audit/docs/DECISIONS.md`
6. `script/sonic_x2_transfer_v2/GR00T_audit/docs/OPERATIONS.md`
7. `script/sonic_x2_transfer_v2/GR00T_audit/docs/EXPERIMENTS.md`

The nested `AGENTS.md` contains the real-hardware safety contract and remains
binding.

## Project hygiene

- Keep all project content under `data/`, `logs/`, `script/`, or `reports/`.
- Commit code, compact text logs, manifests, and Markdown to GitHub.
- Put weights, raw telemetry, rosbag, videos, archives, and large datasets in
  Baidu Netdisk. Commit their byte size, SHA-256, remote path, and status.
- Never commit passwords, OAuth tokens, cookies, private keys, or local
  credential configuration.
- Do not delete a local large artifact until remote byte size and a downloaded
  SHA-256 have both been verified.
- Preserve failed experiment logs. Summarize evidence narrowly.
- The live deployment workspace is intentionally dirty. Do not reset, clean,
  rename, or reformat it while taking snapshots into this branch.
- After each completed stage, update the relevant README or experiment ledger
  immediately.

## v0.2 keyboard-walking reporting

For this task, follow `reports/SONIC_V02_WALK_PROGRESS.md`. Execute only the
stage explicitly assigned by the user. After each stage, submit its evidence,
record the actual ending state, and wait for user review and the next task;
do not self-approve or advance through the roadmap automatically. A review
does not replace current physical safety confirmation for powered work.
Waiting for review must never mean abruptly killing a loaded robot's writer
or leaving custom powered control unattended. Follow the verified safe
ending procedure for the assigned stage and report abnormal events promptly.

## Robot boundary

This repository organization does not authorize robot commands. Before any
custom HAL publisher, re-check current MC ownership, prove exactly one writer,
obtain current physical suspension confirmation, and follow the nested
operations guide. Never infer current robot state from this filesystem alone.
