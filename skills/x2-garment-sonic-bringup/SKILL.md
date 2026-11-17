---
name: x2-garment-sonic-bringup
description: >-
  Use when bringing up, validating, troubleshooting, or handing off AgiBot X2
  wearable-garment-to-SONIC whole-body teleoperation (smartwear_v2,
  x2_smpl_wholebody, x2_deploy_onnx_ref), swapping SMPL upstream sources such
  as LFP to FGP, or diagnosing frame, monotonic clock, BLE disconnect, heading
  guard, or onbot launch failures.
---

# X2 Garment -> SONIC Whole-Body Bringup

Runbook and diagnostic workflow distilled from verified read-only (`--no-publish`)
and powered X2 Ultra walk/turn runs. Keeps all host IPs, paths, and BLE MACs
configurable via environment variables rather than hidden defaults.

## Safety & Isolation Rules

1. Always validate any upstream/tokenizer/deploy change in **isolated read-only
   shadow mode** (`--no-publish`, ports `15557/15558/15566/15573/15574/15575`)
   before running powered control on production ports (`5556/5557/5558/5563/5573/5574`).
2. Before powered ignition, confirm two-stage standing on the controller
   (`L2+X` position stand -> settle -> `R2+X` torque stand), slack safety tether,
   wearer standing a half-step behind and outside arm reach, and **3 s max T-Pose**.
3. Enforce **one `stopping MC` takeover per boot cycle**. If a standing-only deploy
   already ran in the current boot, reboot cleanly before starting whole-body.
4. Keep a physical spotter and e-stop ready whenever testing walking/turning with
   heading guard disabled (`--root-heading-band-deg 0`) or after any code/model change.
5. Shut down strictly in reverse dependency order so stale `wb_state` cannot
   perturb `kplanner`: `rm engage` -> stop token service -> stop deploy -> stop garment.

## Standard Workflow

### 1. Verify / Install On-Board Fixes

Run the SHA-verified backup installer and offline suite before live trials:
- Patched production files & diffs: [script/x2_sonic_live/](../../script/x2_sonic_live/README.md)
- Offline verification (coordinate transform, monotonic clock sync, BLE teardown):
  [install_onboard.sh](../../script/x2_sonic_live/install_onboard.sh) and
  [test_onboard_fixes.py](../../script/x2_sonic_live/test_onboard_fixes.py)

### 2. Read-Only Shadow Handoff Audit (Zero Motor Output)

1. Build or verify the persistent diagnostic C++ binary (`--no-publish --cycle-trace`)
   with [build_diag.sh](../../script/x2_sonic_live/build_diag.sh).
2. Start the garment in foreground (`:15573`): [run_garment_fg.sh](../../script/x2_sonic_live/run_garment_fg.sh).
3. Run the isolated handoff capture & audit: [run_shadow_live.sh](../../script/x2_sonic_live/run_shadow_live.sh).
4. Summarize joint targets, planner slot provenance (`upstream` vs `hold`/`blend`/`cold_idle`),
   and arm tracking via [summarize_capture.py](../../script/x2_sonic_live/summarize_capture.py).

**Gate before powering**: confirmed `pose -> token -> pose` transitions, `100%`
`upstream` provenance around planner phases, expected ipsilateral arm tracking,
and clean timeout release when `engage` is removed.

### 3. Powered 3-Terminal Whole-Body Operation

Use the three foreground scripts in `script/x2_sonic_live/`:
1. **Terminal 1 (Garment)**: [term1_garment.sh](../../script/x2_sonic_live/term1_garment.sh) — connects BLE, triggers 3-2-1 T-Pose on `Enter`, prints live FPS, and closes BlueZ links on `Ctrl-C`.
2. **Terminal 2 (Policy Stack)**: [term2_policy.sh](../../script/x2_sonic_live/term2_policy.sh) — starts/reuses `x2_pose_watchdog` & `pc2_kplanner`, launches `x2_smpl_token_service.py` with `ROOT_HEADING_BAND_DEG=0` (configurable), and invokes `deploy_x2.sh onbot` with explicit `--onbot-*` flags. Pass `--log-dir` when per-joint CSVs are needed.
3. **Terminal 3 (Switch & Monitor)**: [term3_switch.sh](../../script/x2_sonic_live/term3_switch.sh) — prints 1 Hz `pipeline_monitor.py` health and toggles `/home/agi/pico_intent.engage` on `Enter`. Hold 5 s still after engaging before moving.

### 4. Upstream Replacement Gate (e.g., LFP -> FGP)

When swapping the upstream SMPL producer:
- Pass `source_frame="smpl_fk"` explicitly if calling `root_local_joints()` directly (`GarmentIntentAdapter` defaults to `"smpl_fk"`, `root_local_joints()` defaults to `"canonical"`).
- Reject pre-calibration frames by calibration generation/epoch on reset, and keep BLE `close_all()` + `bluetoothctl disconnect` on exit.
- Re-run Step 2 (read-only shadow) under full live `kplanner --ort-gpu` + 50 Hz `deploy` load to verify intent arrival rate and latency before powered trials.

## Known Scope Boundaries (Not Covered Yet)

Do not treat this skill as verified coverage for:
- **Orin FGP live GPU contention**: staged collector parity exists only on offline IMU replay; co-tenancy with `kplanner --ort-gpu` + BLE needs live validation.
- **MorphMimic whole-body replacement**: blocked in offline replay by right-knee lower-limit violations, `-26.72 deg` action-history jumps, waist-pitch/ankle sensitivity, and unverified skeleton normalization.
- **MuJoCo closed-loop replay of high-gain open-loop clamps**: `waist_roll` (`+-25.8 deg`) and `right_ankle_roll` (`-40.1 deg`) clamp hits seen under `--no-publish` have not yet been replayed closed-loop in MuJoCo.

See [HANDOFF_FGP_TO_MORPHMIMIC_20261116.md](../../reports/HANDOFF_FGP_TO_MORPHMIMIC_20261116.md) for the active migration plan.

## References

- Symptom -> root cause -> exact fix table: [symptom_troubleshooting.md](./references/symptom_troubleshooting.md)
- Ports, model triplet, tuning, and env parameterization: [onboard_architecture.md](./references/onboard_architecture.md)
