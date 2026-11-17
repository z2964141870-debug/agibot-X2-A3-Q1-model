# On-Board Architecture & Parameterization

## 1. Host & Process Topology (SoC1 + SoC0)

Defaults on the X2 unit documented in this repo (verify before reuse on another robot):
- **SoC1 (`PC2_HOST`, e.g., `10.0.1.41`, user `agi`)**: runs garment receiver, token service, watchdog, `pc2_kplanner_onnx.py`, and `x2_deploy_onnx_ref`.
- **SoC0 (e.g., `10.0.1.40`, user `run`)**: runs `aima-mc-app-main` (MC controller, managed via HTTP `stop_app`/`start_app` by `deploy_x2.sh`).
- **Python envs on SoC1**:
  - Teleop/garment env (`/home/agi/.old_agi_home.bak/miniconda3/envs/teleop/bin/python`): `torch`, `onnxruntime`, `scipy`, `zmq`, `bleak`, `mujoco`.
  - Gear-SONIC runtime env (`/agibot/data/home/agi/gear-sonic/venv/bin/python`): runs `pc2_kplanner_onnx.py` and `x2_pose_watchdog.py`.

## 2. Port Map (Production vs Isolated Shadow)

| Channel | Production Port | Isolated Shadow Port | Direction |
| --- | ---: | ---: | --- |
| `pico_intent` (`smpl_joints`, `human_quat`, `engaged`) | `5573` | `15573` (`15575` w/ relay) | Garment -> Token service |
| `pose` (`motion_token[64]`, `engaged`) | `5574` | `15574` | Token service -> Deploy |
| `x2_debug` (robot state echo) | `5557` | `15557` | Deploy -> Token service & monitor |
| `kplanner` upstream `pose` | `5556` | `5556` (read-only SUB) | `pc2_kplanner` -> Watchdog |
| Watchdog downstream `pose` | `5558` | `15558` | Watchdog -> Deploy |
| `planner_cmd` | `5563` | `0` (disabled) | Token service -> `pc2_kplanner` |

## 3. Verified SONIC Policy Triplet & Tuning

- Pose head: `/agibot/data/home/agi/gear-sonic/policies/x2_sonic_s1ft16000_g1.onnx`
- Token head: `/agibot/data/home/agi/gear-sonic/policies/x2_sonic_s1ft16000_g1_token.onnx` (`ori_mode=heading`, `codec_fingerprint=f011417bc6265fcf5cef202c...`)
- Tokenizer: `/agibot/data/home/agi/gear-sonic/policies/x2_smpl_tokenizer_v11release.onnx` (`ori_mode=heading`, unstamped fingerprint -> ritual prints warning, does not block)
- Tuning preset: `trained_gains_s0.yaml` (`kp_scale_waist_pr: 2.00`, `kd_scale_waist_pr: 5.51`, unity leg/arm scales, `max_target_dev_waist: 0.45`, `max_target_dev_leg: 0.70`, `action_clip: 20.0`) + `--max-target-dev-head 0.40 --wrist-bypass ik --head-bypass ref`.

## 4. Parameterizable Knobs (Do Not Hardcode Across Units)

Check and override via env/CLI instead of assuming constants:
- Garment BLE MACs (`--addr-up`, `--addr-down`): inspect `bluetoothctl devices` first if switching suits.
- Engage & trigger paths: `PICO_ENGAGE_FILE` (default `/home/agi/pico_intent.engage`), isolated shadow `/tmp/x2_shadow_pico_intent.engage`.
- Heading guard: `ROOT_HEADING_BAND_DEG` (`0` enables free walking/turning under spotter control; `25` locks against >25 deg drift/turn).
- Max runtime window: `X2_MAX_DURATION` (`300` s default for initial trials).
