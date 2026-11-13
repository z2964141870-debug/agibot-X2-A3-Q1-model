# SmartWear / CLOTHO FGP Preview

Read-only clothing acquisition, user-triggered T-pose capture and SMPL24 browser preview.
This package never imports the robot SDK, subscribes to robot feedback or publishes motor commands.
It uses the latest two-stage CLOTHO B canonicalizer and GlobalPose weights.
The 23.2MB weight bundled inside EgoLocate is PIP/FastPIP, not FGP-S. EgoLocate's
FGP adapter imports a separate sibling FGP project that is absent from that archive.

Selection follows EgoLocate's default: `enable_denoise=True`, 30-frame denoising,
`U100hL7.07h_residual.pth` and `Full_GR_OV_10.forward_frame_fast0`. The supplied CLOTHO class
resolves the pose weight to `checkpoints/globalpose/weights.pt`. Startup verifies the three exact
asset hashes; `/api/state.model_profile` exposes the choice. The absent external EgoLocate
`ble_web_pose` source prevents claiming its complete implementation is byte-identical.
The user has chosen this default pipeline; locating FGP-S is not a prerequisite for preparation.

Preparation status: recorded-input parity and browser checks pass, but live BLE notifications
and wearer calibration have not passed. The current Orin CPU B+GlobalPose baseline takes about
781ms/output, far above the 33.33ms input period. Sampling under this load skips temporal inputs;
dynamic reconstruction is not validated. Transport diagnostics and slow SMPL inspection are
available; an adequately fast backend is needed before real-time use. S architecture alone is not trained S.

## SoC1 Start

Native login remains Mac -> Windows -> SoC1, with interactive passwords. No forwarding is required.

After logging into SoC1:

```bash
cd /home/agi/x2_morph_human_20261113
bash script/x2_fgp_preview/run_orin.sh smpl
```

Open `http://10.0.1.41:8768` on a browser with access to the robot LAN (for example the Windows bridge).
The Mac can access this address once connected to the robot LAN directly. An SSH connection alone
does not make this private HTTP address reachable from the Mac. No tunnel is created by these scripts.

The service loads the models, then scans for up to 180 seconds. Each garment connection allows
60 seconds with at most two attempts. Waiting does not imply failure. If a scan ends without a
target, the page reports that observation; it does not retry forever. Ctrl-C releases owned BLE clients.

When both garments have fresh data, wear the clothes, stand in T-pose and click the calibration button.
Remain still for three seconds. At least 60 unique paired samples and the static-window checks are
required. SMPL inference starts only after this user-triggered capture passes the static check.
Stationarity does not prove the wearer is geometrically in T-pose; inspect the reconstructed mesh.
After a garment counter reset, capture calibration again. A gap holds the displayed last pose and
marks it stale; the inference history resets before reconstruction resumes.

For transport/static diagnostics without loading FGP:

```bash
bash script/x2_fgp_preview/run_orin.sh transport
```

This mode displays a static SMPL template, explicitly labelled as a template. It does not reconstruct
the wearer. Only one preview owns this package's lock at a time; stop the first mode before changing.

## Preparation And Diagnostics

The independent venv and private assets are prepared with:

```bash
bash script/x2_fgp_preview/setup_orin.sh
```

This installs only the bundled CPython3.10/aarch64 wheels in a private venv, checks every packaged
asset and loads the two FGP models. It does not open BLE or run the robot. CPU Torch is used;
system TensorRT does not turn it into CUDA inference. The recorded Orin CPU measurement above
does not include BLE, browser rendering or robot latency.
Large weights are staged separately from the small source bundle and checked against the manifest.
Existing matching weights are reused; unexpected existing assets fail verification.

Read-only monitoring, executed on SoC1:

```bash
python3 -c 'import json,urllib.request; print(json.dumps(json.load(urllib.request.urlopen("http://127.0.0.1:8768/api/state",timeout=10)),indent=2))'
python3 -c 'import json,urllib.request; d=json.load(urllib.request.urlopen("http://127.0.0.1:8768/api/smpl",timeout=10)); s=d.pop("sample"); print(json.dumps(dict(d,sample_summary=None if s is None else {k:v for k,v in s.items() if k not in ("pose_mat","joints_yup_m")}),indent=2))'
```

`/api/state` includes per-garment rate, receive age, rejected/duplicate/missing counters, calibration,
SMPL age and inference time. `/api/smpl` additionally exposes 24 local rotation matrices, metric root
translation and world joints in SMPL Y-up. Receive-monotonic timestamps are proxies, not hardware
capture times. The page shows stale holds distinctly; a frozen mesh is not current tracking.

Each launch writes a new `data/analysis/X2_FGP_LIVE_<timestamp>/` directory. Calibration captures
are saved individually. SMPL samples are recorded up to 64MiB per launch; inference continues after
that cap. Close with Ctrl-C. There is no motor-control command or MC handover step.

## Local Recorded Preview

```bash
bash /Users/yu/Documents/ChatGPT/元启/script/x2_fgp_preview/run_mac_replay.sh
```

The default URL is `http://127.0.0.1:8768`; pass another port as the first argument if already in use.
This renders actual saved latest-CLOTHO output. The page labels it historical, not live clothing.

## Interpretation

The 10 active IMUs are selected from 6 upper and 5 lower sensors using CLOTHO's exact order.
217-byte packets, finite acceleration and quaternion norm are checked. The undocumented checksum
is retained as unvalidated. Packet counters and the receiver clock do not establish hardware sync.

The new streaming adapter matches the prior CPU runner on recorded inputs. It preserves upstream
GlobalPose integration dt=1/60 and the existing 30Hz gyro conversion; this is not a validation of
that physical timebase. It keeps the original SMPL translation, without the old Unity display hacks.
This frontend does not alter MorphMimic's existing 180ms preview requirement or establish that the
complete FGP-policy-controller path can run at 50Hz.
