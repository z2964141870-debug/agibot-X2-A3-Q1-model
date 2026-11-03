# Primebot Q1 first, conditional MorphMimic comparison

2026-11-03: the user explicitly requested the five-stage plan. Reproduce Primebot's native trainer on real Q1 V3, finish250000 iterations, evaluate, then only after the frozen run-through gate passes start fresh G1/Q1 shared MorphMimic training. Both evaluations and a comparative report are required. No application goal or agent is created.

## Previous shared capacity failure

NEWGPU_TRAIN01 ended FAILED at2026-11-03T07:59:57.456779Z (15:59:57 China time), not RUNNING. G1/Q1 native execution checks completed. Shared E2048 per robot completed zero full iterations; each performed10 WM and zero motion/critic/KP/KP-critic updates, with413696 transitions. G1 could not allocate174MiB at first motion-policy forward: global free153.75MiB, its process10.19GiB, peer Q1 13.21GiB. Q1's Gloo peer-close is secondary. Guard1015 reaped worker1022/return1. No formal train stage launched. Original failure directory and partial updates remain untouched; cache hooks released0, so no cache-fix success is claimed.

Replacement GPU read-only query returned GPU-c602a45a-0dd8-9ad2-9adc-4f863facd1d4,1MiB,recovery=None. This is not the prior failed GPU; no reset was performed.

## Frozen recipe and differences

Primebot source118338b19dcd14f768d1e7de5a48d04b78987688 complete archive182958B/SHA2562ff69dc30951a174eeae595a48792a85c4ac90a90762a2de9d182f8e3c820604 matches Mac/server. HTTPS Git clone failed; fixed codeload archive succeeded. Source/MIT license stay under ignored data/references.

Original lock: Python3.13, Torch2.7.0, MuJoCo3.6.0, Warp1.12.0, MuJoCo Warp875c4caf06d71b12b2036c327e7a07ce08a78b9a. Independent environment installed46 locked packages with uv sync --frozen --no-dev on Python3.13.11. First sync's missing Git remote-https helper failure remains; explicit existing GIT_EXEC_PATH fixed the bootstrap without changing dependencies. Log: logs/primebot_environment_sync_attempt02_20261103.log. CPU real V3 constructor passed nq29/nv28/nu22/nbody27, state79/context143/obs72, torque motors unchanged, no CUDA context. GPU control, conversion/screen, capacity and formal training are still pending.

Default env/WM/policy4096/2048/2048; explicit capacity CUDA OOM alone permits2048/1024/1024 then1024/512/512. Every formal tier retains250000 iterations, seed0, fresh initialization, full networks, WM10/K8/H32, critic5x4, LR2e-4, original schedule/reward/control/delay. Three optimizers differ from Morph's five. No Morph WM-only warmup; profile weights never seed formal training.

V3 requires explicit NPZ/version adaptation: actual pelvis, WXYZ, named22 joints,50Hz, FK. Apply unchanged upstream screening to559 candidates, retain rejection/review reasons, freeze accepted paired selection. Do not rename V3 as V2. Passive zero-checkpoint/counter hooks and actual control/data deviations are separately recorded.

## Evaluation and follow-up

Freeze forward loop001/A022, start002/A023, stop002/A024, seeds0/1/2 before training. Prior stop005/A026 is absent from this paired selection; no after-result substitution. Gate: final9/9 complete; equal-clip common-valid-prefix joint-angle and pelvis-relative body-position RMSE both strictly improve over zero update, excluding initialization. Missing/nonfinite/incomparable cases fail. Accepted-library results, original-speed videos and CPU checkpoint audit must all be delivered.

Foot clearance/alternation/loaded slip, posture and upper-body timing remain required diagnostics, not strict natural-gait gating. Failure stops phase three. Passing permits fresh shared G1/Q1 v0.1, initially eachE1024/WM512/imag-KP1024/I160000/WM-only2000/warmup400; explicit capacity OOM permits eachE512/I320000 and proportional settings. Both target410009600 transitions per robot; optimizer trajectories differ. Full eight-update/shared-state/save audits precede formal training.

Evaluate initial/final identically and Primebot's closest actual-count checkpoint within1% of the exposure target. Compare Morph motion directly; KP/G1 separately. Seen-Q1 shared training is not unseen-body zero-shot evidence.

## Execution and completion

Only native Windows hp_han to25362 and /opt/data/private/ZHY; boot90395f47-4035-4785-858e-b3d05c44a738/GPU-c602a45a-0dd8-9ad2-9adc-4f863facd1d4. Preserve exclusive lock,30GiB memory, temperature/finite/OOM/device/source guards. Recalculate run disk allowance from actual checkpoint and full replay size; old12GiB may not fit51 snapshots plus replay. Weights/NPZ stay SERVER_LOCAL_ONLY.

Current stage PREPARING, not trained or validated. Entry script/primebot_q1; authorization data/manifests/primebot_q1_staged_authorization_20261103.json. Publish code/config/README, independently verify full remote commit and Mac/server identities before execution. Later receipts determine installation, validation, training, evaluation and delivery status.
