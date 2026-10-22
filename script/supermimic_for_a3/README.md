# SuperMimic for A3

Full A3 state (83), active-joint observations/actions (93/29), a learned differentiable
world model, H=32 BPTT actor updates, and a GAE critic. Real transitions come from
MuJoCo's native threaded rollout. The physics uses the original A3 1 ms timestep
with 20 substeps per 50 Hz control interval; 5 ms was numerically unstable.

The initial port uses the paper's MLP widths and observation/goal history. Batch
sizes are reduced for the shared RTX3090. It has no hardware deployment entry.
See `THIRD_PARTY.md` and the stage report for sources and differences.

Run from `/opt/data/private/yuanqi_ops` using the isolated Python environment
`envs/supermimic_a3_20261023/bin/python`. It reuses the existing ZHY environment's
read-only site packages through `base_packages.pth`; no base packages are updated.
Required versions verified here: Python 3.11, torch 2.8.0+cu128, MuJoCo 3.3.7,
numpy 1.26.0 and scipy 1.15.3. A fresh installation must rerun the backend checks.

```bash
envs/supermimic_a3_20261023/bin/python script/supermimic_for_a3/prepare_motion.py \
  --root /opt/data/private/hechengzhu/ZHY \
  --output data/supermimic_for_a3/NEW_MOTION_VERSION

envs/supermimic_a3_20261023/bin/python script/supermimic_for_a3/verify_backend.py \
  --root /opt/data/private/hechengzhu/ZHY \
  --motions data/supermimic_for_a3/motions_20261023_M02 \
  --output data/supermimic_for_a3/NEW_BACKEND_RECEIPT.json

envs/supermimic_a3_20261023/bin/python script/supermimic_for_a3/train.py \
  --root /opt/data/private/hechengzhu/ZHY \
  --motions data/supermimic_for_a3/motions_20261023_M02 \
  --backend-receipt data/supermimic_for_a3/source_audit_20261023/backend_V04.json \
  --output data/supermimic_for_a3/NEW_RUN --steps 3000 \
  --num-envs 16 --threads 4 --compile \
  --clips 001_walk_front_slow 042_squat_shallow_repeated \
    058_punch_front_alternating 077_reach_overhead_both

envs/supermimic_a3_20261023/bin/python script/supermimic_for_a3/evaluate.py \
  --root /opt/data/private/hechengzhu/ZHY \
  --motions data/supermimic_for_a3/motions_20261023_M02 \
  --checkpoint data/supermimic_for_a3/RUN/final.pt \
  --output data/supermimic_for_a3/NEW_EVALUATION

envs/supermimic_a3_20261023/bin/python script/supermimic_for_a3/audit_run.py \
  --run data/supermimic_for_a3/RUN \
  --baseline data/supermimic_for_a3/BASELINE_EVALUATION/receipt.json \
  --trained data/supermimic_for_a3/TRAINED_EVALUATION/receipt.json \
  --output data/supermimic_for_a3/NEW_MATCHED_AUDIT.json
```

Output directories must be new. Failed runs and receipts are retained. There is no
automatic restart. `step_000000.pt` is this run's baseline. `final.pt` is saved
after actual training and still requires deterministic physics evaluation.
`status.json` is an atomic progress snapshot; `metrics.jsonl` contains training
metrics. SIGTERM/SIGINT requests a checkpointed stop after the current update.

Evaluation starts each complete clip at frame zero, with no domain randomization,
observation/action noise, or reset noise. The return denominator includes the
entire reference clip, including the part lost after early termination. The
reserved complete clips are 004/048/056/080. They are not claimed to be independent
subjects or recordings from a different source distribution.

Evaluate the same eight clips for this run's `step_000000.pt` and `final.pt`.
The audit checks actual network changes, finite parameters/optimizer states,
optimizer step counts, artifact identities, and matching evaluation clips and
protocols. Its verification status confirms the training/evaluation evidence;
the reported returns, completion and failures determine the performance result.

`plot_matched.py --audit MATCHED_AUDIT.json --output NEW_FIGURE_DIRECTORY`
renders the verified per-clip comparison as PNG and PDF, retaining the early
termination counts and full-clip denominator in the figure.

`diagnose_model.py --evaluation FINAL_EVALUATION/receipt.json --root A3_ASSET_ROOT
--output NEW_DIAGNOSTIC.json` measures 1/8/32-step frozen-model errors on the
existing physical traces, with a constant-velocity integration comparison. It
performs no new physics steps or optimizer updates and does not identify a
unique cause of tracking failure.

`calibrate_world_model.py --checkpoint RUN/final.pt --root A3_ASSET_ROOT
--output NEW_CALIBRATION --compile` collects two independent frozen-policy replay
sets from the checkpoint's training clips, then fits only the world model for
1000 updates. The fixed 1/8/32-step validation windows never enter fitting. Policy
and critic updates are zero. The candidate decision is a 20% improvement in
32-step root RMSE with no more than 10% regression in one-step state MSE or
32-step active-joint RMSE. This does not establish policy performance.

Use `launch.py --entry calibrate_world_model` with an explicit new log/record to
launch that calibration with the same recoverable process record as training.

`audit_calibration.py --calibration CALIBRATION_DIRECTORY --output NEW_AUDIT.json`
checks saved replay counts, separate collection seeds, complete validation-window
membership and exclusion from fitting, artifact identities and Adam update counts.
It recomputes initial/final model errors from the saved weights and fixed windows.
The actor freeze evidence remains the hash-bound calibration entry's runtime
tensor-equality check; no independent final actor snapshot is exported by C01.

For a new run, `train.py --initialize-from COMPLETED_RUN/final.pt` inherits the
completed policy, critic, world model and their Adam states. References, backend,
simulator and network/trainer contracts must match. Optional
`--calibration-audit VERIFIED_CALIBRATION.json` substitutes only the calibrated
world model and its Adam state, and requires a passed independently verified gate.
Use an explicit fresh output directory and seed; replay, sampler and episodes are
fresh. The learning rate schedule continues from the source collection count.
Local update counters start at zero and the audit checks optimizer increments
against inherited counts. In this mode `step_000000.pt` is a trained parent
initialization, not a random policy. Choose model warmup explicitly for the new run.

`complete_run.py --run RUN_DIRECTORY --launch TRAINING_LAUNCH.json
--output NEW_COMPLETION_DIRECTORY --log-dir NEW_LOG_DIRECTORY` waits for that exact
PID/start time/boot and command to exit successfully, then runs step-zero and final
eight-clip evaluation, matched auditing and `assess_policy.py`. It never starts or
restarts training. Each stage runs once; a failure stops the pipeline and retains
its logs. `launch.py --entry complete_run` records the independent process.

The policy assessment checks the preregistered engineering gate: all four known
training clips finish without termination, mean full-clip return is at least 0.5,
and every training clip has executed joint RMSE at most 0.15 rad. It reports the
four previously inspected development clips separately. Passing this limited gate
does not establish generalization, hardware readiness or paper-scale performance.
Focused controls: `python script/supermimic_for_a3/test_completion.py -v`.

`MUJOCO_GL=egl python script/supermimic_for_a3/render_trace.py
--evaluation EVALUATION/receipt.json --admission RUN/admission.json --clip CLIP_NAME
--output NEW_RENDER_DIRECTORY` renders the saved policy state and reference target
side by side with the same camera. It uses no physics integration, policy inference
or optimizer updates. It displays full clip duration and early termination status.
Optional `--max-seconds` explicitly makes a truncated preview; the default includes
the complete executed trace. Outputs are an MP4, last-frame PNG and hash receipt.
The verified environment has EGL, Pillow and ffmpeg; the viewer needs no model weights.
