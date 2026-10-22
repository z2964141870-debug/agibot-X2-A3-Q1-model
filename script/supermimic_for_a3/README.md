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
automatic restart. `step_000000.pt` is the untrained baseline. `final.pt` is saved
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
