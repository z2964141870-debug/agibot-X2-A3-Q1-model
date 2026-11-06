# Shared Q1 Checkpoint Convergence Review

2026-11-06 Beijing. User authorized completing the missing six-panel reports and comparing saved checkpoints before recommending whether to continue the original shared run. This is an independent intermediate review; it does not stop training, resume the paused low-latency goal, or replace formal final acceptance.

## Fixed Scope Before Results

Use existing 16000/32000/48000 checkpoints, both motion and human interfaces, and the same three fixed forward/start/stop clips at deterministic seed0. Reuse the completed16000 Q1 motion replay and original dual-robot16000 CPU audit. Audit the new32000/48000 saves before inference. No optimizer updates, new training, checkpoint-frequency changes or changes to frozen training inputs.

Q1 uses the already validated Primebot CPU physical runtime and isolated native MorphMimic CPU policy. G1 attempts the pinned native scene on CPU Warp with CUDA disabled; comparisons are within that same backend, not a claim of CPU/GPU numerical equivalence. Its first real execution is still pending. No GPU compute-lock contention; GPU EGL is used only for original-speed48000 replay media after physics processes exit.

Keep the original device, boot, five PID/startticks and guard55672 lock checks. Admission22GiB, exit24GiB, GPU18GiB/80C,100GiB free disk,3GiB total output,1800seconds per phase and7200seconds overall. Resource exits affect only owned review process groups. The original30GiB/86C training policy remains intact; no cache maintenance or training signals.

## Reporting and Interpretation

Deliver standard2x3 PNG/PDF/CSV/source receipts for each shared branch with raw points and trailing500-iteration curves; episode panels pool finished-episode denominators. Snapshot growing event files before reading. Complete the Primebot report from its existing final log; unrecorded standard metrics remain missing, with a separate diagnostic six-panel plot for the metrics actually recorded.

Compare completion first, then per-clip full-motion joint/body error, sole clearance, alternations, loaded slip, tilt, root error and upper-body error. Failed clips remain failures; do not select successful clips to manufacture a plateau. Before inspecting new results, a2% tracking gain,10% slip change,1mm clearance change,.01rad tilt change and1cm root change are practical exploratory comparison scales. These are neither acceptance thresholds nor an automatic stopping rule. Inspect every clip, both interfaces and G1 availability; training losses alone cannot establish physical convergence.

The report will distinguish a training-signal plateau, a physical-performance plateau on the tested clips, and unmet natural-gait/full-library/robot acceptance. The160000 target is a predefined sampling budget, not proof that those iterations are necessary. Actual results, publication verification, server execution and local artifact delivery are pending.

Local validation:4 focused selection/pooling/final-gate tests and8 existing intermediate resource/reaping/provenance tests passed. A first direct invocation of the existing tests lacked its requiredPYTHONPATH; the corrected documented module paths passed. Native CPU execution, scalar-tag availability and physical results still require the pinned server runtime.
