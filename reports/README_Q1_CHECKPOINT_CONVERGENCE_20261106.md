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

## Actual Q1 Stage and Isolated G1 Correction

The published `c9ca15eac6a7ad47a5df6e8e990a0683bba947e0` entry was dispatched once at2026-11-06 13:48:21 Beijing as controller246946. Both32k/48k dual-robot CPU audits passed. Four standard/diagnostic six-panel reports were generated from1078 shared logged rows through53900 and10000 completed Primebot rows. Historical Primebot standard panels0/1/2/4/5 were not recorded and remain missing.

All five new Q1 groups completed three fixed cases each:16k human,32k motion/human and48k motion/human. Reusing the prior16k motion gives18 complete Q1 cases. Six48k Q1 videos were rendered with NVIDIA EGL. These are complete nominal training motions, not full-library or natural-gait acceptance.

The next G1 phase failed before a physical transition at the contact diagnostic's floor check. The saved unchanged native model contains exactly one horizontal plane at world z0, geom0 `terrain`, on static body1 `terrain`. The frozen evaluator incorrectly requires `geom_bodyid == 0`. Its worker250541 exited1 and was reaped; the failure and original output remain preserved. Peak review cgroup total25417736192B stayed below24GiB; training signals0.

The frozen native evaluator is among the1793 original training inputs and must not be edited. Independent `native_cpu.py` therefore uses the same rollout body and inherited force reader, changing only floor identification to a plane welded to world with verified world pose. An AST parity test verifies every other rollout operation is unchanged. Static-world, static-child, dynamic-ancestor and tilted/elevated-floor checks pass. The analyzer preserves failed cases and missing groups instead of filling missing performance with zero.

`resume_g1.py` prepares a new output and reuses the completed Q1 cases and three audits. It only runs the three missing G1 checkpoint groups and48k G1 media. It validates old guards/cleanup, original identities/resources/frozen inputs, and keeps the original combined7200second/3GiB budgets; no new Q1 physical replay or training. The formal final completion gate remains unchanged. Local checkpoint-review suite:12 tests passed. Actual continuation publication, server validation, dispatch and results remain pending.

Server Q1-stage output: `/opt/data/private/ZHY/data/supermimic_for_q1/Q1_SHARED_CONVERGENCE_CKPT16_32_48_20261106`. Compact six-panel/Q1 records package: `/tmp/q1_convergence_q1_curves_20261106.tgz`,6654473B, SHA256 `cf252b4469ab36cd1f4f3a52268177cfa058fe5a9697af679e4e01fb426b42e4`; Mac delivery and visual inspection pending. A separate transfer connection failed after Windows authentication; the existing11925 Linux session remains usable and no authentication loop was started.

The correction was published as `8514043ee1ae932e291be75b2e11dcdfe9390362`, independently matched to remoteQ1 and11 Mac records/645206B. Ten code/report records68405B were mirrored through the existing Linux session;12 pinned-runtime tests and the saved native terrain check passed. A single G1 continuation252090 was dispatched at14:18:21 Beijing. It completed16k motion forward1031/1031 transitions, then failed at the second clip reset: native action buffers created inside inference mode were being reset outside it. This is an evaluation-context error; the first physical result and failure remain preserved.

The next isolated correction wraps the entire per-case replay, including reset, in `torch.inference_mode()`. It reuses that saved first G1 case only after checking checkpoint, identical compiled scene, reference hashes and trace integrity. The new continuation still reuses all18 Q1 cases and both prior failure roots, with their combined original time/storage budgets. A regression check reproduces the inference-tensor reset error and validates consecutive resets in the proper context. Local suite now13 passed; new publication, server validation and continuation pending. No frozen input, dynamics, final gate, original training process or paused goal was modified.

First-delivery PNGs and compact JSON package1022697B/SHA256 `9f261bfd50b3abb2577da6a5034d311ea0b635704d36ef83520ad0db8dc8fb4a` was prepared on the server. Delivery uses bounded blocks through11925 because the separate transfer login failed. All four full-resolution PNGs remain pending Mac verification; no quality conclusion is inferred from delivery progress.
