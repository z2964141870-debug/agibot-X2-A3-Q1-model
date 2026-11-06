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

## 已完成的实际对照与结论

2026-11-06本次中途检查已经完成。`80cbe2620a4bc63749573ed52c88cfd99e03b3ac` 公开并独立核远端后，Mac八记录634316B核同、服务器十一记录75016B核同及13运行时检查通过，14:34:05单次派发253425；14:45:08完成三G1保存点和GPU媒体。第一段G1前走严格复用，Q1与两次失败根均保留，没有重跑已完成物理。两型、两接口、三个保存点、三固定片共36/36完整；其中旧16k Q1 motion三例复用，本次33新例。它们为训练集固定seed0名义回放，不是统计独立试验或全393库结果。

**建议：当前配方的主要收益已进入平台期，不建议为了达到160000轮而继续长跑；应把32000保存点作为优先候选，再完成较大范围验收后决定收尾。** 这不是宣称全部动作充分收敛或自然步态已合格，也不是自动停止命令。原训练仍在运行；本次没有发送训练信号、改变保存频率或放宽final正式完成门。用户对检查点的采纳仍为null，低延迟goal仍paused。

三片等权完整关节RMSE，单位rad，越低越好：

| 机器人/接口 | 16000 | 32000 | 48000 | 32k到48k误差变化 |
| --- | ---: | ---: | ---: | ---: |
| Q1 motion | .071873 | .066745 | .068501 | +2.63% |
| Q1 human/H10 | .075750 | .069598 | .071632 | +2.92% |
| G1 motion | .078618 | .060437 | .061877 | +2.38% |
| G1 human/H10 | .104456 | .094814 | .108175 | +14.09% |

16k到32k的对应改善为7.14%/8.12%/23.13%/9.23%，32k到48k却没有继续改善。其他指标并非全部变差：G1 motion前走上肢和离地仍改善，Q1 motion平均右滑速下降约6.96%，起走躯干倾角下降；因此结论是收益饱和并有权衡/回退，不能把每项都描述成完全不动或单调退化。逐片变化见[结构化汇总](../data/manifests/q1_checkpoint_convergence_summary_20261106.json)。

尤其与遥操相关的human接口：Q1平均承重滑速左/右从11.16/24.86增至18.45/43.54mm/s，关节body相对误差增7.73%，上肢误差增8.09%；G1 human上肢误差增18.56%，躯干倾角RMS从.1071增至.1693rad，末根XY误差从.1522增至.2137m。Q1 motion前走末根误差从32k的.0375回到48k的.2157m，说明更低训练loss没有保证执行质量继续提高。

48k Q1脚底高度p95三片均值，motion左/右10.32/7.47mm、human8.28/7.77mm，对应参考17.87/17.82mm；低离地仍未解决。G1两接口完成率均高，但human上肢和侧倾仍有明显差距。不能将“跑完/不跌倒”当成“自然遥操”。

### 六宫图与训练信号

此前未向用户交付六宫图属于交付遗漏，现已补齐并逐图目视检查。共享G1/Q1各一张标准2x3；Primebot一张真实已记录指标2x3，以及一张明确标出五个历史缺项的标准2x3。旧Primebot没有记录完整回合统计、K8和想象存活率，不能回填伪造。共享图的六项都存在，原始点加500轮平滑，回合指标按真实结束回合数加权；快照截至53900轮。

最近2000轮窗口48k到53.9k，Q1标准化回报.742181→.743655（相对+0.20%），G1 .824447→.826005（+0.19%）；步奖励接近平台。世界模型八步误差仍有小幅下降：Q1 .442306→.435997（-1.43%），G1 .105252→.103432（-1.73%）。所以不是优化器不再更新，而是当前模型学习的增益没有在这些固定真实物理片上转化为稳定收益。失败比例低和想象存活接近1也不能代替真实步态诊断。

Primebot原生10000已完成训练的结论仍为“跟随跑通、自然步态未合格”。同三片motion完整关节RMSE均值.165292rad，Morph Q1的16k为.071873rad，下降约56.5%，滑移和上肢也多数更好；但前走末根误差曾由Primebot .003600m变Morph16k .268490m，不能说全面胜出。两套训练环境、样本预算和接口不同，不用轮数直接比较训练效率。

### 实际资源与交付

**2026-11-06 14:57:07北京时间**恢复会话63485实核原boot/GPU、唯一根、五PID/starttick及guard55672锁一致，TRAINING/RUNNING，两型各55000/160000轮、各141209600真实转移；五Adam550000/53000/1060000/53000/1060000、非有限全0。GPU16177MiB/53C/None，cgroup总23157530624B/实际限34292629504B/即时余11135098880B/OOMkill0/under_oom0。训练环境是**G1 1024+Q1 1024=总2048**，不是单型2048或“21024”。此次仍是11月4日11:06启动的原共享动作跟踪训练，没有新开训。

G1续查峰总25440194560B低24GiB=25769803776B，GPU峰16589MiB，OOM0；两次评估失败子进程均回收，原训练信号0、200执行源码/资产前后SHA保持。12段48k原速GPU EGL视频均完成服务器50fps全帧RGB解码，共6348帧。四组首中末联系表已Mac逐图目视；它们只显示离散帧，不代替连续视频判断。Q1/G1 human起走两段原速视频已Mac大小SHA和完整FFmpeg解码通过；其余十段完整MP4仍在服务器，未伪报全部Mac视频交付完成。

Mac本地：

- [Q1训练六宫图](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_SHARED_CONVERGENCE_CKPT16_32_48_20261106/curves/q1/training.png)
- [G1训练六宫图](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_SHARED_CONVERGENCE_CKPT16_32_48_20261106/curves/g1/training.png)
- [Primebot真实诊断六宫图](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_SHARED_CONVERGENCE_CKPT16_32_48_20261106/curves/primebot_diagnostics/training.png)
- [Primebot标准图及历史缺项](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_SHARED_CONVERGENCE_CKPT16_32_48_20261106/curves/primebot_standard/training.png)
- [Q1 human物理对照六宫图](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_SHARED_CONVERGENCE_G1_RESET_CONTEXT_20261106/analysis/q1_human.png)
- [G1 human物理对照六宫图](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_SHARED_CONVERGENCE_G1_RESET_CONTEXT_20261106/analysis/g1_human.png)
- [Q1 motion物理对照](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_SHARED_CONVERGENCE_G1_RESET_CONTEXT_20261106/analysis/q1_motion.png)
- [G1 motion物理对照](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_SHARED_CONVERGENCE_G1_RESET_CONTEXT_20261106/analysis/g1_motion.png)
- [Q1 human 48k起走原速视频](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_SHARED_CONVERGENCE_CKPT16_32_48_20261106/common_q1/48000/human/walk_forward_start_002__A023_s0/tracking.mp4)
- [G1 human 48k起走原速视频](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_SHARED_CONVERGENCE_G1_RESET_CONTEXT_20261106/g1_native_cpu/48000/human/walk_forward_start_002__A023_s0/tracking.mp4)

四训练PNG/JSON首包1022697B/SHA9f261bfd和最终结果/联系表包468141B/SHA215c6b5d均经原Windows/native25362传回并核完整SHA，分别16成员1261443B与18成员855661B逐大小SHA同。两human起走视频包433972B/SHAe4b5cda6完整核同。物理汇总仅在Mac读取保存JSON，0新推理/物理/优化；四张PNG/PDF/CSV和全部逐片变化已生成并目视。训练图PDF/原日志CSV仍在服务器；本次不为补媒体而重跑物理。

复现Mac保存结果分析：

```sh
data/environments/q1-teleop-mps-20261106/bin/python script/q1_checkpoint_review/analyze.py --source /Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_SHARED_CONVERGENCE_G1_RESET_CONTEXT_20261106/comparisons.json --output NEW_ANALYSIS_DIR --plot
```

下一项应是对优先保存点进行更大动作范围/扰动/真正human流验收，并另行决定原训练的正常收尾；本次未授权停止训练，不自行执行。32k只是三固定片平均关节误差更好的候选，不能据已看结果把它自动认证成全局最佳。原正式final全门与完整五阶段要求继续保留。
