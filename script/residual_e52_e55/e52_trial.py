"""E52 trial driver: register, prepare, three single-run arms, verify.

Controller never imports tensor runtimes; worker stages run the guarded vendor
launcher. All writes stay under /opt/data/private/yuanqi_ops.
"""

import argparse
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

OPS = Path("/opt/data/private/yuanqi_ops")
TRAIN_ROOT = OPS / "data/training/a3_zhy_projection_finetune_20261018_E52"
sys.path.insert(0, str(OPS))

ROOT = Path("/opt/data/private/hechengzhu/ZHY")
sys.path.insert(0, str(ROOT))

from script.a3 import zhy_trial as guard  # noqa: E402
from script.a3.checkpoint_store import atomic_json, sha256  # noqa: E402
from script.a3.host_health import append_record, command  # noqa: E402
import e52_protocol as P  # noqa: E402

EXPERIMENT = "A3_ZHY_PROJECTION_FINETUNE_20261018_E52"
OUT = OPS / "data/experiments/a3_zhy_projection_finetune_20261018_E52"
LOG = OPS / "logs/a3_zhy_projection_finetune_20261018_E52"
MANIFEST = OPS / "data/manifests/a3_zhy_projection_finetune_20261018_E52.json"
LAUNCHER = ROOT / "script/vendor/sonic_for_a3/train_a3_035_fromscratch.sh"
SOURCE = ROOT / P.SOURCE_CHECKPOINT
ANCHOR = ROOT / "data/experiments/a3_zhy_gradient_20261012_E11/rollout.pt"
ANCHOR_TARGETS_E52 = OUT / "anchors_e45_step4.pt"
TARGETS = ROOT / "data/experiments/a3_zhy_modules_20261012_E12/module_outputs.pt"
MOTIONLIB_SOURCES = [
    "data/experiments/a3_zhy_cloth_resource_20261013_E38/motionlib/G_0113_zj_train__183926_zj_walk.pkl",
    "data/experiments/a3_zhy_cloth_resource_20261013_E38/motionlib/G_0113_zj_train__192333_zj_warmUp.pkl",
    "data/experiments/a3_zhy_cloth_resource_20261013_E38/motionlib/G_20251226_fjw__fjw_140008_walk.pkl",
    "data/experiments/a3_zhy_cloth_failure_20261013_E45/motionlib/G_20251229_zzx__zzx_111505_walk.pkl",
]
CODE_FILES = [
    "script/a3/cloth_constrained_trial.py", "script/a3/cloth_constrained_trainer.py",
    "script/a3/cloth_cached_tokens.py", "script/a3/cloth_cached_trial.py",
    "script/a3/cloth_curve_trainer.py", "script/a3/cloth_curve_protocol.py",
    "script/a3/cloth_failure_trial.py", "script/a3/cloth_failure_protocol.py",
    "script/a3/cloth_failure_callback.py", "script/a3/finite_retention_trainer.py",
    "script/a3/finite_retention_protocol.py", "script/a3/finite_adam.py",
    "script/a3/optimizer_metadata_contract.py", "script/a3/test_optimizer_metadata_contract.py",
    "script/a3/adam_direction_protocol.py", "script/a3/adam_direction_probe.py",
    "script/a3/candidate_retention_protocol.py", "script/a3/zhy_trial.py",
    "script/a3/host_health.py", "script/a3/checkpoint_store.py",
    "script/a3/resumable_sonic.py", "script/a3/evaluate_mujoco.py",
    "script/a3/frozen_evaluation_launch.py", "script/a3/cloth_cached_tokens.py",
    "script/vendor/sonic_for_a3/train_a3_035_fromscratch.sh",
]
LOCAL_CODE = ["e52_protocol.py", "e52_trainer.py", "e52_trial.py", "test_e52_controls.py"]


def read(path):
    return json.loads(Path(path).read_text())


def _foreign_large_processes():
    listing = subprocess.run(["ps", "-eo", "rss=,comm="], capture_output=True, text=True).stdout
    rows = []
    for line in listing.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) > 1024**2:
            if not any(name in parts[1] for name in ("sshd", "bash", "ps ", "python", "tmux")):
                rows.append(line.strip())
    return rows


def preflight(baseline=None):
    rows = []
    for index in range(3):
        value = guard.preflight()
        gpus = value.get("gpus") or []
        if not gpus or any(float(g.get("utilization.gpu", -1)) != 0.0 for g in gpus):
            raise RuntimeError("GPU not idle; no startup")
        if value["container_memory"]["available_estimate_bytes"] < P.ENTRY_HEADROOM_GIB * 1024**3:
            raise RuntimeError("Insufficient 8 GiB entry headroom; no startup")
        if value["container_memory"]["oom_kill"] != 0:
            raise RuntimeError("Container OOM kills observed; no startup")
        foreign = _foreign_large_processes()
        if foreign:
            raise RuntimeError("Other large process; no parallel computation: " + repr(foreign[:3]))
        if value.get("temperatures_c", {}).get("x86_pkg_temp", 0) >= P.MAX_CPU_C:
            raise RuntimeError("CPU package temperature at stop threshold")
        if any(float(g.get("temperature.gpu", 0)) >= P.MAX_GPU_C for g in gpus):
            raise RuntimeError("GPU temperature at stop threshold")
        rows.append(value)
        if index < 2:
            time.sleep(5)
    return rows


def register():
    if OUT.exists() or (OPS / "data/training/a3_zhy_projection_finetune_20261018_E52").exists():
        raise FileExistsError("E52 must not be reset or repeated")
    baseline = preflight()
    source_sha = sha256(SOURCE)
    if source_sha != "3e679581fb1386e1fe718a1497af62615d8555a4bc03853e092a3b3b09edd645":
        raise ValueError("E45 step4 checkpoint identity changed")
    e45 = read(ROOT / "data/experiments/a3_zhy_cloth_failure_20261013_E45/registration.json")
    if e45["training_sequences"] != ["G_0113_zj_train/183926_zj_walk", "G_0113_zj_train/192333_zj_warmUp",
                                     "G_20251226_fjw/fjw_140008_walk", "G_20251229_zzx/zzx_111505_walk"]:
        raise ValueError("E45 corpus identity changed")
    e50 = read(ROOT / "data/experiments/a3_zhy_candidate_retention_followup_20261014_E50/result.json")
    if "LOCAL_PROJECTION_REJECTED" not in json.dumps(e50):
        raise ValueError("E50 single-step projection history changed")
    inputs = {P.SOURCE_CHECKPOINT: source_sha,
              "data/experiments/a3_zhy_gradient_20261012_E11/rollout.pt": sha256(ANCHOR),
              "data/experiments/a3_zhy_modules_20261012_E12/module_outputs.pt": sha256(TARGETS)}
    for rel in MOTIONLIB_SOURCES:
        inputs[rel] = sha256(ROOT / rel)
    code = {}
    for rel in CODE_FILES:
        code[rel] = sha256(ROOT / rel)
    for rel in LOCAL_CODE:
        code["yuanqi_ops/" + rel] = sha256(OPS / rel)
    for directory in (OUT, LOG, OPS / "data/manifests", OPS / "data/training",
                      OPS / "reports", OPS / "data/training/a3_zhy_projection_finetune_20261018_E52"):
        directory.mkdir(parents=True, exist_ok=True)
    job = dict(
        experiment=EXPERIMENT, trial="R2x", registered_at=time.time(),
        contract="analysis/EXP_CONTRACTS.md section 1 (Exp-1a); lead approval 2026-10-18 with five execution requirements",
        hypothesis="H1 Adam displacement mechanism; E48/E50 evidence chain",
        main_variable="three-arm Adam displacement treatment (A original+hard gate, B E50 projection+gate recorded, C LRx0.1+gate recorded)",
        arms=dict(P.ARMS_NOTES), arm_launch_overrides={k: list(v) for k, v in P.ARMS_LAUNCH_OVERRIDES.items()},
        source_experiment=P.SOURCE_EXPERIMENT, source_checkpoint=P.SOURCE_CHECKPOINT,
        source_updates=P.SOURCE_UPDATES, source_optimizer_steps=P.SOURCE_OPTIMIZER_STEPS,
        updates=P.UPDATES, optimizer_steps=P.OPTIMIZER_STEPS, sampled_transitions=P.TRANSITIONS,
        checkpoints=list(P.CHECKPOINTS), seed=0, num_envs=16, epochs=1, minibatches=4,
        fp32=True, frozen_codec_std=True, original_reward_routing_reset=True,
        original_adaptive_lr_rule=True, actor_learning_rate_base=P.BASE_LR,
        arm_c_lr_scale=P.LR_SCALE_ARM_C, teacher_anchor_frames=384,
        anchor_rmse_budget=0.05, anchor_max_abs_budget=0.5,
        arm_a_gate_intercepts=True, arm_b_gate_records=True, arm_c_gate_records=True,
        runtime_limit_train_seconds=P.RUNTIME_LIMIT_TRAIN_SECONDS,
        runtime_limit_eval_seconds=P.RUNTIME_LIMIT_EVAL_SECONDS,
        memory_reserve_gib=P.MEMORY_RESERVE_GIB, max_cpu_c=P.MAX_CPU_C, max_gpu_c=P.MAX_GPU_C,
        boot_id=baseline[-1]["boot_id"],
        baseline_oom_kill=baseline[-1]["container_memory"]["oom_kill"],
        entry_headroom_bytes=baseline[-1]["container_memory"]["available_estimate_bytes"],
        code_sha256=code, input_sha256=inputs,
        criteria=list(P.CRITERIA),
        stages={k: list(v) for k, v in P.STAGES.items()},
        writes_root=str(OPS), repo_tree_zero_writes=True,
        single_run="each arm exactly one training launch; no re-run after any terminal status",
        candidate_deployment="FORBIDDEN_NEVER_REPLACE_035",
    )
    atomic_json(OUT / "registration.json", job)
    atomic_json(MANIFEST, dict(experiment=EXPERIMENT, status="REGISTERED",
                               registration="data/experiments/a3_zhy_projection_finetune_20261018_E52/registration.json"))
    for value in baseline:
        append_record(LOG / "preflight.jsonl", value)
    print("E52 registered; arms A/B/C single-run; repo tree untouched", flush=True)


def arm_launch_argv(arm):
    trainer = P.ARMS_TRAINER[arm]
    callback = "script.a3.finite_retention_trainer.FiniteCheckpointCallback"
    return ["bash", str(LAUNCHER),
            "seed=0",
            "trainer._target_=" + trainer,
            "algo.config.num_learning_epochs=1",
            "algo.config.actor_learning_rate=" + ("2e-7" if arm == "C" else "2e-6"),
            "algo.config.adaptive_lr_min=" + ("1e-7" if arm == "C" else "1e-6"),
            "algo.config.adaptive_lr_max=" + ("2e-6" if arm == "C" else "2e-5"),
            "callbacks.model_save._target_=" + callback,
            "callbacks.model_save.save_frequency=8",
            "+callbacks.model_save.target_global_step=8",
            "++algo.trl.bf16=false", "++algo.trl.fp16=false", "++algo.trl.tf32=false",
            ] + list(P.ARMS_LAUNCH_OVERRIDES[arm])


def train_stage(arm):
    job = read(OUT / "registration.json")
    if sha256(SOURCE) != job["input_sha256"][P.SOURCE_CHECKPOINT]:
        raise ValueError("Source checkpoint changed since registration")
    run_dir = OPS / ("data/training/a3_zhy_projection_finetune_20261018_E52/arm_" + arm.lower())
    log_dir = LOG / ("arm_" + arm.lower())
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "motionlib").mkdir(exist_ok=True)
    started = time.time()
    env = dict(os.environ)
    env.update(
        ISAAC_PYTHON=str(ROOT / "data/environments/a3-sonic/bin/python"),
        REPO_DIR=str((ROOT / "script/vendor/sonic_for_a3").resolve()),
        PYTHONPATH=str(OPS) + ":" + str(ROOT),
        PYTHONDONTWRITEBYTECODE="1",
        EXPERIMENT_DIR=str(run_dir),
        YUANQI_RESUME_LOG_DIR=str(log_dir),
        MOTION_FILE=str(run_dir / "motionlib"),
        CHECKPOINT=str(SOURCE),
        YUANQI_CLOTH_ANCHOR_ROLLOUT=str(ANCHOR),
        YUANQI_CLOTH_ANCHOR_OUTPUTS=str(ANCHOR_TARGETS_E52),
        YUANQI_E52_ARM=arm,
        YUANQI_E52_SOURCE_CHECKPOINT=str(SOURCE),
        NUM_ENVS="16", NUM_MINI_BATCHES="4", NUM_PROCESSES="1",
        NUM_LEARNING_ITERATIONS="8",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        WANDB_MODE="disabled", ACCELERATE_MIXED_PRECISION="no",
    )
    atomic_json(OUT / ("training_launch_arm_" + arm.lower() + ".json"),
                dict(argv=arm_launch_argv(arm), motion_file=str(run_dir / "motionlib"),
                     checkpoint=str(SOURCE), num_learning_iterations=8))
    for rel in MOTIONLIB_SOURCES:
        target = run_dir / "motionlib" / Path(rel).name
        if not target.exists():
            target.symlink_to(str(ROOT / rel))
    console = (log_dir / "console.log").open("a")
    console.write("== E52 arm %s start %s ==\n" % (arm, time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    for i in range(3):
        value = guard.preflight()
        append_record(log_dir / "preflight.jsonl", value)
        if i < 2:
            time.sleep(5)
    console.flush()
    process = subprocess.Popen(arm_launch_argv(arm), stdout=console, stderr=console,
                               cwd=str(ROOT), env=env)
    try:
        code = process.wait(timeout=P.RUNTIME_LIMIT_TRAIN_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        atomic_json(OUT / ("train_arm_" + arm.lower() + ".json"),
                    dict(arm=arm, status="STOPPED_RUNTIME_CAP", elapsed=time.time() - started))
        return 2
    console.close()
    contract_path = log_dir / "cloth_contract.json"
    gate_row = None
    if contract_path.exists():
        rows = read(contract_path).get("optimizer_steps", [])
        for row in rows:
            if row.get("teacher_action_rmse", 0) > 0.05 or row.get("teacher_action_max_abs", 0) > 0.5:
                gate_row = row
                break
    status = "TRAINED" if code == 0 else ("ARM_A_TRAINED_GATED" if gate_row else "STOPPED_LAUNCH_FAILURE")
    if arm != "A" and gate_row:
        status = "STOPPED_UNEXPECTED_GATE"
    atomic_json(OUT / ("train_arm_" + arm.lower() + ".json"), dict(
        arm=arm, status=status, exit_code=code, elapsed_seconds=round(time.time() - started, 3),
        gate_hit_step=(gate_row or {}).get("optimizer_step"),
        gate_row=gate_row, launcher=str(LAUNCHER)))
    print("E52 arm", arm, "status", status, "gate", (gate_row or {}).get("optimizer_step"), flush=True)
    return 0 if status in ("TRAINED", "ARM_A_TRAINED_GATED") else 2


def prepare_stage():
    job = read(OUT / "registration.json")
    failures = []
    for name, identity in job["input_sha256"].items():
        path = ROOT / name if not name.startswith("yuanqi_ops/") else OPS / name.split("/", 1)[1]
        if not path.exists() or sha256(path) != identity:
            failures.append(name)
    if failures:
        raise ValueError("Registered input or code changed: " + repr(failures))
    env = dict(os.environ)
    env.update(PYTHONPATH=str(OPS) + ":" + str(ROOT) + ":" + env.get("PYTHONPATH", ""),
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([os.environ.get("E52_PYTHON", "python"), str(OPS / "test_e52_controls.py")],
                            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    (LOG / "control_tests.log").write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        raise ValueError("E52 control tests failed; see control_tests.log")
    atomic_json(OUT / "prepare.json", dict(status="PREPARED", control_tests="PASS",
                                           tested_at=time.time()))
    print("E52 prepare complete; controls pass", flush=True)


ARM_CHECKPOINTS = {
    "A": "arm_a/milestone_4/model_step_000004.pt",
    "B": "arm_b/milestone_8/model_step_000008.pt",
    "C": "arm_c/milestone_8/model_step_000008.pt",
}
EVAL_ROOT = ROOT / "data/experiments/a3_zhy_projection_finetune_20261018_E52"
EVAL_LOG_ROOT = ROOT / "logs/a3_zhy_projection_finetune_20261018_E52"
BASELINE_CLOTH = ROOT / "data/experiments/a3_zhy_smpl_frozen_20261013_E26/evaluation"
BASELINE_OFFICIAL = ROOT / "data/experiments/a3_zhy_search_20261012_E02/official_oracle"


def references_farm(kind):
    farm = EVAL_ROOT / (kind + "_references")
    farm.mkdir(parents=True, exist_ok=True)
    if kind == "cloth":
        selection = read(ROOT / "data/experiments/a3_zhy_smpl_frozen_20261013_E26/registration.json")["selection"]
        for row in selection:
            source = ROOT / row["source"] / "reference.csv"
            link = farm / row["filename"]
            if not link.exists():
                link.symlink_to(os.path.relpath(source, farm))
    else:
        motions = read(ROOT / "data/experiments/a3_zhy_search_20261012_E02/job.json")["diagnostic_motions"]
        for row in motions:
            path = Path(row["path"])
            link = farm / path.name
            if not link.exists():
                link.symlink_to(os.path.relpath(path, farm))
    return farm


def evaluate_stage(arm, kind):
    from script.a3 import evaluate_mujoco as evaluation
    from script.a3.frozen_evaluation_launch import LaunchProxy, arguments
    source_checkpoint = TRAIN_ROOT / ARM_CHECKPOINTS[arm]
    if not source_checkpoint.exists():
        raise ValueError("Arm checkpoint missing: " + str(source_checkpoint))
    milestone_dir = ARM_CHECKPOINTS[arm].split("/")[1]
    copy_dir = EVAL_ROOT / "checkpoints" / ("arm_" + arm.lower()) / milestone_dir
    copy_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = copy_dir / source_checkpoint.name
    if not checkpoint.exists():
        shutil.copy2(source_checkpoint, checkpoint)
        sidecar = source_checkpoint.with_suffix(".json")
        if sidecar.exists():
            shutil.copy2(sidecar, copy_dir / sidecar.name)
    run_config = TRAIN_ROOT / ARM_CHECKPOINTS[arm].split("/")[0] / "config.yaml"
    if run_config.exists():
        shutil.copy2(run_config, copy_dir / "config.yaml")
        meta = run_config.parent / "meta.yaml"
        if meta.exists():
            shutil.copy2(meta, copy_dir / "meta.yaml")
    else:
        raise ValueError("Arm run config.yaml missing for eval identity")
    rel = checkpoint.relative_to(ROOT / "data")
    import script.a3.checkpoint_store as store
    args = arguments(EVAL_ROOT / ("eval_arm_" + arm.lower() + "_" + kind),
                     EVAL_LOG_ROOT / ("eval_arm_" + arm.lower() + "_" + kind),
                     checkpoint, references_farm(kind))
    if kind == "official":
        args.frame_stride = 4
    evaluation.preflight = guard.preflight
    evaluation.monitor_child = lambda child, directory, config: (child.wait(), None)
    evaluation.subprocess = LaunchProxy(evaluation.subprocess)
    if evaluation.evaluate_explicit(args):
        raise RuntimeError("E52 arm " + arm + " " + kind + " evaluation failed; no repeat")
    atomic_json(OUT / ("eval_arm_" + arm.lower() + "_" + kind + ".json"),
                dict(arm=arm, kind=kind,
                     checkpoint=str(checkpoint),
                     checkpoint_source=str(source_checkpoint),
                     checkpoint_source_sha256=sha256(source_checkpoint),
                     checkpoint_sha256=sha256(checkpoint), completed=time.time(),
                     output=str(EVAL_ROOT / ("eval_arm_" + arm.lower() + "_" + kind)),
                     writes_scope="NEW_NAMESPACE_ONLY_repo_data_experiments_a3_zhy_projection_finetune"))
    print("E52 arm", arm, kind, "evaluation complete", flush=True)


def verify_effect_stage():
    import numpy as np
    from script.a3.verify_zhy_search import verify_metrics

    def read_json(path):
        return json.loads(Path(path).read_text())

    rows, per_arm = [], {}
    for arm in P.ARMS:
        summaries = {}
        for kind, baseline in (("cloth", BASELINE_CLOTH), ("official", BASELINE_OFFICIAL)):
            directory = EVAL_ROOT / ("eval_arm_" + arm.lower() + "_" + kind)
            published = read_json(directory / "explicit_summary.json")["motions"]
            for record in verify_metrics(directory):
                name = record["motion"]
                before = read_json(baseline / (name + ".metrics.json"))["motions"][0]
                after = read_json(directory / (name + ".metrics.json"))["motions"][0]
                trace = read_json(directory / (name + ".timeseries.json"))["motions"][0]
                q, target = np.asarray(trace["q_state_29"]), np.asarray(trace["reference_q_29"])
                old_rmse = before["tracking"]["all_29_rmse"]
                new_rmse = float(np.sqrt(np.mean((q - target) ** 2)))
                falling = np.flatnonzero((np.asarray(trace["root_height"]) < .45) |
                    (np.abs(trace["root_roll_deg"]) > 60.) | (np.abs(trace["root_pitch_deg"]) > 60.))
                fallen = bool(len(falling))
                if fallen != after["fall"]:
                    raise ValueError("Independent fall reconstruction failed")
                retention = (not fallen and new_rmse <= old_rmse * 1.02) if not before["fall"] else None
                current_window = next(r for r in published if Path(r["motion"]).stem == name)["windows"]["full_rollout"]
                old_window = next(r for r in read_json(baseline / "explicit_summary.json")["motions"]
                                  if Path(r["motion"]).stem == name)["windows"]["full_rollout"]
                world_ratios = {k: current_window[k] / max(old_window[k], 1e-12) for k in
                                ("root_pos_error_m_mean", "wrists_world_mean_m")}
                rows.append(dict(arm=arm, kind=kind, motion=name, baseline_rmse_rad=old_rmse,
                    candidate_rmse_rad=new_rmse, ratio=new_rmse / old_rmse, baseline_fall=bool(before["fall"]),
                    candidate_fall=fallen, retention_passed=retention, world_error_ratios=world_ratios))
                summaries.setdefault(kind, []).append((name, new_rmse, old_rmse, fallen, world_ratios))
        cloth = summaries["cloth"]
        per_arm[arm] = dict(
            six_cloth_mean_rad=float(np.mean([v[1] for v in cloth])),
            six_cloth_baseline_mean_rad=float(np.mean([v[2] for v in cloth])),
            cloth_falls=sum(1 for v in cloth if v[3]),
            official_world_ratios={k: float(np.mean([v[4][k] for v in summaries["official"]]))
                                   for k in ("root_pos_error_m_mean", "wrists_world_mean_m")},
            official_retention_gates=[r for r in rows if r["arm"] == arm and r["kind"] == "official"
                                      and r["retention_passed"] is not None],
            e36_style_development_gate=bool(
                len(cloth) == 6 and not any(v[3] for v in cloth) and
                per_arm_mean(cloth) <= float(np.mean([v[2] for v in cloth])) * .99),
        )
    means = {arm: per_arm[arm]["six_cloth_mean_rad"] for arm in P.ARMS}
    gap_b_a = means["A"] - means["B"]
    conclusion = dict(
        six_cloth_mean_rad=means,
        b_vs_a_gap_rad=gap_b_a, b_vs_a_gap_percent=gap_b_a / means["A"] * 100.0,
        projection_mechanism=("SUPPORT" if means["B"] < means["A"] and gap_b_a / means["A"] >= .03
                              else "NOT_SUPPORTED_BY_THIS_READING"),
        a_gate_hit_step=read_json(OUT / "train_arm_a.json")["gate_hit_step"],
        b_max_teacher_rmse=max(r["teacher_action_rmse"] for r in
            read_json(LOG / "arm_b" / "cloth_contract.json")["optimizer_steps"]),
        c_max_teacher_rmse=max(r["teacher_action_rmse"] for r in
            read_json(LOG / "arm_c" / "cloth_contract.json")["optimizer_steps"]),
        a_gate_rmse=read_json(OUT / "train_arm_a.json")["gate_row"]["teacher_action_rmse"],
        criteria=list(P.CRITERIA),
    )
    atomic_json(OUT / "effect_verification.json", dict(rows=rows, per_arm=per_arm,
                                                       conclusion=conclusion, generated_at=time.time()))
    print(json.dumps(conclusion, indent=1)[:1200], flush=True)


def per_arm_mean(cloth):
    import numpy as np
    return float(np.mean([v[1] for v in cloth]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "prepare", "train", "evaluate", "effect", "status"))
    parser.add_argument("--arm", choices=P.ARMS, default="A")
    parser.add_argument("--kind", choices=("cloth", "official"), default="cloth")
    args = parser.parse_args()
    if args.action == "register":
        register()
    elif args.action == "prepare":
        prepare_stage()
    elif args.action == "train":
        raise SystemExit(train_stage(args.arm))
    elif args.action == "evaluate":
        evaluate_stage(args.arm, args.kind)
    elif args.action == "effect":
        verify_effect_stage()
    else:
        print(json.dumps({p.name: read(p) for p in OUT.glob("*.json")}, indent=1)[:2000])


if __name__ == "__main__":
    main()
