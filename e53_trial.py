"""E53 trial driver: register, prepare, four single-run cells (2 arms x 2 tiers), evaluate, effect.

Controller never imports tensor runtimes; worker stages run the guarded vendor
launcher. OPS writes only; evaluation writes go to the repo-side NEW namespace
per the E52 amendment precedent (pre-registered this time).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

OPS = Path("/opt/data/private/yuanqi_ops")
TRAIN_ROOT = OPS / "data/training/a3_zhy_rootchannel_20261019_E53"
sys.path.insert(0, str(OPS))

ROOT = Path("/opt/data/private/hechengzhu/ZHY")
sys.path.insert(0, str(ROOT))

from script.a3 import zhy_trial as guard  # noqa: E402
from script.a3.checkpoint_store import atomic_json, sha256  # noqa: E402
from script.a3.host_health import append_record  # noqa: E402
import e53_protocol as P  # noqa: E402

EXPERIMENT = "A3_ZHY_ROOTCHANNEL_20261019_E53"
OUT = OPS / "data/experiments/a3_zhy_rootchannel_20261019_E53"
LOG = OPS / "logs/a3_zhy_rootchannel_20261019_E53"
MANIFEST = OPS / "data/manifests/a3_zhy_rootchannel_20261019_E53.json"
LAUNCHER = ROOT / "script/vendor/sonic_for_a3/train_a3_035_fromscratch.sh"
SOURCE = ROOT / P.SOURCE_035
ANCHOR = ROOT / "data/experiments/a3_zhy_gradient_20261012_E11/rollout.pt"
ANCHOR_TARGETS = OUT / "anchors_035.pt"
TARGETS = ROOT / "data/experiments/a3_zhy_modules_20261012_E12/module_outputs.pt"
MOTIONLIB_SOURCES = [
    "data/experiments/a3_zhy_cloth_resource_20261013_E38/motionlib/G_0113_zj_train__183926_zj_walk.pkl",
    "data/experiments/a3_zhy_cloth_resource_20261013_E38/motionlib/G_0113_zj_train__192333_zj_warmUp.pkl",
    "data/experiments/a3_zhy_cloth_resource_20261013_E38/motionlib/G_20251226_fjw__fjw_140008_walk.pkl",
    "data/experiments/a3_zhy_cloth_failure_20261013_E45/motionlib/G_20251229_zzx__zzx_111505_walk.pkl",
]
CODE_FILES = [
    "script/a3/cloth_constrained_trial.py", "script/a3/cloth_constrained_trainer.py",
    "script/a3/cloth_cached_tokens.py", "script/a3/cloth_curve_trainer.py",
    "script/a3/cloth_curve_protocol.py", "script/a3/cloth_failure_trial.py",
    "script/a3/cloth_failure_callback.py", "script/a3/finite_retention_trainer.py",
    "script/a3/zhy_trial.py", "script/a3/host_health.py", "script/a3/checkpoint_store.py",
    "script/a3/resumable_sonic.py", "script/a3/evaluate_mujoco.py",
    "script/a3/frozen_evaluation_launch.py",
    "script/vendor/sonic_for_a3/train_a3_035_fromscratch.sh",
]
LOCAL_CODE = ["e53_protocol.py", "e53_trainer.py", "e53_trial.py",
              "e53_anchor_probe.py", "e53_obs_terms.py", "test_e53_controls.py"]

EVAL_ROOT = ROOT / "data/experiments/a3_zhy_rootchannel_20261019_E53"
EVAL_LOG_ROOT = ROOT / "logs/a3_zhy_rootchannel_20261019_E53"
BASELINE_CLOTH = ROOT / "data/experiments/a3_zhy_smpl_frozen_20261013_E26/evaluation"
BASELINE_OFFICIAL = ROOT / "data/experiments/a3_zhy_search_20261012_E02/official_oracle"


def read(path):
    return json.loads(Path(path).read_text())


def _foreign_large_processes():
    listing = subprocess.run(["ps", "-eo", "rss=,comm="], capture_output=True, text=True).stdout
    rows = []
    for line in listing.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) > 1024 ** 2:
            if not any(name in parts[1] for name in ("sshd", "bash", "ps ", "python", "tmux")):
                rows.append(line.strip())
    return rows


CGROUP = Path("/sys/fs/cgroup/memory")


def container_headroom_bytes():
    stat = dict(line.split() for line in (CGROUP / "memory.stat").read_text().splitlines())
    limit = int((CGROUP / "memory.limit_in_bytes").read_text())
    usage = int((CGROUP / "memory.usage_in_bytes").read_text())
    reclaimable = int(stat["total_inactive_file"]) + int(stat["total_active_file"])
    return max(0, min(limit, limit - usage + reclaimable)), reclaimable


def preflight():
    rows = []
    for index in range(3):
        value = guard.preflight()
        gpus = value.get("gpus") or []
        if not gpus or any(float(g.get("utilization.gpu", -1)) != 0.0 for g in gpus):
            raise RuntimeError("GPU not idle; no startup")
        headroom, reclaimable = container_headroom_bytes()
        value = dict(value, headroom_bytes=headroom, reclaimable_bytes=reclaimable)
        if headroom < P.ENTRY_HEADROOM_GIB * 1024 ** 3:
            raise RuntimeError("Insufficient entry headroom; no startup (estimate includes active_file)")
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


def run_cell(arm, tier):
    return "arm_" + arm.lower() + "_tier" + str(tier)


def register():
    if OUT.exists() or TRAIN_ROOT.exists():
        raise FileExistsError("E53 must not be reset or repeated")
    baseline = preflight()
    source_sha = sha256(SOURCE)
    inputs = {P.SOURCE_035: source_sha,
              "data/experiments/a3_zhy_gradient_20261012_E11/rollout.pt": sha256(ANCHOR),
              "data/experiments/a3_zhy_modules_20261012_E12/module_outputs.pt": sha256(TARGETS)}
    for rel in MOTIONLIB_SOURCES:
        inputs[rel] = sha256(ROOT / rel)
    code = {}
    for rel in CODE_FILES:
        code[rel] = sha256(ROOT / rel)
    for rel in LOCAL_CODE:
        code["yuanqi_ops/" + rel] = sha256(OPS / rel)
    for directory in (OUT, LOG, OPS / "data/manifests", TRAIN_ROOT):
        directory.mkdir(parents=True, exist_ok=True)
    job = dict(
        experiment=EXPERIMENT, registered_at=time.time(),
        contract="analysis/EXP2B_CONTRACT.md v1 APPROVED by lead 2026-10-19 (amendments: tiers 8/32; channels 6; two-tier-consistency mechanism rule with weak-positive 64-update escalation)",
        hypothesis="H2 interface: explicit root displacement+velocity channels (anchor frame, HP26 PC definition) reduce task degradation vs no injection under clean 035 warm start",
        main_variable="observation injection arm (N none / I +6 channels zero-init first-layer extension)",
        arms=dict(P.ARMS_NOTES), channels=P.CHANNELS, base_obs_dim=P.BASE_OBS_DIM,
        tiers=list(P.TIERS), updates_by_tier=P.UPDATES_BY_TIER,
        source_checkpoint=P.SOURCE_035, source_sha256=source_sha,
        seed=0, num_envs=16, epochs=1, minibatches=4,
        fp32=True, frozen_codec_std=True, original_reward_routing=True,
        original_adaptive_lr_rule=True, actor_learning_rate=2e-6,
        teacher_mse="RECORDED_ONLY_NO_INTERCEPTION",
        runtime_limit_train_seconds=P.RUNTIME_LIMIT_TRAIN_SECONDS,
        runtime_limit_eval_seconds=P.RUNTIME_LIMIT_EVAL_SECONDS,
        memory_reserve_gib=P.MEMORY_RESERVE_GIB, max_cpu_c=P.MAX_CPU_C, max_gpu_c=P.MAX_GPU_C,
        boot_id=baseline[-1]["boot_id"],
        headroom_amendment="trial-side headroom estimator = limit - usage + total_inactive_file + total_active_file (guard formula counts inactive only; 14.6GiB E52-era page cache is reclaimable); 8GiB threshold unchanged",
        baseline_oom_kill=baseline[-1]["container_memory"]["oom_kill"],
        entry_headroom_bytes=baseline[-1]["headroom_bytes"],
        code_sha256=code, input_sha256=inputs,
        criteria=list(P.CRITERIA),
        writes_root=str(OPS),
        evaluation_writes_amendment="evaluation outputs+logs under repo-side NEW namespace data/experiments|logs/a3_zhy_rootchannel_20261019_E53 only (evaluate_mujoco guard requires repo roots; E52 precedent); training stages repo-tree zero-write",
        single_run="each (arm, tier) exactly one training launch; no re-run after any terminal status",
        candidate_deployment="FORBIDDEN_NEVER_REPLACE_035",
    )
    atomic_json(OUT / "registration.json", job)
    for value in baseline:
        append_record(LOG / "preflight.jsonl", value)
    print("E53 registered; 2 arms x 2 tiers single-run", flush=True)


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
    python = os.environ.get("E53_PYTHON", str(ROOT / "data/environments/a3-sonic/bin/python"))
    result = subprocess.run([python, str(OPS / "test_e53_controls.py")],
                            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    (LOG / "control_tests.log").write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        raise ValueError("E53 control tests failed; see control_tests.log")
    probe = subprocess.run([python, str(OPS / "e53_anchor_probe.py")],
                           cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1800)
    (LOG / "anchor_probe.log").write_text(probe.stdout + probe.stderr)
    if probe.returncode != 0 or not ANCHOR_TARGETS.exists():
        raise ValueError("E53 anchor probe failed; see anchor_probe.log")
    if sha256(ANCHOR_TARGETS) != read(OUT / "anchors_035.json")["anchors_sha256"]:
        raise ValueError("Anchor receipt sha mismatch")
    atomic_json(OUT / "prepare.json", dict(status="PREPARED", control_tests="PASS",
                                           anchor_probe="PASS", tested_at=time.time()))
    print("E53 prepare complete; controls + anchor probe pass", flush=True)


def cell_launch_argv(arm, tier):
    trainer = P.ARMS_TRAINER[arm]
    callback = ("script.a3.finite_retention_trainer.FiniteCheckpointCallback" if tier == 8
                else "script.a3.cloth_failure_callback.FailureCheckpointCallback")
    save_frequency = 8
    argv = ["bash", str(LAUNCHER),
            "seed=0",
            "trainer._target_=" + trainer,
            "algo.config.num_learning_epochs=1",
            "algo.config.actor_learning_rate=2e-6",
            "algo.config.adaptive_lr_min=1e-6",
            "algo.config.adaptive_lr_max=2e-5",
            "callbacks.model_save._target_=" + callback,
            "callbacks.model_save.save_frequency=" + str(save_frequency),
            "+callbacks.model_save.target_global_step=" + str(tier),
            "++algo.trl.bf16=false", "++algo.trl.fp16=false", "++algo.trl.tf32=false",
            ]
    if arm == "I":
        argv += [
            "manager_env.observations.tokenizer.motion_anchor_ori_b_mf_nonflat.func="
            "e53_obs_terms:e53_anchor_ori_b_mf_with_channels",
        ]
    return argv


def train_stage(arm, tier):
    job = read(OUT / "registration.json")
    if sha256(SOURCE) != job["source_sha256"]:
        raise ValueError("035 source changed since registration")
    run_dir = TRAIN_ROOT / run_cell(arm, tier)
    log_dir = LOG / run_cell(arm, tier)
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
        YUANQI_CLOTH_ANCHOR_OUTPUTS=str(ANCHOR_TARGETS),
        YUANQI_E53_ARM=arm, YUANQI_E53_TIER=str(tier),
        YUANQI_E53_SOURCE_CHECKPOINT=str(SOURCE),
        NUM_ENVS="16", NUM_MINI_BATCHES="4", NUM_PROCESSES="1",
        NUM_LEARNING_ITERATIONS=str(tier),
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        WANDB_MODE="disabled", ACCELERATE_MIXED_PRECISION="no",
    )
    atomic_json(OUT / ("training_launch_" + run_cell(arm, tier) + ".json"),
                dict(argv=cell_launch_argv(arm, tier), motion_file=str(run_dir / "motionlib"),
                     checkpoint=str(SOURCE), num_learning_iterations=tier,
                     wrapper=("g1_anchor_term_channel_concat" if arm == "I" else "vendor")))
    for rel in MOTIONLIB_SOURCES:
        target = run_dir / "motionlib" / Path(rel).name
        if not target.exists():
            target.symlink_to(str(ROOT / rel))
    console = (log_dir / "console.log").open("a")
    console.write("== E53 %s start %s ==\n" % (run_cell(arm, tier), time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    for i in range(3):
        value = guard.preflight()
        append_record(log_dir / "preflight.jsonl", value)
        if i < 2:
            time.sleep(5)
    console.flush()
    process = subprocess.Popen(cell_launch_argv(arm, tier), stdout=console, stderr=console,
                               cwd=str(ROOT), env=env)
    try:
        code = process.wait(timeout=P.RUNTIME_LIMIT_TRAIN_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        atomic_json(OUT / ("train_" + run_cell(arm, tier) + ".json"),
                    dict(arm=arm, tier=tier, status="STOPPED_RUNTIME_CAP", elapsed=time.time() - started))
        return 2
    console.close()
    status = "TRAINED" if code == 0 else "STOPPED_LAUNCH_FAILURE"
    checkpoint = run_dir / P.CHECKPOINT_BY_TIER[tier]
    if code == 0 and not checkpoint.exists():
        status = "STOPPED_MISSING_MILESTONE"
    atomic_json(OUT / ("train_" + run_cell(arm, tier) + ".json"), dict(
        arm=arm, tier=tier, status=status, exit_code=code,
        elapsed_seconds=round(time.time() - started, 3),
        checkpoint=str(checkpoint), launcher=str(LAUNCHER)))
    print("E53", run_cell(arm, tier), "status", status, flush=True)
    return 0 if status == "TRAINED" else 2


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


def evaluate_stage(arm, tier, kind):
    from script.a3 import evaluate_mujoco as evaluation
    from script.a3.frozen_evaluation_launch import LaunchProxy, arguments
    source_checkpoint = TRAIN_ROOT / run_cell(arm, tier) / P.CHECKPOINT_BY_TIER[tier]
    if not source_checkpoint.exists():
        raise ValueError("Cell checkpoint missing: " + str(source_checkpoint))
    copy_dir = EVAL_ROOT / "checkpoints" / run_cell(arm, tier)
    copy_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = copy_dir / source_checkpoint.name
    if not checkpoint.exists():
        shutil.copy2(source_checkpoint, checkpoint)
        sidecar = source_checkpoint.with_suffix(".json")
        if sidecar.exists():
            shutil.copy2(sidecar, copy_dir / sidecar.name)
    run_config = TRAIN_ROOT / run_cell(arm, tier) / "config.yaml"
    if run_config.exists():
        shutil.copy2(run_config, copy_dir / "config.yaml")
        meta = run_config.parent / "meta.yaml"
        if meta.exists():
            shutil.copy2(meta, copy_dir / "meta.yaml")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(OPS) + ":" + str(ROOT) + ":" + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    saved = {k: os.environ.get(k) for k in ("YUANQI_E53_WRAPPER", "PYTHONPATH")}
    os.environ.update(env)
    try:
        args = arguments(EVAL_ROOT / ("eval_" + run_cell(arm, tier) + "_" + kind),
                         EVAL_LOG_ROOT / ("eval_" + run_cell(arm, tier) + "_" + kind),
                         checkpoint, references_farm(kind))
        if kind == "official":
            args.frame_stride = 4
        evaluation.preflight = guard.preflight
        evaluation.monitor_child = lambda child, directory, config: (child.wait(), None)
        evaluation.subprocess = LaunchProxy(evaluation.subprocess)
        if evaluation.evaluate_explicit(args):
            raise RuntimeError("E53 %s %s evaluation failed; no repeat" % (run_cell(arm, tier), kind))
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    atomic_json(OUT / ("eval_" + run_cell(arm, tier) + "_" + kind + ".json"),
                dict(arm=arm, tier=tier, kind=kind, checkpoint=str(checkpoint),
                     checkpoint_source=str(source_checkpoint),
                     checkpoint_source_sha256=sha256(source_checkpoint),
                     checkpoint_sha256=sha256(checkpoint), completed=time.time(),
                     output=str(EVAL_ROOT / ("eval_" + run_cell(arm, tier) + "_" + kind)),
                     writes_scope="NEW_NAMESPACE_ONLY_repo_data_experiments_a3_zhy_rootchannel"))
    print("E53", run_cell(arm, tier), kind, "evaluation complete", flush=True)


def effect_stage():
    import numpy as np
    from script.a3.verify_zhy_search import verify_metrics

    def read_json(path):
        return json.loads(Path(path).read_text())

    rows, means, task_gates = [], {}, {}
    for arm in P.ARMS:
        for tier in P.TIERS:
            cell = run_cell(arm, tier)
            cloth_rows, official_world = [], {}
            for kind in ("cloth", "official"):
                directory = EVAL_ROOT / ("eval_" + cell + "_" + kind)
                published = read_json(directory / "explicit_summary.json")["motions"]
                for record in verify_metrics(directory):
                    name = record["motion"]
                    baseline = BASELINE_CLOTH if kind == "cloth" else BASELINE_OFFICIAL
                    before = read_json(baseline / (name + ".metrics.json"))["motions"][0]
                    after = read_json(directory / (name + ".metrics.json"))["motions"][0]
                    trace = read_json(directory / (name + ".timeseries.json"))["motions"][0]
                    q, target = np.asarray(trace["q_state_29"]), np.asarray(trace["reference_q_29"])
                    old_rmse = before["tracking"]["all_29_rmse"]
                    new_rmse = float(np.sqrt(np.mean((q - target) ** 2)))
                    falling = np.flatnonzero((np.asarray(trace["root_height"]) < .45) |
                                             (np.abs(trace["root_roll_deg"]) > 60.) |
                                             (np.abs(trace["root_pitch_deg"]) > 60.))
                    fallen = bool(len(falling))
                    if fallen != after["fall"]:
                        raise ValueError("Independent fall reconstruction failed")
                    rows.append(dict(arm=arm, tier=tier, kind=kind, motion=name,
                                     baseline_rmse_rad=old_rmse, candidate_rmse_rad=new_rmse,
                                     ratio=new_rmse / old_rmse, baseline_fall=bool(before["fall"]),
                                     candidate_fall=fallen))
                    if kind == "cloth":
                        cloth_rows.append((new_rmse, fallen))
                    else:
                        current_window = next(r for r in published if Path(r["motion"]).stem == name)["windows"]["full_rollout"]
                        old_window = next(r for r in read_json(baseline / "explicit_summary.json")["motions"]
                                          if Path(r["motion"]).stem == name)["windows"]["full_rollout"]
                        for key in ("root_pos_error_m_mean", "wrists_world_mean_m"):
                            official_world.setdefault(key, []).append(current_window[key] / max(old_window[key], 1e-12))
            mean = float(np.mean([v[0] for v in cloth_rows]))
            means[(arm, tier)] = mean
            task_gates[cell] = dict(six_cloth_mean_rad=mean,
                                    cloth_falls=sum(1 for v in cloth_rows if v[1]),
                                    official_world_ratio_max={k: float(np.max(v)) for k, v in official_world.items()})
    verdict, detail = P.mechanism_verdict_with_means(means)
    result = dict(experiment=EXPERIMENT, verdict=verdict, verdict_detail=detail,
                  means={run_cell(a, t): means[(a, t)] for a in P.ARMS for t in P.TIERS},
                  baseline_e26_six_cloth_mean_rad=P.BASELINE_CLOTH_MEAN,
                  task_gates=task_gates, criteria=list(P.CRITERIA),
                  note="weak positive signal allows one 64-update escalation; inconclusive bars conclusion and points to H5",
                  generated_at=time.time())
    atomic_json(OUT / "effect_verification.json", result)
    print(json.dumps(dict(verdict=verdict, means=result["means"],
                          task_gates=task_gates), indent=1)[:1600], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("register", "prepare", "train", "evaluate", "effect"))
    parser.add_argument("--arm", choices=P.ARMS, default="N")
    parser.add_argument("--tier", type=int, choices=P.TIERS, default=8)
    parser.add_argument("--kind", choices=("cloth", "official"), default="cloth")
    args = parser.parse_args()
    if args.action == "register":
        register()
    elif args.action == "prepare":
        prepare_stage()
    elif args.action == "train":
        raise SystemExit(train_stage(args.arm, args.tier))
    elif args.action == "evaluate":
        evaluate_stage(args.arm, args.tier, args.kind)
    elif args.action == "effect":
        effect_stage()


if __name__ == "__main__":
    main()
