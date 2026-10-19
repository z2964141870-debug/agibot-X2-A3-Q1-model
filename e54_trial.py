"""E54 trial driver: stats, register, prepare, two single-run arms (P/Q @16 updates), evaluate, effect.

Controller never imports tensor runtimes; worker stages run the guarded vendor
launcher. OPS writes only; evaluation writes go to the repo-side NEW namespace
(E52/E53 precedent, pre-registered).
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
TRAIN_ROOT = OPS / "data/training/a3_zhy_channel_probe_20261019_E54"
sys.path.insert(0, str(OPS))

ROOT = Path("/opt/data/private/hechengzhu/ZHY")
sys.path.insert(0, str(ROOT))

from script.a3 import zhy_trial as guard  # noqa: E402
from script.a3.checkpoint_store import atomic_json, sha256  # noqa: E402
from script.a3.host_health import append_record  # noqa: E402
import e54_protocol as P  # noqa: E402

EXPERIMENT = P.EXPERIMENT
NS = P.NS
OUT = OPS / ("data/experiments/" + NS)
LOG = OPS / ("logs/" + NS)
LAUNCHER = ROOT / "script/vendor/sonic_for_a3/train_a3_035_fromscratch.sh"
SOURCE = ROOT / P.SOURCE_035
ANCHOR = ROOT / "data/experiments/a3_zhy_gradient_20261012_E11/rollout.pt"
ANCHOR_RECEIPT = P.ANCHOR_RECEIPT
ANCHOR_TARGETS = P.ANCHOR_TARGETS
STATS = OUT / P.STATS_NAME
EVAL_ROOT = ROOT / ("data/experiments/" + NS)
EVAL_LOG_ROOT = ROOT / ("logs/" + NS)
BASELINE_CLOTH = ROOT / "data/experiments/a3_zhy_smpl_frozen_20261013_E26/evaluation"
BASELINE_OFFICIAL = ROOT / "data/experiments/a3_zhy_search_20261012_E02/official_oracle"
E53 = OPS / ("data/experiments/" + P.E53_NS)
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
LOCAL_CODE = ["e53_protocol.py", "e53_trainer.py", "e53_anchor_probe.py",
              "e54_protocol.py", "e54_obs_terms.py", "e54_channel_stats.py",
              "e54_trial.py", "test_e54_controls.py"]


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


def stats_stage():
    if STATS.exists():
        raise FileExistsError("E54 stats manifest must not be rebuilt")
    env = dict(os.environ)
    env.update(PYTHONPATH=str(OPS) + ":" + str(ROOT) + ":" + env.get("PYTHONPATH", ""),
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, str(OPS / "e54_channel_stats.py")],
                            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=900)
    (LOG.mkdir(parents=True, exist_ok=True) if not LOG.exists() else None)
    (LOG / "stats_build.log").write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        raise ValueError("E54 stats build failed; see stats_build.log")
    manifest = json.loads(STATS.read_text())
    official = [n for n in manifest if not n.startswith("G_")]
    if len(manifest) != 10 or len(official) != 4:
        raise ValueError("E54 stats manifest coverage wrong: %d entries" % len(manifest))
    atomic_json(LOG / "stats_receipt.json", dict(path=str(STATS), sha256=sha256(STATS),
                                                 motions=sorted(manifest), built_at=time.time()))
    print("E54 stats built:", len(manifest), "motions", flush=True)


def register():
    if (OUT / "registration.json").exists() or TRAIN_ROOT.exists():
        raise FileExistsError("E54 must not be reset or repeated")
    if not STATS.exists():
        raise ValueError("E54 stats manifest missing; run stats first")
    baseline = preflight()
    source_sha = sha256(SOURCE)
    inputs = {P.SOURCE_035: source_sha,
              "data/experiments/a3_zhy_gradient_20261012_E11/rollout.pt": sha256(ANCHOR),
              "yuanqi_ops/data/experiments/" + P.E53_NS + "/anchors_035.pt": sha256(ANCHOR_TARGETS)}
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
        contract="analysis/E54_CONTRACT.md v1 APPROVED by lead 2026-10-19 (amendments: eff_max_bound=min(1.165, 1.138*1.02)=1.1608 conjunction written into effect stage; t=0 anchor identity reused from E11/E53 anchor receipt)",
        hypothesis="channel family recalibration fixes E53-I official out-of-domain world drift without task regression: P per-motion standardization vs Q reference-error form",
        main_variable="6-channel feature family (P standardized / Q error-form) on the unchanged E53 arm-I slot-interleave mechanism",
        arms=dict(P.ARMS_NOTES), channels=P.CHANNELS, base_obs_dim=P.BASE_OBS_DIM,
        tiers=list(P.TIERS), updates_by_tier=P.UPDATES_BY_TIER,
        source_checkpoint=P.SOURCE_035, source_sha256=source_sha,
        stats_manifest=str(STATS), stats_sha256=sha256(STATS),
        appendix_b_baselines=P.WDR_BASELINES,
        e53_i8_cloth6=P.E53_I8_CLOTH6,
        eff_max_bound=P.EFF_MAX_BOUND, cloth6_bound=P.CLOTH6_BOUND,
        seed=0, num_envs=16, epochs=1, minibatches=4,
        fp32=True, frozen_codec_std=True, original_reward_routing=True,
        original_adaptive_lr_rule=True, actor_learning_rate=2e-6,
        teacher_mse="RECORDED_ONLY_NO_INTERCEPTION",
        runtime_limit_train_seconds=P.RUNTIME_LIMIT_TRAIN_SECONDS,
        runtime_limit_eval_seconds=P.RUNTIME_LIMIT_EVAL_SECONDS,
        memory_reserve_gib=P.MEMORY_RESERVE_GIB, max_cpu_c=P.MAX_CPU_C, max_gpu_c=P.MAX_GPU_C,
        boot_id=baseline[-1]["boot_id"],
        headroom_amendment="trial-side headroom estimator = limit - usage + total_inactive_file + total_active_file (E53 precedent); 8GiB threshold unchanged",
        baseline_oom_kill=baseline[-1]["container_memory"]["oom_kill"],
        entry_headroom_bytes=baseline[-1]["headroom_bytes"],
        input_sha256=inputs, code_sha256=code,
        writes_scope="OPS writes only; training stages repo-tree zero-write",
        evaluation_writes_amendment="evaluation outputs+logs under repo-side NEW namespace data/experiments|logs/" + NS + " only (E52/E53 precedent, pre-registered)",
        anchor_receipt_reuse="E53 anchors_035.pt + receipt reused (same mechanism geometry; channel family is downstream of the anchor block; zero-column t=0 gating unchanged)",
        single_run="each arm exactly one training launch at 16 updates; no re-run after any terminal status",
        candidate_deployment="FORBIDDEN_NEVER_REPLACE_035",
    )
    atomic_json(OUT / "registration.json", job)
    for value in baseline:
        append_record(LOG / "preflight.jsonl", value)
    print("E54 registered; 2 arms x 1 tier (16 updates) single-run", flush=True)


def prepare_stage():
    job = read(OUT / "registration.json")
    failures = []
    for name, identity in job["input_sha256"].items():
        if name.startswith("yuanqi_ops_data/"):
            path = OPS / name.split("/", 1)[1]
        elif name.startswith("yuanqi_ops/"):
            path = OPS / name.split("/", 1)[1]
        else:
            path = ROOT / name
        if not path.exists() or sha256(path) != identity:
            failures.append(name)
    if failures:
        raise ValueError("Registered input or code changed: " + repr(failures))
    if sha256(STATS) != job["stats_sha256"]:
        raise ValueError("Stats manifest changed since registration")
    receipt = read(ANCHOR_RECEIPT)
    if receipt.get("anchors_sha256") != sha256(ANCHOR_TARGETS):
        raise ValueError("E53 anchor receipt sha mismatch")
    env = dict(os.environ)
    env.update(PYTHONPATH=str(OPS) + ":" + str(ROOT) + ":" + env.get("PYTHONPATH", ""),
               PYTHONDONTWRITEBYTECODE="1", E54_CHANNEL_STATS=str(STATS))
    python = os.environ.get("E54_PYTHON", str(ROOT / "data/environments/a3-sonic/bin/python"))
    result = subprocess.run([python, str(OPS / "test_e54_controls.py")],
                            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600)
    (LOG / "control_tests.log").write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        raise ValueError("E54 control tests failed; see control_tests.log")
    atomic_json(OUT / "prepare.json", dict(status="PREPARED", control_tests="PASS",
                                           anchor_receipt="REUSED_E53", tested_at=time.time()))
    print("E54 prepare complete; controls pass, anchor receipt reused", flush=True)


def cell_launch_argv(arm, tier):
    argv = ["bash", str(LAUNCHER),
            "seed=0",
            "trainer._target_=" + P.ARMS_TRAINER[arm],
            "algo.config.num_learning_epochs=1",
            "algo.config.actor_learning_rate=2e-6",
            "algo.config.adaptive_lr_min=1e-6",
            "algo.config.adaptive_lr_max=2e-5",
            "callbacks.model_save._target_=script.a3.finite_retention_trainer.FiniteCheckpointCallback",
            "callbacks.model_save.save_frequency=" + str(tier),
            "+callbacks.model_save.target_global_step=" + str(tier),
            "++algo.trl.bf16=false", "++algo.trl.fp16=false", "++algo.trl.tf32=false",
            "manager_env.observations.tokenizer.motion_anchor_ori_b_mf_nonflat.func=" + P.FUNC_BY_ARM[arm],
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
        YUANQI_E54_ARM=arm, YUANQI_E54_TIER=str(tier),
        YUANQI_E53_SOURCE_CHECKPOINT=str(SOURCE),
        E54_CHANNEL_STATS=str(STATS),
        NUM_ENVS="16", NUM_MINI_BATCHES="4", NUM_PROCESSES="1",
        NUM_LEARNING_ITERATIONS=str(tier),
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        WANDB_MODE="disabled", ACCELERATE_MIXED_PRECISION="no",
    )
    atomic_json(OUT / ("training_launch_" + run_cell(arm, tier) + ".json"),
                dict(argv=cell_launch_argv(arm, tier), motion_file=str(run_dir / "motionlib"),
                     checkpoint=str(SOURCE), num_learning_iterations=tier,
                     obs_func=P.FUNC_BY_ARM[arm], channel_stats=str(STATS),
                     wrapper="g1_anchor_term_channel_concat_e54"))
    for rel in MOTIONLIB_SOURCES:
        target = run_dir / "motionlib" / Path(rel).name
        if not target.exists():
            target.symlink_to(str(ROOT / rel))
    console = (log_dir / "console.log").open("a")
    console.write("== E54 %s start %s ==\n" % (run_cell(arm, tier), time.strftime("%Y-%m-%dT%H:%M:%S%z")))
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
    print("E54", run_cell(arm, tier), "status", status, flush=True)
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
    env["E54_CHANNEL_STATS"] = str(STATS)
    env["YUANQI_E54_ARM"] = arm
    saved = {k: os.environ.get(k) for k in ("YUANQI_E53_WRAPPER", "PYTHONPATH", "E54_CHANNEL_STATS")}
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
            raise RuntimeError("E54 %s %s evaluation failed; no repeat" % (run_cell(arm, tier), kind))
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
                     writes_scope="NEW_NAMESPACE_ONLY_repo_data_experiments_" + NS))
    print("E54", run_cell(arm, tier), kind, "evaluation complete", flush=True)


def effect_stage():
    import numpy as np

    def world_means(trace):
        ref = np.asarray(trace["ref_key_body_pos_w"], dtype=float)
        sim = np.asarray(trace["sim_key_body_pos_w"], dtype=float)
        err = np.linalg.norm(sim - ref, axis=-1)
        names = trace["key_body_names"]
        torso = [i for i, n in enumerate(names) if ("torso" in n or "pelvis" in n)]
        wrists = [i for i, n in enumerate(names) if "wrist" in n]
        legs = [i for i, n in enumerate(names) if ("ankle" in n or "knee" in n or "foot" in n or "hip" in n)]
        return float(err[:, torso].mean()), float(err[:, wrists].mean()), float(err[:, legs].mean())

    candidates = {}
    per_motion = {}
    for arm in P.ARMS:
        tier = P.TIERS[0]
        cell = run_cell(arm, tier)
        cloth_vals, wdrs = [], {}
        for kind in ("cloth", "official"):
            directory = EVAL_ROOT / ("eval_" + cell + "_" + kind)
            published = read(directory / "explicit_summary.json")["motions"]
            for record in published:
                name = Path(record["motion"]).stem
                metrics = read(directory / (name + ".metrics.json"))["motions"][0]
                trace = read(directory / (name + ".metrics.json").replace(".metrics.json", ".timeseries.json"))["motions"][0]
                q, target = np.asarray(trace["q_state_29"]), np.asarray(trace["reference_q_29"])
                rmse = float(np.sqrt(np.mean((q - target) ** 2)))
                falling = np.flatnonzero((np.asarray(trace["root_height"]) < .45) |
                                         (np.abs(trace["root_roll_deg"]) > 60.) |
                                         (np.abs(trace["root_pitch_deg"]) > 60.))
                fallen = bool(len(falling))
                if fallen != metrics["fall"]:
                    raise ValueError("Independent fall reconstruction failed for " + name)
                if kind == "cloth":
                    cloth_vals.append(rmse)
                else:
                    torso, wrists, legs = world_means(trace)
                    wdrs[name] = P.wdr(torso, wrists, legs)
                per_motion.setdefault(cell, {})[name] = dict(kind=kind, rmse_rad=rmse, fall=fallen)
        candidates[arm] = dict(wdr=dict(wdrs, max=float(max(wdrs.values()))),
                               cloth6=float(np.mean(cloth_vals)))
    passes, selection = P.verdict(candidates)
    result = dict(experiment=EXPERIMENT, candidates=candidates, passes=passes, selection=selection,
                  pinned_bounds=dict(eff_max_bound=P.EFF_MAX_BOUND, cloth6_bound=P.CLOTH6_BOUND,
                                     e53_i8_cloth6=P.E53_I8_CLOTH6),
                  baselines=P.WDR_BASELINES, per_motion=per_motion,
                  criteria=["primary: official4 wdr max <= %.4f (conjunction of 1.294*(1-0.10) and N8*1.02)" % P.EFF_MAX_BOUND,
                            "secondary: cloth6 <= %.6f (E53-I8 * 1.05)" % P.CLOTH6_BOUND,
                            "selection: min wdr max, tie <=2% -> P; single/none -> record + escalate"],
                  generated_at=time.time())
    atomic_json(OUT / "effect_verification.json", result)
    print(json.dumps(dict(candidates={a: dict(wdr_max=passes[a]["wdr_max"], cloth6=passes[a]["cloth6"],
                                              both_pass=passes[a]["both_pass"]) for a in P.ARMS},
                          selection=selection), indent=1)[:1500], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("stats", "register", "prepare", "train", "evaluate", "effect"))
    parser.add_argument("--arm", choices=P.ARMS, default="P")
    parser.add_argument("--tier", type=int, choices=P.TIERS, default=16)
    parser.add_argument("--kind", choices=("cloth", "official"), default="cloth")
    args = parser.parse_args()
    if args.action == "stats":
        stats_stage()
    elif args.action == "register":
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
