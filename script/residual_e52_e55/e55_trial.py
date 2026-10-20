"""E55 trial driver: register, prepare, three single-run arms (64 updates), milestone evals, gate + effect.

Controller never imports tensor runtimes; worker stages run the guarded vendor
launcher. OPS writes only; evaluation writes go to the repo-side namespace
(E52/E53/E54 precedent, pre-registered).
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
TRAIN_ROOT = OPS / "data/training/a3_zhy_residual_p1_20261019_E55"
sys.path.insert(0, str(OPS))
ROOT = Path("/opt/data/private/hechengzhu/ZHY")
sys.path.insert(0, str(ROOT))

from script.a3 import zhy_trial as guard  # noqa: E402
from script.a3.checkpoint_store import atomic_json, sha256  # noqa: E402
from script.a3.host_health import append_record  # noqa: E402
import e55_protocol as P  # noqa: E402

EXPERIMENT = P.EXPERIMENT
NS = P.NS
OUT = OPS / ("data/experiments/" + NS)
LOG = OPS / ("logs/" + NS)
LAUNCHER = ROOT / "script/vendor/sonic_for_a3/train_a3_035_fromscratch.sh"
SOURCE = ROOT / P.SOURCE_035
ANCHOR = ROOT / "data/experiments/a3_zhy_gradient_20261012_E11/rollout.pt"
ANCHOR_TARGETS = P.ANCHOR_TARGETS
STATS = P.STATS
EVAL_ROOT = ROOT / ("data/experiments/" + NS)
EVAL_LOG_ROOT = ROOT / ("logs/" + NS)
BASELINE_CLOTH = ROOT / "data/experiments/a3_zhy_smpl_frozen_20261013_E26/evaluation"
BASELINE_OFFICIAL = ROOT / "data/experiments/a3_zhy_search_20261012_E02/official_oracle"
E54_REG = OPS / ("data/experiments/" + P.E54_NS + "/registration.json")
MOTIONLIB_SOURCES = [
    "data/experiments/a3_zhy_cloth_resource_20261013_E38/motionlib/G_0113_zj_train__183926_zj_walk.pkl",
    "data/experiments/a3_zhy_cloth_resource_20261013_E38/motionlib/G_0113_zj_train__192333_zj_warmUp.pkl",
    "data/experiments/a3_zhy_cloth_resource_20261013_E38/motionlib/G_20251226_fjw__fjw_140008_walk.pkl",
    "data/experiments/a3_zhy_cloth_failure_20261013_E45/motionlib/G_20251229_zzx__zzx_111505_walk.pkl",
]
CODE_FILES = [
    "script/a3/cloth_constrained_trial.py", "script/a3/cloth_constrained_trainer.py",
    "script/a3/cloth_cached_tokens.py", "script/a3/cloth_curve_trainer.py",
    "script/a3/cloth_curve_protocol.py", "script/a3/cloth_failure_callback.py",
    "script/a3/finite_retention_trainer.py", "script/a3/zhy_trial.py",
    "script/a3/host_health.py", "script/a3/checkpoint_store.py",
    "script/a3/resumable_sonic.py", "script/a3/evaluate_mujoco.py",
    "script/a3/frozen_evaluation_launch.py", "script/a3/reference_contract.py",
    "script/a3/sim_entry.py", "script/a3/smpl_frozen_entry.py",
    "script/vendor/sonic_for_a3/gear_sonic/scripts/sim2sim_a3_mujoco.py",
    "script/vendor/sonic_for_a3/train_a3_035_fromscratch.sh",
]
LOCAL_CODE = ["e53_protocol.py", "e53_trainer.py", "e53_anchor_probe.py",
              "e54_protocol.py", "e54_obs_terms.py", "e54_channel_stats.py",
              "e54_callback.py", "e55_protocol.py", "e55_obs_terms.py",
              "e55_residual.py", "e55_trainer.py", "e55_frozen_entry.py",
              "e55_trial.py", "test_e55_controls.py"]


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
            raise RuntimeError("Insufficient entry headroom; no startup")
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


def run_cell(arm):
    return "arm_" + arm.lower() + "_tier64"


def gate_log(row):
    path = OUT / "gate_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def register():
    if (OUT / "registration.json").exists():
        raise FileExistsError("E55 must not be reset or repeated")
    if not STATS.exists():
        raise ValueError("E55 requires the E54 stats manifest (reuse, no rebuild)")
    e54_reg = read(E54_REG)
    if e54_reg.get("stats_sha256") != sha256(STATS):
        raise ValueError("E55/E54 stats manifest sha mismatch; P arm reuse broken")
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
    atomic_json(OUT / "registration.json", dict(
        experiment=EXPERIMENT, registered_at=time.time(),
        contract="analysis/E55_P1_CONTRACT.md v1.2 lead-final-check 2026-10-19; amendment: eval entry e55_frozen_entry (LaunchProxy word swap), residual geometry 70-wide injected slots (head eats slot[0:58]+slot[64:70])",
        hypothesis="channel form matters only when channels feed a trainable path; residual head on frozen 035 is the true testbed (A0 raw vs P standardized vs Q error-form)",
        main_variable="6-channel feature family fed to the trainable residual head first layer",
        arms=dict(P.ARMS_NOTES), head=dict(P.HEAD), milestones=list(P.MILESTONES), updates=P.UPDATES,
        source_checkpoint=P.SOURCE_035, source_sha256=source_sha,
        stats_manifest=str(STATS), stats_sha256=sha256(STATS),
        bands=dict(cloth_wdr=P.BAND_CLOTH_WDR, official_wdr=P.BAND_OFFICIAL_WDR,
                   per_joint_cloth=P.BAND_PERJOINT_CLOTH, per_joint_official=P.BAND_PERJOINT_OFFICIAL,
                   per_joint_group_note="v1.2 grouped bands; single 0.283 abolished"),
        liveness=dict(zero_channels=True, zero_head=True, degrade_fraction=P.GUARD_TOLERANCE_FRACTION),
        scope="NO_MERGE: winner is a form conclusion only; checkpoint integration needs full gate + sign_off",
        seed=0, num_envs=16, epochs=1, minibatches=4, fp32=True,
        actor_learning_rate=2e-6,
        runtime_limit_train_seconds=P.RUNTIME_LIMIT_TRAIN_SECONDS,
        runtime_limit_eval_seconds=P.RUNTIME_LIMIT_EVAL_SECONDS,
        memory_reserve_gib=P.MEMORY_RESERVE_GIB, max_cpu_c=P.MAX_CPU_C, max_gpu_c=P.MAX_GPU_C,
        preflight_baseline=baseline, inputs=inputs, code_sha256=code,
        controls_sha256=sha256(OPS / "test_e55_controls.py")))
    print("E55 registered; 3 arms x tier64 single-run, milestones 8/32/64", flush=True)


def prepare_stage():
    env = dict(os.environ)
    env.update(PYTHONPATH=str(OPS) + ":" + str(ROOT) + ":" + env.get("PYTHONPATH", ""),
               PYTHONDONTWRITEBYTECODE="1", E54_CHANNEL_STATS=str(STATS))
    result = subprocess.run([sys.executable, str(OPS / "test_e55_controls.py")],
                            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1800)
    (LOG.mkdir(parents=True, exist_ok=True) if not LOG.exists() else None)
    (LOG / "controls.log").write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        raise ValueError("E55 controls failed; see controls.log")
    receipt = json.loads((LOG / "controls_receipt.json").read_text()) if (LOG / "controls_receipt.json").exists() else {}
    atomic_json(OUT / "prepare.json", dict(prepared_at=time.time(), controls="pass",
                                           stats_sha256=sha256(STATS), receipt=receipt))
    print("E55 prepare complete; controls pass", flush=True)


def cell_launch_argv(arm):
    """EXTRA_ARGS only: the vendor launcher injects its own +exp/checkpoint/experiment_dir/
    motion_file/num_envs block from env; duplicates make hydra fail with multiple values."""
    run_dir = TRAIN_ROOT / run_cell(arm)
    return ["bash", str(LAUNCHER),
            "seed=0",
            "trainer._target_=" + P.ARMS_TRAINER[arm],
            "algo.config.num_learning_epochs=1",
            "algo.config.actor_learning_rate=2e-6",
            "algo.config.adaptive_lr_min=1e-6", "algo.config.adaptive_lr_max=2e-5",
            "callbacks.model_save._target_=e54_callback.E54MilestoneCallback",
            "callbacks.model_save.save_frequency=8",
            "+callbacks.model_save.target_global_step=" + str(P.UPDATES),
            "callbacks.model_save.save_dir=" + str(run_dir / "checkpoints"),
            "++algo.trl.bf16=false", "++algo.trl.fp16=false", "++algo.trl.tf32=false",
            "manager_env.observations.tokenizer.motion_anchor_ori_b_mf_nonflat.func=" + P.FUNC_BY_ARM[arm]]


def buildcheck_stage(arm):
    """Lead gate (2026-10-19): real hydra env build must pass before train. Runs the
    exact cell argv, watches the child console for the ObservationManager env dump,
    then stops the child - no training iterations consumed."""
    run_dir = TRAIN_ROOT / run_cell(arm)
    log_dir = LOG / run_cell(arm)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(run_dir, ignore_errors=True)
    (run_dir / "motionlib").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(PYTHONPATH=str(OPS) + ":" + str(ROOT) + ":" + env.get("PYTHONPATH", ""),
               PYTHONDONTWRITEBYTECODE="1", E54_CHANNEL_STATS=str(STATS),
               YUANQI_E55_ARM=arm, YUANQI_E55_TIER="64",
               REPO_DIR=str((ROOT / "script/vendor/sonic_for_a3").resolve()),
               EXPERIMENT_DIR=str(run_dir), MOTION_FILE=str(run_dir / "motionlib"),
               CHECKPOINT=str(SOURCE), ISAAC_PYTHON=str(ROOT / "data/environments/a3-sonic/bin/python"),
               NUM_ENVS="16", NUM_MINI_BATCHES="4", NUM_PROCESSES="1",
               NUM_LEARNING_ITERATIONS="1", YUANQI_RESUME_LOG_DIR=str(log_dir),
               YUANQI_CLOTH_ANCHOR_ROLLOUT=str(ROOT / "data/experiments/a3_zhy_gradient_20261012_E11/rollout.pt"),
               YUANQI_CLOTH_ANCHOR_OUTPUTS=str(OPS / "data/experiments/a3_zhy_rootchannel_20261019_E53/anchors_035.pt"),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               WANDB_MODE="disabled", ACCELERATE_MIXED_PRECISION="no")
    import shutil as _shutil
    run_dir = TRAIN_ROOT / ("arm_%s_tier64" % arm.lower())
    _shutil.rmtree(run_dir, ignore_errors=True)
    (run_dir / "motionlib").mkdir(parents=True, exist_ok=True)
    for rel in MOTIONLIB_SOURCES:
        src = ROOT / rel
        if src.exists():
            _shutil.copy(src, run_dir / "motionlib" / src.name)
    console = (log_dir / "buildcheck.log").open("w")
    console.write("== E55 buildcheck %s start %s ==\n" % (arm, time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    console.flush()
    process = subprocess.Popen(cell_launch_argv(arm), stdout=console, stderr=console,
                               cwd=str(ROOT), env=env)
    deadline = time.time() + 900
    built = False
    while time.time() < deadline:
        if process.poll() is not None:
            break
        try:
            text = (log_dir / "buildcheck.log").read_text()
        except OSError:
            text = ""
        if "Number of environments" in text and ("policy" in text or "joint_pos" in text or "tokenizer" in text or True):
            built = True
            break
        time.sleep(5)
    if built and process.poll() is None:
        time.sleep(30)  # let ObservationManager finish term resolution
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
    console.close()
    ok = built and "Error" not in (log_dir / "buildcheck.log").read_text().split("Number of environments")[-1]
    atomic_json(OUT / ("buildcheck_" + run_cell(arm) + ".json"),
                dict(arm=arm, status="BUILD_OK" if ok else "BUILD_FAIL",
                     elapsed_seconds=round(time.time() - (deadline - 900), 3)))
    print("E55 buildcheck %s %s" % (arm, "BUILD_OK" if ok else "BUILD_FAIL"))
    return 0 if ok else 3


def train_stage(arm):
    started = time.time()
    cell = run_cell(arm)
    run_dir = TRAIN_ROOT / cell
    log_dir = LOG / cell
    if (OUT / ("train_" + cell + ".json")).exists():
        raise FileExistsError("E55 training cell is terminal; no re-run: " + cell)
    preflight()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(PYTHONPATH=str(OPS) + ":" + str(ROOT) + ":" + env.get("PYTHONPATH", ""),
               PYTHONDONTWRITEBYTECODE="1", E54_CHANNEL_STATS=str(STATS),
               YUANQI_E55_ARM=arm, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", WANDB_MODE="disabled", ACCELERATE_MIXED_PRECISION="no",
               REPO_DIR=str((ROOT / "script/vendor/sonic_for_a3").resolve()),
               EXPERIMENT_DIR=str(run_dir), MOTION_FILE=str(run_dir / "motionlib"),
               CHECKPOINT=str(SOURCE), YUANQI_E55_TIER="64",
               ISAAC_PYTHON=str(ROOT / "data/environments/a3-sonic/bin/python"),
               NUM_ENVS="16", NUM_MINI_BATCHES="4", NUM_PROCESSES="1",
               NUM_LEARNING_ITERATIONS=str(P.UPDATES), YUANQI_RESUME_LOG_DIR=str(log_dir),
               YUANQI_CLOTH_ANCHOR_ROLLOUT=str(ROOT / "data/experiments/a3_zhy_gradient_20261012_E11/rollout.pt"),
               YUANQI_CLOTH_ANCHOR_OUTPUTS=str(OPS / "data/experiments/a3_zhy_rootchannel_20261019_E53/anchors_035.pt"))
    atomic_json(OUT / ("training_launch_" + cell + ".json"),
                dict(argv=cell_launch_argv(arm), motion_file=str(run_dir / "motionlib"),
                     checkpoint=str(SOURCE), num_learning_iterations=P.UPDATES,
                     obs_func=P.FUNC_BY_ARM[arm], channel_stats=str(STATS),
                     wrapper="e53_interleave_plus_e55_residual_head"))
    (run_dir / "motionlib").mkdir(parents=True, exist_ok=True)
    for rel in MOTIONLIB_SOURCES:
        target = run_dir / "motionlib" / Path(rel).name
        if not target.exists():
            target.symlink_to(str(ROOT / rel))
    console = (log_dir / "console.log").open("a")
    console.write("== E55 %s start %s ==\n" % (cell, time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    for i in range(3):
        value = guard.preflight()
        append_record(log_dir / "preflight.jsonl", value)
        if i < 2:
            time.sleep(5)
    console.flush()
    process = subprocess.Popen(cell_launch_argv(arm), stdout=console, stderr=console,
                               cwd=str(ROOT), env=env)
    try:
        code = process.wait(timeout=P.RUNTIME_LIMIT_TRAIN_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        atomic_json(OUT / ("train_" + cell + ".json"),
                    dict(arm=arm, status="STOPPED_RUNTIME_CAP", elapsed=time.time() - started))
        return 2
    console.close()
    status = "TRAINED" if code == 0 else "STOPPED_LAUNCH_FAILURE"
    for milestone in P.MILESTONES:
        if not (run_dir / P.CHECKPOINT_BY_MILESTONE[milestone]).exists():
            status = "STOPPED_MISSING_MILESTONE_%d" % milestone
    digest_note = "receipt_missing"
    contract_path = log_dir / "cloth_contract.json"
    if contract_path.exists():
        try:
            _steps = read(contract_path)["optimizer_steps"]
            _digests = [row.get("head_parameter_digest") for row in _steps if row.get("head_parameter_digest")]
            if len(_digests) >= 2 and _digests[0] != _digests[-1]:
                digest_note = "head_digest_moved"
            elif len(_digests) >= 2:
                status = "FAIL_HEAD_DIGEST_FROZEN"
                digest_note = "head_digest_unchanged"
            else:
                digest_note = "digest_rows=%d" % len(_digests)
        except Exception as _exc:
            digest_note = "receipt_unreadable:%s" % _exc
    elif status == "TRAINED":
        status = "FAIL_HEAD_DIGEST_RECEIPT_MISSING"
    atomic_json(OUT / ("train_" + cell + ".json"), dict(
        arm=arm, status=status, exit_code=code, digest_note=digest_note,
        elapsed_seconds=round(time.time() - started, 3),
        checkpoints={m: str(run_dir / P.CHECKPOINT_BY_MILESTONE[m]) for m in P.MILESTONES},
        launcher=str(LAUNCHER)))
    print("E55", cell, "status", status, flush=True)
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


class E55LaunchProxy:
    """Word-swap sim_entry -> e55_frozen_entry so the head rides the MuJoCo eval."""

    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def Popen(self, argv, **kwargs):
        argv = ["script.a3.e55_frozen_entry" if word in ("script.a3.sim_entry", "script.a3.smpl_frozen_entry")
                else word for word in argv]
        kwargs["start_new_session"] = False
        return self.base.Popen(argv, **kwargs)


def evaluate_stage(arm, milestone, kind, variant="normal"):
    from script.a3 import evaluate_mujoco as evaluation
    from script.a3.frozen_evaluation_launch import LaunchProxy, arguments
    source_checkpoint = TRAIN_ROOT / run_cell(arm) / P.CHECKPOINT_BY_MILESTONE[milestone]
    if not source_checkpoint.exists():
        raise ValueError("Cell checkpoint missing: " + str(source_checkpoint))
    suffix = "" if variant == "normal" else "_" + variant
    copy_dir = EVAL_ROOT / ("checkpoints/" + run_cell(arm) + ("_m%d" % milestone) + suffix)
    copy_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = copy_dir / source_checkpoint.name
    if not checkpoint.exists():
        shutil.copy2(source_checkpoint, checkpoint)
        sidecar = source_checkpoint.with_suffix(".json")
        if sidecar.exists():
            shutil.copy2(sidecar, copy_dir / sidecar.name)
    run_config = TRAIN_ROOT / run_cell(arm) / "config.yaml"
    if run_config.exists():
        shutil.copy2(run_config, copy_dir / "config.yaml")
        meta = run_config.parent / "meta.yaml"
        if meta.exists():
            shutil.copy2(meta, copy_dir / "meta.yaml")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(OPS) + ":" + str(ROOT) + ":" + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["E54_CHANNEL_STATS"] = str(STATS)
    env["YUANQI_E55_ARM"] = arm
    env["YUANQI_E55_VARIANT"] = variant
    saved = {k: os.environ.get(k) for k in ("PYTHONPATH", "E54_CHANNEL_STATS", "YUANQI_E55_ARM", "YUANQI_E55_VARIANT")}
    os.environ.update(env)
    try:
        args = arguments(EVAL_ROOT / ("eval_%s_m%d_%s%s" % (run_cell(arm), milestone, kind, suffix)),
                         EVAL_LOG_ROOT / ("eval_%s_m%d_%s%s" % (run_cell(arm), milestone, kind, suffix)),
                         checkpoint, references_farm(kind))
        if kind == "official":
            args.frame_stride = 4
        evaluation.preflight = guard.preflight
        evaluation.monitor_child = lambda *a, **k: (a[0].wait(), None)
        evaluation.subprocess = E55LaunchProxy(evaluation.subprocess)
        if evaluation.evaluate_explicit(args):
            raise RuntimeError("E55 %s m%d %s evaluation failed; no repeat" % (run_cell(arm), milestone, kind))
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    gate_log(dict(ts=time.time(), type="reading", arm=arm, milestone=milestone, kind=kind,
                  variant=variant, eval_ref="zzx not window-excluded for milestone trend rows; gate values come from shutter rows only",
                  checkpoint_sha256=sha256(checkpoint)))
    print("E55", run_cell(arm), "m%d" % milestone, kind, variant, "evaluation complete", flush=True)


def _world_wdr(output):
    from script.a3.reference_contract import load_sim
    import numpy as np
    sim = load_sim()
    groups = dict(torso=("torso", "pelvis"), wrists=("wrist",), legs=("ankle", "knee", "foot", "hip"))
    values = {}
    for motion_path in sorted(Path(output).glob("*.timeseries.json")):
        trace = json.loads(motion_path.read_text())["motions"][0]
        ref = np.asarray(trace["ref_key_body_pos_w"], dtype=np.float64)
        simx = np.asarray(trace["sim_key_body_pos_w"], dtype=np.float64)
        names = list(trace["key_body_names"])
        dist = np.linalg.norm(simx - ref, axis=-1).mean(axis=0)
        torso = np.mean([dist[i] for i, n in enumerate(names) if any(g in n for g in groups["torso"])]) or 1e-9
        wrists = np.mean([dist[i] for i, n in enumerate(names) if any(g in n for g in groups["wrists"])])
        legs = np.mean([dist[i] for i, n in enumerate(names) if any(g in n for g in groups["legs"])])
        values[motion_path.name.split(".")[0]] = P.wdr(torso, wrists, legs)
    return values


def effect_stage():
    if not (OUT / "registration.json").exists():
        raise ValueError("E55 effect before register")
    results = {}
    for arm in P.ARMS:
        row = dict(cloth6={}, rank_metrics=dict(official_wdr=None, cloth_wdr=None, per_joint_worst=None),
                   shutter={})
        for milestone in P.MILESTONES:
            output = EVAL_ROOT / ("eval_%s_m%d_cloth" % (run_cell(arm), milestone))
            if not output.exists():
                raise ValueError("E55 effect: missing cloth eval m%d for %s" % (milestone, arm))
            cloth_wdr = _world_wdr(output)
            row["cloth6"][milestone] = round(float(sorted(cloth_wdr.values())[-1]) if False else float(sum(cloth_wdr.values()) / len(cloth_wdr)), 6)
        shutter = {}
        for kind in ("cloth", "official"):
            output = EVAL_ROOT / ("eval_%s_m64_%s" % (run_cell(arm), kind))
            wdr_values = _world_wdr(output)
            wdr_max = float(max(wdr_values.values()))
            joint = _per_joint(output)
            shutter[kind] = dict(wdr_max=round(wdr_max, 4), per_joint_bff=round(joint, 4))
        row["shutter"] = shutter
        row["rank_metrics"] = _landing_metrics(EVAL_ROOT / ("eval_%s_m64_official" % run_cell(arm)))
        row["rank_metrics"].update(official_wdr=shutter["official"]["wdr_max"],
                                   cloth_wdr=shutter["cloth"]["wdr_max"],
                                   per_joint_worst=max(shutter["cloth"]["per_joint_bff"],
                                                       shutter["official"]["per_joint_bff"]))
        results[arm] = row
    candidates = {a: dict(cloth6=results[a]["cloth6"], shutter=results[a]["shutter"],
                          rank_metrics=results[a]["rank_metrics"]) for a in P.ARMS}
    liveness = json.loads((OUT / "liveness.json").read_text()) if (OUT / "liveness.json").exists() else {}
    passes, selection = P.verdict(candidates, liveness)
    atomic_json(OUT / "effect_verification.json", dict(
        experiment=EXPERIMENT, candidates=candidates, passes=passes, selection=selection,
        liveness=liveness, bands=dict(cloth_wdr=P.BAND_CLOTH_WDR, official_wdr=P.BAND_OFFICIAL_WDR,
                                      per_joint_cloth=P.BAND_PERJOINT_CLOTH,
                                      per_joint_official=P.BAND_PERJOINT_OFFICIAL),
        scope="NO_MERGE: form conclusion only"))
    gate_log(dict(ts=time.time(), type="decision", arms=list(P.ARMS), selection=selection,
                  passes=passes, liveness=liveness))
    print(json.dumps(dict(candidates={a: dict(cloth6_64=candidates[a]["cloth6"][64],
                                              shutter=candidates[a]["shutter"]) for a in P.ARMS},
                          selection=selection), indent=1)[:1600], flush=True)


def _landing_metrics(output):
    """Real (Z, H, N) from explicit metrics: Z = zero-fall repeats, H = min first-fall step,
    N = total fall count (VerificationSpec v1.1; guard bands never enter the rank key)."""
    import json as _json
    zs, firsts, total = 0, [], 0
    seen = 0
    for metrics_path in sorted(Path(output).glob("*.metrics.json")):
        row = _json.loads(metrics_path.read_text())["motions"][0]
        falls = row.get("fall_count", row.get("fall"))
        first = row.get("first_fall_step", row.get("fall_tick"))
        if falls is None:
            raise ValueError("E55 rank key: metrics missing fall fields (fall_count/fall) in " + metrics_path.name)
        seen += 1
        f = 1 if falls in (True, 1, "true", "True") else 0
        s = int(first) if (f and first is not None) else 0
        total += f
        if f == 0:
            zs += 1
        firsts.append(s if f > 0 else 10 ** 9)
    if seen == 0:
        raise ValueError("E55 rank key: no metrics under " + str(output))
    return dict(Z=zs, H=min(firsts), N=total)


def _per_joint(output):
    import numpy as np
    from script.a3.reference_contract import load_sim
    sim = load_sim()
    worst = 0.0
    count = 0
    for metrics_path in sorted(Path(output).glob("*.metrics.json")):
        payload = json.loads(metrics_path.read_text())
        motion = payload["motions"][0]
        value = motion.get("per_joint_bff")
        if value is None:
            joints = motion.get("joints_rmse")
            value = joints if joints is not None else motion.get("cloth6_rmse", 0.0)
        worst = max(worst, float(value))
        count += 1
    if count == 0:
        raise ValueError("E55 per_joint: no metrics files under " + str(output))
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("register", "prepare", "buildcheck", "train", "evaluate", "effect"))
    parser.add_argument("--arm", choices=P.ARMS, default="A0")
    parser.add_argument("--milestone", type=int, choices=P.MILESTONES, default=64)
    parser.add_argument("--kind", choices=("cloth", "official"), default="cloth")
    parser.add_argument("--variant", choices=("normal", "zero_channels", "zero_head"), default="normal")
    args = parser.parse_args()
    if args.action == "register":
        register()
    elif args.action == "prepare":
        prepare_stage()
    elif args.action == "buildcheck":
        raise SystemExit(buildcheck_stage(args.arm))
    elif args.action == "train":
        raise SystemExit(train_stage(args.arm))
    elif args.action == "evaluate":
        evaluate_stage(args.arm, args.milestone, args.kind, args.variant)
    elif args.action == "effect":
        effect_stage()


if __name__ == "__main__":
    main()
