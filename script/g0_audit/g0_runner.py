import json, os, shutil, subprocess, sys, time, hashlib
from pathlib import Path

OPS = Path("/opt/data/private/yuanqi_ops")
ROOT = Path("/opt/data/private/hechengzhu/ZHY")
E26 = ROOT / "data/experiments/a3_zhy_smpl_frozen_20261013_E26"
E02_JOB = ROOT / "data/experiments/a3_zhy_search_20261012_E02/job.json"
CKPT = ROOT / "data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt"
OUT = ROOT / "data/experiments/a3_zhy_reachability_audit_20261019_G0"
LOGS = ROOT / "logs/a3_zhy_reachability_audit_20261019_G0"
OPS_OUT = OPS / "data/experiments/a3_zhy_reachability_audit_20261019_G0"
OPS_LOGS = OPS / "logs/a3_zhy_reachability_audit_20261019_G0"
SEEDS = [0, 1, 2]

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_json(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, p)

def preflight():
    rows = []
    for i in range(3):
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True)
        rows.append([line.strip() for line in out.stdout.strip().splitlines()])
        if i < 2:
            time.sleep(5)
    idle = True
    for sample in rows:
        for line in sample:
            util, mem = [int(x.strip()) for x in line.split(",")]
            if util > 5 or mem > 1024:
                idle = False
    memmax = None
    for cand in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        p = Path(cand)
        if p.exists():
            memmax = p.read_text().strip()
            break
    receipt = dict(gpu_samples=rows, gpu_idle=idle, cgroup_memory_max=memmax, at=time.time())
    OPS_LOGS.mkdir(parents=True, exist_ok=True)
    with (OPS_LOGS / "preflight.jsonl").open("a") as stream:
        stream.write(json.dumps(receipt) + "\n")
    if not idle:
        raise SystemExit("G0 preflight FAIL: GPU not idle 3x/5s")
    return receipt

def build_farm(kind, farm):
    farm.mkdir(parents=True, exist_ok=True)
    names = []
    if kind == "cloth":
        for csv in sorted(E26.glob("references/*.csv")):
            link = farm / csv.name
            if not link.exists():
                link.symlink_to(os.path.relpath(csv, farm))
            names.append(csv.stem)
    else:
        motions = json.loads(E02_JOB.read_text())["diagnostic_motions"]
        for row in motions:
            path = Path(row["path"])
            link = farm / path.name
            if not link.exists():
                link.symlink_to(os.path.relpath(path, farm))
            names.append(path.stem)
    return names

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    OPS_OUT.mkdir(parents=True, exist_ok=True)
    for p in (CKPT, E26 / "references", E02_JOB):
        if not p.exists():
            raise SystemExit("G0 missing input: " + str(p))
    preflight()
    sys.path.insert(0, str(OPS))
    sys.path.insert(0, str(ROOT))
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    from script.a3 import evaluate_mujoco as evaluation
    from script.a3.frozen_evaluation_launch import LaunchProxy, arguments
    evaluation.monitor_child = lambda child, directory, config: (child.wait(), None)
    evaluation.subprocess = LaunchProxy(evaluation.subprocess)
    reg26 = json.loads((E26 / "registration.json").read_text())
    batch_receipts = []
    for seed in SEEDS:
        for kind in ("cloth", "official"):
            tag = "seed%d_%s" % (seed, kind)
            out_dir = OUT / ("seed%d" % seed) / kind
            log_dir = LOGS / tag
            receipt_path = OPS_OUT / ("batch_%s.json" % tag)
            if receipt_path.exists() and json.loads(receipt_path.read_text()).get("status") == "passed":
                print("skip", tag, "(receipt)")
                continue
            farm = OPS_OUT / "farms" / kind
            names = build_farm(kind, farm)
            args = arguments(out_dir, log_dir, CKPT, farm)
            args.frame_stride = 1
            args.joint_noise_seed = seed
            args.capture_inputs = False
            fields = {}
            for k in ("reference_fps", "frame_stride", "joint_noise_seed", "joint_noise_std",
                      "capture_inputs", "action_delay_ms"):
                if hasattr(args, k):
                    fields[k] = str(getattr(args, k))
            t0 = time.time()
            status, err = "passed", None
            try:
                if evaluation.evaluate_explicit(args):
                    status, err = "failed", "evaluate_explicit nonzero"
            except Exception as exc:
                status, err = "failed", repr(exc)
            motions = {}
            summary_path = out_dir / "explicit_summary.json"
            if summary_path.exists():
                for record in json.loads(summary_path.read_text())["motions"]:
                    motions[record["motion"]] = dict(fall=record.get("fall"),
                                                     fall_time_s=record.get("fall_time_s"),
                                                     num_policy_steps=record.get("num_policy_steps"))
            receipt = dict(tag=tag, seed=seed, kind=kind, status=status, error=err,
                           elapsed_s=time.time() - t0, motions=motions, args=fields,
                           checkpoint=str(CKPT), checkpoint_sha256=sha256(CKPT),
                           eval_output=str(out_dir),
                           protocol="E26-exact: csv_source_fps=30 frame_stride=1 batch_once action_delay=0 mjcf=twostage_fit_optimized",
                           e26_frame_stride=reg26.get("frame_stride"),
                           e26_fall_height_m=reg26.get("fall_height_m"),
                           writes_scope="repo-side NEW namespace (E52/E53 precedent) + OPS mirror receipts",
                           at=time.time())
            atomic_json(receipt_path, receipt)
            batch_receipts.append(receipt)
            print(tag, status, len(motions), "motions", flush=True)
            if status != "passed":
                raise SystemExit("G0 batch failed: " + tag + " " + str(err))
    atomic_json(OPS_OUT / "g0_eval_complete.json",
                dict(status="COMPLETE", batches=batch_receipts, checkpoint=str(CKPT),
                     checkpoint_sha256=sha256(CKPT), seeds=SEEDS, at=time.time()))
    print("G0 EVAL DRIVER DONE", flush=True)

if __name__ == "__main__":
    main()
