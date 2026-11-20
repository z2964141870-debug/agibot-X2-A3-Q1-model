"""Isolated closed-loop jobs with separate collection, selection and blind evaluation."""

from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import subprocess
import sys
from protocol import ROOT, EVALUATION


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def job(spec):
    split, surface, arm, kind = spec
    target = ROOT / "runs" / split / surface / arm / kind
    if (target / "receipt.json").exists():
        return
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    with (logs / ("_".join(spec) + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (target / "receipt.json").is_file():
            return
        with (logs / ("_".join(spec) + ".txt")).open("w") as f:
            subprocess.run(
                [sys.executable, str(ROOT / "worker.py"), *spec],
                check=True,
                stdout=f,
                stderr=subprocess.STDOUT,
            )


def run(splits, surfaces, arms, kinds=("static", "relay")):
    jobs = [(s, f, a, k) for s in splits for f in surfaces for a in arms for k in kinds]
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(job, jobs))


def selection():
    arms = ["published", "linear", "net_0.5", "net_1.0", "net_1.5"]
    run(["tune"], ["lag", "rough"], arms)
    scores = {}
    for arm in arms:
        stat = []
        dyn = []
        hold = []
        transfers = []
        for surface in ["lag", "rough"]:
            f = ROOT / "runs/tune" / surface / arm
            s = read(f / "static/trials.json")
            d = read(f / "relay/relay.json")
            stat.extend(x["success"] for x in s)
            dyn.extend(x["success"] for x in d)
            hold.extend(min(x["terminal_stable_ticks"], 50) / 50 for x in s)
            transfers.extend(x["released"] for x in read(f / "relay/receipt.json")["episodes"])
        score = (
            0.45 * sum(stat) / len(stat)
            + 0.45 * sum(dyn) / len(dyn)
            + 0.05 * sum(hold) / len(hold)
            + 0.05 * sum(transfers) / len(transfers)
        )
        scores[arm] = dict(static=sum(stat), relay=sum(dyn), released=sum(transfers), score=score)
    chosen = max(
        (a for a in arms if a.startswith("net")),
        key=lambda a: (scores[a]["score"], -abs(float(a.split("_")[1]) - 1)),
    )
    record = dict(
        selected=chosen,
        tune=scores,
        onnx_sha256=sha(ROOT / "trained/predict.onnx"),
        protocol=sha(ROOT / "protocol.py"),
    )
    (ROOT / "selection.json").write_text(json.dumps(record, indent=2))
    print(record, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "data":
        run(["collect0", "collect1", "valid"], ["normal", "lag", "rough"], ["published"])
    elif cmd == "diagnose":
        run(["audit"], ["normal", "lag", "recovery_only", "outside_recovery"], ["published"])
    elif cmd == "select":
        selection()
    elif cmd == "test":
        frozen = read(ROOT / "selection.json")
        assert frozen["onnx_sha256"] == sha(ROOT / "trained/predict.onnx") and frozen[
            "protocol"
        ] == sha(ROOT / "protocol.py")
        run(["test0", "test1"], EVALUATION, ["published", "linear", frozen["selected"]])
    else:
        raise ValueError(cmd)
