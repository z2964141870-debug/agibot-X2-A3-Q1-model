"""Collect completed development runs and plot the actual optimization logs."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root, out):
    out.mkdir(exist_ok=True)
    families = ("joined_roll_dev","settled_entry_dev","support_feedback_01",
                "support_feedback_variants","side_entry_01","clearance_dev_01")
    groups = {}
    for family in families:
        for path in sorted((root/family).glob("*/results.json")):
            rows = json.loads(path.read_text())
            manifest = path.with_name("manifest.json")
            config = json.loads(manifest.read_text())
            kinds = {}
            for kind in dict.fromkeys(r["pose"] or config["group"] for r in rows):
                subset = [r for r in rows if (r["pose"] or config["group"]) == kind]
                passed = [r for r in subset if r["success"]]
                soft = [r["trial"] for r in passed if r["peak_qd"]<=12 and r["ground_mm"]<=10 and r["self_mm"]<=2]
                kinds[kind] = dict(n=len(subset),fell=sum(r["phases"][0]["fell"] for r in subset),
                    handoffs=len(passed),soft=soft,wrong=sum(r["wrong_release"] for r in subset),
                    owned_max=[max(r.get(key,0.) for r in subset) for key in ("peak_qd","ground_mm","self_mm")],
                    trial_max=[max(r[key] for r in subset) for key in ("trial_peak_qd","trial_ground_mm","trial_self_mm")])
            groups[str(path.parent.relative_to(root))] = dict(seed=config["seed"],group=config["group"],
                results_sha256=sha(path),runtime_manifest_sha256=sha(manifest),outcomes=kinds)
    metrics = {}
    fig, axes = plt.subplots(2,3,figsize=(14,7),layout="constrained")
    for column,family in enumerate(("gentle_roll_01","support_feedback_01","side_entry_01")):
        path = root/family/"metrics.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        metrics[family] = dict(source_sha256=sha(path),count=len(records))
        for side in sorted(set(row.get("side",0) for row in records)):
            runs=[r for r in records if r.get("side",0)==side]
            for axis, field in zip(axes[:,column],(("train_score","valid_best") if column==0 else ("training","validation")),strict=True):
                samples = [r for r in runs if field in r]
                axis.plot([r["generation"] for r in samples],[r[field] for r in samples],label=f"side {side}" if column==2 else family)
                axis.set(xlabel="Generation",ylabel=field,title=family)
                axis.grid(alpha=.25)
        axes[0,column].legend(fontsize=8)
        with (out/(family+".csv")).open("w") as f:
            writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in records))))
            writer.writeheader()
            writer.writerows(records)
    fig.suptitle("Q1 recovery | Optimization scores from development simulations")
    fig.savefig(out/"training.png",dpi=160)
    fig.savefig(out/"training.pdf")
    plt.close(fig)
    models={str(p.relative_to(root)):sha(p) for family in families for ext in ("*.onnx","*.npz") for p in (root/family).glob(ext)}
    (out/"manifest.json").write_text(json.dumps(dict(final_acceptance="not_run",runs=groups,
        training=metrics,models=models),indent=2))


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    collect(args.root,args.out)
