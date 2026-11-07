"""Plot saved H4 specialist training evidence without inference or model loading."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile


def deliver(source, output):
    names = ("result.json", "fit_result.json", "training_trace.json", "fit.guard.json", "reused_dataset.json")
    raw = {name:(source/name).read_bytes() for name in names}
    values = {name:json.loads(value) for name, value in raw.items()}
    campaign, fit, trace, guard, dataset = (values[name] for name in names)
    if (campaign.get("status") != "TRAINED_PENDING_CLOSED_LOOP"
            or fit.get("status") != "TRAINED_PENDING_CLOSED_LOOP"):
        raise ValueError("Require a completed actual H4 fit")
    if (campaign.get("status") != "TRAINED_PENDING_CLOSED_LOOP"
            or campaign.get("source_integrity") != "PASSED"
            or campaign.get("cost_settlement", {}).get("state") != "SETTLED"
            or guard.get("status") != "WORKER_COMPLETED"
            or guard.get("cleanup", {}).get("worker_reaped") is not True
            or guard["cleanup"].get("worker_return_code") != 0
            or fit.get("method") != "h4_only_continuation" or fit.get("human_real_future_frames") != 4
            or fit.get("actual_updates", 0) <= 0 or fit.get("selection_split") != "validation"
            or fit.get("test_used_for_selection") is not False
            or not dataset.get("actual_states_and_labels_reused")):
        raise ValueError("Missing real H4 provenance, validation selection or clean settlement")
    if (not trace or trace[0]["update"] != 0 or trace[-1]["update"] != fit["actual_updates"]
            or any(a["update"] >= b["update"] for a, b in zip(trace, trace[1:]))
            or fit["selected_update"] != min(trace, key=lambda row:row["validation_risk"])["update"]):
        raise ValueError("Training trace and selected validation minimum differ")
    if output.exists():
        raise FileExistsError("Use a fresh H4 evidence figure root")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output.mkdir(parents=True)
    updates = [r["update"] for r in trace]
    selected = next(r for r in trace if r["update"] == fit["selected_update"])
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(updates, [r["validation_risk"] for r in trace], label="Equal-source H4 (selection)", color="#16816a")
    ax.plot(updates, [r["sample_weighted_h4_risk"] for r in trace], label="Sample-weighted H4", color="#4074a7")
    ax.axvline(fit["selected_update"], color="#cc5a4b", linestyle="--", label=f"Selected {fit['selected_update']}")
    ax.set(title="H4 validation action MSE", xlabel="Additional Adam updates", ylabel="MSE")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for i, horizon in enumerate((10, 5, 4, 2, 1)):
        ax.plot(updates, [r["diagnostic_horizons"][i]["action_mse"] for r in trace], label=f"H{horizon}")
    ax.set(title="Other horizons: diagnostics, unsupported in deployment", xlabel="Additional Adam updates", ylabel="MSE")
    ax.legend(fontsize=8)
    observed = [r for r in trace if r["loss"] is not None]
    ax = axes[0, 2]
    ax.plot([r["update"] for r in observed], [r["loss"] for r in observed], color="#4074a7")
    ax.set(title="Recorded H4 training batch loss", xlabel="Additional Adam updates", ylabel="MSE")
    ax = axes[1, 0]
    ax.plot([r["update"] for r in observed], [r["gradient_norm"] for r in observed], color="#9b624a")
    ax.axhline(1., linestyle=":", color="#555555", label="Gradient clipping limit")
    ax.set(title="Encoder gradient before clipping", xlabel="Additional Adam updates", ylabel="L2 norm")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.bar(np.arange(5), fit["visible_prefix_counts"], color="#16816a")
    ax.set_xticks(range(5))
    ax.set(title="Actual visible-prefix exposure (requested H4)", xlabel="Available future slots", ylabel="Training records")
    ax = axes[1, 2]
    before = {r["clip_id"]:r for r in trace[0]["validation"]}
    after = {r["clip_id"]:r for r in selected["validation"]}
    keys = sorted(before)
    if set(keys) != set(after):
        raise ValueError("Validation source identity changed during fitting")
    changes = [100*(after[k]["action_mse"]/before[k]["action_mse"]-1) for k in keys]
    ax.bar([str(k) for k in keys], changes, color=["#16816a" if v < 0 else "#cc5a4b" for v in changes])
    ax.axhline(0., color="#555555", linewidth=.8)
    ax.set(title="Selected vs update0, every validation source", xlabel="Owned source clip ID", ylabel="Action MSE change (%)")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    charge = campaign["cost_settlement"]["charged_seconds"]
    fig.suptitle(f"Q1 H4 continuation | fit {fit['elapsed_fit_seconds']:.2f}s | charged {charge}s | closed-loop pending", fontsize=13)
    for suffix in ("png", "pdf"):
        fig.savefig(output/("training_six_panel."+suffix), dpi=160)
    plt.close(fig)
    with (output/"training_curves.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("update", "equal_source_h4_mse", "sample_weighted_h4_mse", "loss", "gradient_norm", "elapsed_seconds"))
        writer.writeheader()
        for row in trace:
            writer.writerow(dict(update=row["update"], equal_source_h4_mse=row["validation_risk"],
                sample_weighted_h4_mse=row["sample_weighted_h4_risk"], loss=row["loss"],
                gradient_norm=row["gradient_norm"], elapsed_seconds=row["elapsed_seconds"]))
    record = dict(stage="H4_SPECIALIST_SAVED_TRAINING_FIGURES", actual_updates=fit["actual_updates"],
        selected_update=fit["selected_update"], charged_gpu_seconds=charge,
        source_inputs=[dict(path=str(source/name), bytes=len(value), sha256=hashlib.sha256(value).hexdigest()) for name, value in raw.items()],
        equal_source_validation_relative_change=selected["validation_risk"]/trace[0]["validation_risk"]-1,
        sample_weighted_validation_relative_change=selected["sample_weighted_h4_risk"]/trace[0]["sample_weighted_h4_risk"]-1,
        per_source_validation_changes_percent=dict(zip(map(str, keys), changes)),
        new_inference_calls=0, new_physical_steps=0, new_optimizer_updates=0,
        closed_loop_acceptance="PENDING", natural_teleoperation_qualified=False, hardware_latency_ms=None)
    (output/"audit.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    return record


def package(source, figures, bundle, receipt):
    if any(p.exists() for p in (bundle, receipt)):
        raise FileExistsError("Use fresh H4 training delivery paths")
    manifest = json.loads((source/"originals.json").read_text())
    for row in manifest["files"]:
        value = (source/row["name"]).read_bytes()
        if len(value) != row["bytes"] or hashlib.sha256(value).hexdigest() != row["sha256"]:
            raise ValueError("Downloaded H4 original changed: "+row["name"])
    audit = json.loads((figures/"audit.json").read_text())
    payload = {"originals/"+p.name:p.read_bytes() for p in source.iterdir() if p.is_file()}
    payload.update({"figures/"+p.name:p.read_bytes() for p in figures.iterdir() if p.is_file()})
    payload["README.md"] = '''# H4专用适配训练结果

2026-11-08，北京时间。从原R1验证选中的5000步编码器继续，只适配H4/80ms请求；复用1404训练和711验证真实记录，教师、decoder、归一化及物理保持。旧R1本身已经混合训练过H4。

实际3000次新增Adam更新，十个参数计数均3000；验证选择1500步，六次无改善早停。等来源H4验证MSE为0.05586137→0.05534572，下降0.923%；样本加权下降0.749%。六个验证来源分别变化+0.18%、-1.10%、-6.18%、-0.60%、-5.93%、+7.33%，改善并不一致。

- [训练六宫图](figures/training_six_panel.png)
- [PDF](figures/training_six_panel.pdf)
- [逐验证点CSV](figures/training_curves.csv)
- [训练结果原件](originals/fit_result.json)
- [原始曲线](originals/training_trace.json)
- [真实回收与资源记录](originals/q1_h4_specialist_completed_20261108.json)
- [原件身份](originals/originals.json)
- [包内SHA清单](MANIFEST.json)

纯fit50.62秒，GPU占用77秒已结算，累计2569/14400秒，无未结算预留。915保护输入核同；控制器与自有worker已退。没有新物理、渲染或视频；本次绘图只读取保存记录。原共享训练仍暂停，采纳null。

todo：选中的1500权重须独立完成三名义片加四事件共七个闭环，并复用旧R1/H4/H10保存对照。当前小幅拟合收益不能当作自然遥操改善。H4仍要求最多80ms未来等待；实际可见prefix可能更短，原始人体预处理因果性未证，硬件端到端延迟未测。
'''.encode("utf-8")
    members = [dict(name=n, bytes=len(b), sha256=hashlib.sha256(b).hexdigest()) for n,b in sorted(payload.items())]
    payload["MANIFEST.json"] = (json.dumps(dict(schema="q1_h4_training_evidence_v1", members=members), indent=2)+"\n").encode()
    with zipfile.ZipFile(bundle, "x", zipfile.ZIP_DEFLATED) as target:
        for name, raw in sorted(payload.items()):
            target.writestr(name, raw)
    with zipfile.ZipFile(bundle) as target:
        if target.testzip() is not None or any(target.read(n) != b for n,b in payload.items()):
            raise ValueError("H4 evidence CRC or member identity failed")
        links = re.findall(r"\]\(([^)]+)\)", target.read("README.md").decode())
        if any(n not in target.namelist() for n in links):
            raise ValueError("Broken H4 evidence README link")
    raw = bundle.read_bytes()
    value = dict(stage="H4_SPECIALIST_TRAINING_DELIVERY", status="PASSED",
        bundle=dict(path=str(bundle), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
        members=len(payload), readme_links=len(links), crc_and_member_hashes="PASSED",
        saved_training_audit=audit, original_files=len(manifest["files"]), new_videos=0,
        cumulative_gpu_charged_seconds=2569, user_accepted_checkpoint=None)
    with receipt.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-only", action="store_true")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.package_only:
        if args.bundle is None or args.receipt is None:
            parser.error("Packaging requires --bundle and --receipt")
        result = package(args.source, args.output, args.bundle, args.receipt)
        print(json.dumps(dict(bundle=result["bundle"], members=result["members"], links=result["readme_links"])))
    else:
        print(json.dumps(deliver(args.source, args.output), allow_nan=False))
