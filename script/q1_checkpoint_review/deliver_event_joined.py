"""Render a new local evidence bundle from the verified saved84 union."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from event_summary import CLIPS, CONFIGS, EVENTS, identity, read, require
from preview_summary import plot


def deliver(source, output, expected_sha):
    summary_path = source / "summary.json"
    before = [identity(source / name) for name in ("summary.json", "audit.json", "per_case.csv")]
    summary, audit = read(summary_path), read(source / "audit.json")
    require(before[0]["sha256"] == expected_sha
            and summary.get("stage") == "JOINED84_SAVED_EVENT_ANALYSIS"
            and audit.get("status") == "JOINED84_ANALYSIS_PASSED"
            and audit.get("sources_unchanged") is True
            and summary.get("total_cases") == summary.get("complete_cases") == 84
            and summary.get("original_campaign_status") == "FAILED"
            and summary.get("original_full84_campaign_gate_passed") is False
            and summary.get("joined84_sources_verified") is True,
            "Require the independently verified complete saved84 comparison")
    expected = {(e, m, h, c) for e in EVENTS for m, h in CONFIGS for c in CLIPS}
    rows = summary["rows"]
    require(len(rows) == 84 and {(r["event"], r["method"], r["horizon_frames"], r["clip"])
            for r in rows} == expected, "Changed event comparison coverage")
    require(output.is_absolute() and output.resolve() == output and not output.exists()
            and not output.is_relative_to(source), "Use a new independent local delivery root")
    archive = output.with_suffix(".zip")
    require(not archive.exists(), "Preserve existing evidence archives")
    (output / "analysis").mkdir(parents=True)
    for record in before:
        shutil.copyfile(record["path"], output / "analysis" / Path(record["path"]).name)
    figures = []
    for event in EVENTS:
        folder = output / "figures" / event
        folder.mkdir(parents=True)
        plotting = [dict(row, clip=event + "__" + row["clip"] + "__JOINED84")
                    for row in rows if row["event"] == event]
        plot(plotting, folder)
        figures.extend(sorted(folder.glob("*.png")))
    require(len(figures) == 12, "Expected twelve six-panel figures")
    from PIL import Image

    overview = Image.new("RGB", (1800, 1372), "white")
    for index, path in enumerate(figures):
        with Image.open(path) as image:
            require(image.width == 2100 and image.height == 1200, "Unexpected six-panel dimensions")
            overview.paste(image.convert("RGB").resize((600, 343), Image.Resampling.LANCZOS),
                           ((index % 3) * 600, (index // 3) * 343))
    overview.save(output / "six_panel_overview.png")
    lines = ["# Q1低延迟事件：独立合并84例证据", "",
        "2026-11-07。原任务在7200秒上限退出，保留FAILED；独立补齐六例后，读取保存78+6例完成全部对照。",
        "原共享训练保持暂停，本次分析和绘图没有新增推理、积分或训练。", "",
        "| 合成事件 | 200ms上肢RMSE(rad) | 80ms阻尼RMSE(rad) | 上肢变化 | 双脚平均承重滑移变化 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for event in EVENTS:
        selected = [r for r in summary["groups"] if r["event"] == event]
        def mean(method, horizon):
            return next(r["full_fixed_set_means"] for r in selected
                        if r["method"] == method and r["horizon_frames"] == horizon)
        control, candidate = mean("masked", 10), mean("damped_cv", 4)
        upper = 100 * (candidate["upper_rmse_rad"] / control["upper_rmse_rad"] - 1)
        slip = 100 * ((candidate["slip_left_m_s"] + candidate["slip_right_m_s"])
                      / (control["slip_left_m_s"] + control["slip_right_m_s"]) - 1)
        lines.append(f"| {event} | {control['upper_rmse_rad']:.6f} | {candidate['upper_rmse_rad']:.6f} | {upper:+.2f}% | {slip:+.2f}% |")
    lines += ["", "这些均值来自三固定片。预览等待200ms到80ms不等于实测硬件端到端少120ms；合成事件不代替真实人体遥操。",
        "侧倾、事件窗口上肢误差、动作突变与恢复逐例保留在summary.json。软件目标调用仍有超20ms长尾，未判自然遥操合格。", "",
        "- [12张六宫图总览](six_panel_overview.png)，PNG和PDF位于figures/四事件目录。",
        "- [逐例CSV](analysis/per_case.csv)、[数值与来源](analysis/summary.json)、[独立分析审计](analysis/audit.json)。",
        "- delivery_audit.json记录本地绘图来源；bundle_manifest.json列出成员大小与SHA。", "",
        "本包未含新增事件MP4，原速视频仍在服务器待传，已有视频不重渲染。旧部分图表包与约15MB历史证据包保持原样。",
        "原160k正式完成门和用户采纳null保持，本包不结算整个低延迟探索。", ""]
    (output / "README.md").write_text("\n".join(lines))
    require(all(identity(Path(r["path"])) == r for r in before), "Numerical sources changed while plotting")
    record = dict(status="PASSED", source_files=before, source_inputs_unchanged=True,
        source_summary_sha256=expected_sha, total_cases=84, complete_cases=84, six_panel_figures=12,
        new_inference=0, new_physics=0, new_training=0, new_videos_in_bundle=0,
        original_campaign_status="FAILED", original_campaign_final_gate_passed=False,
        measured_end_to_end_latency_ms=None, natural_teleoperation_qualified=False,
        user_accepted_checkpoint=None, visual_inspection="PENDING")
    (output / "delivery_audit.json").write_text(json.dumps(record, indent=2) + "\n")
    members = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            value = identity(path)
            members.append(dict(relative=str(path.relative_to(output)), bytes=value["bytes"], sha256=value["sha256"]))
    (output / "bundle_manifest.json").write_text(json.dumps(dict(files=members), indent=2) + "\n")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                stream.write(path, path.relative_to(output))
    with zipfile.ZipFile(archive) as stream:
        require(stream.testzip() is None, "Evidence archive CRC failure")
        for member in members:
            value = stream.read(member["relative"])
            require(len(value) == member["bytes"] and hashlib.sha256(value).hexdigest() == member["sha256"],
                    "Archive member differs from local evidence")
    return dict(archive=identity(archive), files=len(members) + 1, record=record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(deliver(args.source, args.output, args.summary_sha256)))
