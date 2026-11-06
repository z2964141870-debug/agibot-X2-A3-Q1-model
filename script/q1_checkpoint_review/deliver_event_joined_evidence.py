"""Package verified saved84 figures and six original-speed representative videos."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

from event_summary import identity, read, require


def package(figures, media, output):
    require(output.is_absolute() and output.resolve() == output,
            "Use an absolute independent output directory")
    require(output.is_dir() and {p.name for p in output.iterdir()} == {"README.md"},
            "Output may contain only the reviewed README")
    archive = output.with_suffix(".zip")
    require(not archive.exists(), "Preserve existing evidence archives")
    before = []
    source_manifest = read(figures / "bundle_manifest.json")
    for member in source_manifest["files"]:
        source = figures / member["relative"]
        value = identity(source)
        require(value["bytes"] == member["bytes"] and value["sha256"] == member["sha256"],
                "Figure source differs from its sealed manifest")
        before.append(value)
    summary = read(figures / "analysis" / "summary.json")
    audit = read(figures / "analysis" / "audit.json")
    media_audit = read(media / "mac_media_audit.json")
    summary_sha = identity(figures / "analysis" / "summary.json")["sha256"]
    require(audit["status"] == "JOINED84_ANALYSIS_PASSED" and audit["sources_unchanged"]
            and summary["total_cases"] == summary["complete_cases"] == 84
            and summary["original_campaign_status"] == "FAILED"
            and summary["original_full84_campaign_gate_passed"] is False,
            "Require the independent complete saved84 comparison")
    require(media_audit["status"] == "PASSED"
            and media_audit["source_summary_sha256"] == summary_sha
            and len(media_audit["videos"]) == 6
            and media_audit["total_decoded_frames"] == 6192
            and not media_audit["late_burst_video_in_mac_bundle"],
            "Require the delivered and fully decoded six-video set")
    copies = [(figures / "analysis", output / "analysis"),
              (figures / "figures", output / "figures")]
    for source, target in copies:
        shutil.copytree(source, target)
    for source, target in ((figures / "six_panel_overview.png", output / "six_panel_overview.png"),
                           (figures / "delivery_audit.json", output / "source_figure_delivery_audit.json"),
                           (media / "mac_media_audit.json", output / "source_media_audit.json"),
                           (media / "media_contact.png", output / "media_contact.png")):
        before.append(identity(source))
        shutil.copyfile(source, target)
    for video in media_audit["videos"]:
        source = media / video["relative"]
        value = identity(source)
        require(value["bytes"] == video["bytes"] and value["sha256"] == video["sha256"]
                and video["original_speed"] and video["decoded_frames"] == 1032
                and video["metadata"]["r_frame_rate"] == "50/1",
                "Video differs from its decoded original-speed source")
        before.append(value)
        target = output / video["relative"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    require(len(list(output.glob("figures/*/*.png"))) == 12
            and len(list(output.glob("figures/*/*.pdf"))) == 12,
            "Require all twelve six-panel figures and PDFs")
    links = re.findall(r"\]\(([^)]+)\)", (output / "README.md").read_text())
    require(all(identity(Path(value["path"])) == value for value in before),
            "Saved evidence changed during packaging")
    result = dict(status="PASSED", total_cases=84, complete_cases=84,
        source_summary_sha256=summary_sha, source_inputs_unchanged=True,
        six_panel_png=12, six_panel_pdf=12, original_speed_videos=6,
        decoded_frames=6192, media_events=["stop", "reverse", "dropout"],
        late_burst_video_in_bundle=False, readme_links_checked=links,
        visual_inspection=dict(status="REVIEWED", figure_scope="twelve-figure overview and one late_burst detail",
            media_scope="six-row contact sheet at frames 0,49,60,1031; sparse inspection only",
            assets_rendered=True, natural_gait_qualification=False),
        historical_source_audits_retained=True,
        new_inference=0, new_physics=0, new_training=0,
        original_campaign_status="FAILED", original_campaign_final_gate_passed=False,
        measured_end_to_end_latency_ms=None, natural_teleoperation_qualified=False,
        user_accepted_checkpoint=None)
    (output / "evidence_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    for target in links:
        path = (output / target).resolve()
        require(path.is_relative_to(output) and path.exists(), "Broken evidence README link")
    members = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            value = identity(path)
            members.append(dict(relative=str(path.relative_to(output)),
                                bytes=value["bytes"], sha256=value["sha256"]))
    (output / "bundle_manifest.json").write_text(json.dumps(dict(files=members), indent=2) + "\n")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                stream.write(path, path.relative_to(output))
    with zipfile.ZipFile(archive) as stream:
        require(stream.testzip() is None, "Evidence ZIP CRC failure")
        require(set(stream.namelist()) == {m["relative"] for m in members} | {"bundle_manifest.json"},
                "Unexpected archive member")
        for member in members:
            data = stream.read(member["relative"])
            require(len(data) == member["bytes"]
                    and hashlib.sha256(data).hexdigest() == member["sha256"],
                    "Archived evidence differs from local evidence")
        require(stream.read("bundle_manifest.json") == (output / "bundle_manifest.json").read_bytes(),
                "Archive manifest differs")
    return dict(archive=identity(archive), files=len(members) + 1, audit=result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.figures, args.media, args.output)))
