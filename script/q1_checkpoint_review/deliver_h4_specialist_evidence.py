"""Package audited H4 training, matched replay figures and original media."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

from deliver_student_h4_review import saved_gate
from student_h4_summary import identity,read,require


def package(root,output):
    require(root.is_absolute() and output.is_absolute() and output.resolve() == output
        and output.is_dir() and {p.name for p in output.iterdir()} == {"README.md"},
        "Use an independent directory containing its reviewed README only")
    archive = output.with_suffix(".zip")
    require(not archive.exists(),"Preserve prior evidence archives")
    figures = root/"data/analysis/Q1_H4_SPECIALIST_REVIEW_FIGURES_20261108"
    training = root/"data/analysis/Q1_H4_SPECIALIST01_FIGURES_20261108"
    value,audit = read(figures/"summary.json"),read(figures/"audit.json")
    rows = saved_gate(value,audit)
    media = read(figures/"mac_media_audit.json")
    require(media.get("status") == "PASSED" and media.get("visual_contact_inspected") is True
        and media.get("new_review_videos_on_mac",0) >= 1
        and media.get("new_review_videos_on_mac",0)+media.get("other_new_review_videos_remaining_on_server",-1) == 7,
        "Require actual decoded representative media with explicit remaining delivery")
    sources = []

    def copy(source,relative):
        source = Path(source)
        record = identity(source)
        target = output/relative
        require(target.resolve().is_relative_to(output) and not target.exists(),"Duplicate or escaping evidence member")
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        sources.append(record)

    for suffix in ("png","pdf"):
        paths = sorted(p for p in figures.glob("*."+suffix) if p.stem.startswith(("event_","nominal_")))
        require(len(paths) == 7,"Require all seven saved comparison six-panel figures")
        for source in paths:
            copy(source,"figures/"+source.name)
    for name in ("overview.png","per_case.csv","software_timing.csv","summary.json","audit.json",
            "delivery_audit.json","mac_media_audit.json"):
        copy(figures/name,"analysis/"+name)
    for name in ("training_six_panel.png","training_six_panel.pdf","training_curves.csv","audit.json"):
        copy(training/name,"training/"+name)
    copy(root/"data/deliveries/Q1_H4_SPECIALIST_REVIEW_NUMERIC_20261108.tgz","raw/server_numeric_original.tgz")
    copy(figures/"media_contact.png","media/media_contact.png")
    frames = 0
    for row in media["videos"]:
        source = Path(row["path"])
        found = identity(source)
        require(found["bytes"] == row["bytes"] and found["sha256"] == row["sha256"]
            and row.get("full_decode_passed") is True and row.get("fps") == 50
            and row.get("original_speed") is True and row.get("failure_hold_retained") is True
            and row["frames"] > 0,"Video differs from its fully decoded original")
        frames += row["frames"]
        copy(source,"media/"+row["delivery_name"])
    links = re.findall(r"\]\(([^)]+)\)",(output/"README.md").read_text())
    for link in links:
        target = (output/link).resolve()
        require(target.is_relative_to(output) and target.is_file(),"Broken evidence README link")
    require(all(identity(r["path"]) == r for r in sources),"Source changed while packaging H4 evidence")
    evidence = dict(status="PASSED",selected_update=1500,new_saved_cases=7,
        complete_new_cases=value["complete_new_cases"],reused_controls=17,total_comparison_cases=24,
        six_panel_comparison_png=7,six_panel_comparison_pdf=7,training_six_panel_png=1,training_six_panel_pdf=1,
        original_speed_videos=len(media["videos"]),full_decoded_frames=frames,
        new_review_videos_on_mac=media["new_review_videos_on_mac"],
        new_review_videos_remaining_on_server=media["other_new_review_videos_remaining_on_server"],
        readme_links_checked=links,source_inputs_unchanged=True,new_inference=0,new_physics=0,new_training=0,
        hardware_end_to_end_latency_ms=None,natural_teleoperation_qualified=False,
        original_shared_training_resumed=False,user_accepted_checkpoint=None)
    (output/"evidence_audit.json").write_text(json.dumps(evidence,indent=2)+"\n")
    members = []
    for source in sorted(output.rglob("*")):
        if source.is_file():
            record = identity(source)
            members.append(dict(relative=str(source.relative_to(output)),bytes=record["bytes"],sha256=record["sha256"]))
    (output/"bundle_manifest.json").write_text(json.dumps(dict(files=members),indent=2)+"\n")
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED) as stream:
        for source in sorted(output.rglob("*")):
            if source.is_file():
                stream.write(source,source.relative_to(output))
    with zipfile.ZipFile(archive) as stream:
        require(stream.testzip() is None,"H4 evidence CRC failed")
        require(set(stream.namelist()) == {r["relative"] for r in members}|{"bundle_manifest.json"},"Unexpected evidence member")
        for row in members:
            b = stream.read(row["relative"])
            require(len(b) == row["bytes"] and hashlib.sha256(b).hexdigest() == row["sha256"],"Archived H4 evidence changed")
    return dict(archive=identity(archive),members=len(members)+1,audit=evidence)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    print(json.dumps(package(args.root,args.output)))
