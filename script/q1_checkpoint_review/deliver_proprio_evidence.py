"""Seal the failed selected3000 review, training plots and decoded saved media."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

from deliver_student_proprio_review import saved_gate
from student_proprio_summary import identity, read, require


def package(root, output):
    require(root.is_absolute() and output.is_absolute() and output.resolve() == output,
            "Require absolute project and independent delivery paths")
    require(output.is_dir() and {p.name for p in output.iterdir()} == {"README.md"},
            "The new delivery directory may contain only its reviewed README")
    archive = output.with_suffix(".zip")
    require(not archive.exists(), "Preserve previous evidence archives")
    figures = root/"data/analysis/Q1_F0_PROPRIO_REVIEW_FIGURES_20261108"
    training = root/"data/analysis/Q1_F0_PROPRIO_FIGURES_20261108"
    summary, audit = read(figures/"summary.json"), read(figures/"audit.json")
    rows = saved_gate(summary, audit)
    current = [r for r in rows if r["variant"] == "proprio_f0"]
    require(len(current) == 7 and not any(r["complete"] for r in current)
            and all(r["valid_steps"] < r["reference_steps"] for r in current),
            "Preserve all seven actual failures")
    media = read(figures/"mac_media_audit.json")
    require(media["status"] == "PASSED" and len(media["videos"]) == 2
            and media["new_review_videos_on_mac"] == 1
            and media["other_new_review_videos_remaining_on_server"] == 6
            and media["visual_contact_inspected"] is True,
            "Require decoded representative media and explicit remaining delivery")
    sources = []

    def copy(source, relative):
        source = Path(source)
        value = identity(source)
        target = output/relative
        require(target.resolve().is_relative_to(output) and not target.exists(),
                "Do not overwrite evidence or escape the new directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        sources.append(value)

    for suffix in ("png", "pdf"):
        paths = sorted(p for p in figures.glob("*."+suffix)
                       if p.stem.startswith(("event_", "nominal_")))
        require(len(paths) == 7, "Require seven saved comparison six-panel figures")
        for source in paths:
            copy(source, "figures/"+source.name)
    for name in ("overview.png", "per_case.csv", "software_timing.csv", "summary.json",
                 "audit.json", "delivery_audit.json", "mac_media_audit.json"):
        copy(figures/name, "analysis/"+name)
    for name in ("causal_history_six_panel.png", "causal_history_six_panel.pdf", "causal_history_curves.csv",
                 "zero_history_six_panel.png", "zero_history_six_panel.pdf", "zero_history_curves.csv", "delivery.json"):
        copy(training/name, "training/"+name)
    copy(figures/"proprio_f0_start_contact.png", "media/proprio_f0_start_contact.png")
    for row in media["videos"]:
        source = Path(row["path"])
        require(identity(source)["bytes"] == row["bytes"]
                and identity(source)["sha256"] == row["sha256"]
                and row["full_decode_passed"] is True and row["fps"] == 50
                and row["frames"] == 152 and row["original_speed"] is True
                and row["failure_hold_retained"] is True,
                "Representative video differs from its decoded original")
        copy(source, "media/"+row["delivery_name"])
    copy(root/"data/deliveries/Q1_F0_PROPRIO_REVIEW_NUMERIC_20261108.tgz",
         "raw/server_numeric_original.tgz")
    copy(root/"data/deliveries/Q1_F0_PROPRIO_RAW_20261108.tgz", "raw/paired_training_original.tgz")
    links = re.findall(r"\]\(([^)]+)\)", (output/"README.md").read_text())
    for link in links:
        target = (output/link).resolve()
        require(target.is_relative_to(output) and target.is_file(), "Broken evidence README link")
    require(all(identity(Path(v["path"])) == v for v in sources),
            "Source evidence changed during packaging")
    evidence = dict(status="PASSED", selected_update=3000, new_cases=7,
        complete_new_cases=0, reused_saved_controls=21, comparison_cases=28,
        comparison_six_panel_png=7, comparison_six_panel_pdf=7,
        training_six_panel_png=2, training_six_panel_pdf=2,
        original_speed_videos=2, new_review_videos=1, historical_comparator_videos=1,
        new_review_videos_remaining_on_server=6, full_decoded_frames=304,
        source_inputs_unchanged=True, readme_links_checked=links,
        visual_inspection="comparison overview, A023 detail and six sampled media frames",
        new_inference_calls=0, new_physical_transitions=0, new_training_updates=0,
        hardware_end_to_end_latency_ms=None, natural_teleoperation_qualified=False,
        original_shared_training_resumed=False, user_accepted_checkpoint=None)
    (output/"evidence_audit.json").write_text(json.dumps(evidence, indent=2)+"\n")
    members = []
    for source in sorted(output.rglob("*")):
        if source.is_file():
            value = identity(source)
            members.append(dict(relative=str(source.relative_to(output)), bytes=value["bytes"],
                                sha256=value["sha256"]))
    (output/"bundle_manifest.json").write_text(json.dumps(dict(files=members), indent=2)+"\n")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as stream:
        for source in sorted(output.rglob("*")):
            if source.is_file():
                stream.write(source, source.relative_to(output))
    with zipfile.ZipFile(archive) as stream:
        require(stream.testzip() is None, "Evidence archive CRC failed")
        require(set(stream.namelist()) == {r["relative"] for r in members}|{"bundle_manifest.json"},
                "Unexpected evidence archive member")
        for row in members:
            raw = stream.read(row["relative"])
            require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"],
                    "Archived evidence does not match its manifest")
    return dict(archive=identity(archive), members=len(members)+1, audit=evidence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.root, args.output)))
