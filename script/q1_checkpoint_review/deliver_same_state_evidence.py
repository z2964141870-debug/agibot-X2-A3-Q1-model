"""Package existing same-state diagnostics without rerunning their policies."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"primebot_q1"))
from common import identity, require, save


def read(path):
    return json.loads(path.read_text())


def package(root, output):
    require(root.is_absolute() and output.is_absolute() and output.resolve() == output,
            "Require absolute project and independent output paths")
    require(output.is_dir() and {p.name for p in output.iterdir()} == {"README.md"}
            and not output.with_suffix(".zip").exists(), "Preserve previous deliveries")
    raw = root/"data/analysis/Q1_SAME_STATE01_ORIGINALS_20261108"
    figures = root/"data/analysis/Q1_SAME_STATE01_FIGURES_20261108"
    numeric = root/"data/deliveries/Q1_SAME_STATE_NUMERIC_20261108.tgz"
    summary, audit, original = (read(raw/n) for n in ("summary.json", "audit.json", "source_result.json"))
    require(summary.get("stage") == "SAME_STATE_SAVED_ANALYSIS" and summary.get("status") == "PASSED"
            and summary.get("source_integrity") == "PASSED"
            and summary.get("original_campaign_status") == original.get("status") == "FAILED"
            and summary.get("reused_policy_calls") == 936
            and all(summary.get(k) == 0 for k in ("new_policy_calls", "new_physical_steps", "new_optimizer_updates"))
            and summary.get("privileged_future_for_offline_reference_only") is True
            and summary.get("hardware_end_to_end_latency_ms") is None
            and summary.get("natural_teleoperation_qualified") is False
            and audit.get("status") == "SAME_STATE_SAVED_ANALYSIS_PASSED"
            and audit.get("source_inputs_unchanged") is True,
            "Require the independent saved936 result and preserved original failure")
    guards = summary.get("phase_guards", [])
    require(len(guards) == 8 and all(g.get("status") == "WORKER_COMPLETED"
            and g.get("cleanup", {}).get("worker_reaped") is True
            and g["cleanup"].get("worker_return_code") == 0 for g in guards), "Require eight clean guards")
    cases = summary.get("comparisons", [])
    require(len(cases) == 2 and sorted(c["prefix_calls"] for c in cases) == [106, 128]
            and all(c["canonical_upper_indices"] == [15,16,17,18,22,23,24,25] for c in cases)
            and all(c["full_clip_metrics"] is None for c in cases), "Require actual named prefix comparisons")
    manifest = read(raw/"originals.json")
    require(len(manifest) == 6 and len({r["name"] for r in manifest}) == 6, "Require six original records")
    with tarfile.open(numeric, "r:gz") as stream:
        require(set(stream.getnames()) == {r["name"] for r in manifest}|{"originals.json"}, "Changed numeric archive")
        require(stream.extractfile("originals.json").read() == (raw/"originals.json").read_bytes(), "Manifest differs")
        for row in manifest:
            value = identity(raw/row["name"])
            require(value["bytes"] == row["bytes"] and value["sha256"] == row["sha256"], "Changed original record")
            require(stream.extractfile(row["name"]).read() == (raw/row["name"]).read_bytes(), "Changed archived original")
    figure_sources = read(figures/"sources.json")
    require(len(figure_sources.get("inputs", [])) == 3
            and all(identity(r["path"]) == r for r in figure_sources["inputs"]), "Figures have different inputs")
    sources = []

    def copy(source, relative):
        value = identity(source)
        target = output/relative
        require(target.resolve().is_relative_to(output) and not target.exists(), "Unsafe or duplicate delivery member")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied = identity(target)
        require(copied["bytes"] == value["bytes"] and copied["sha256"] == value["sha256"], "Copied member differs")
        sources.append(value)

    for case in cases:
        for suffix in ("png", "pdf"):
            copy(figures/(case["clip"]+"."+suffix), "figures/"+case["clip"]+"."+suffix)
    for name in ("same_state_residuals.csv", "sources.json"):
        copy(figures/name, "analysis/"+name)
    for name in ("summary.json", "audit.json", "source_result.json", "originals.json"):
        copy(raw/name, "analysis/"+name)
    for row in manifest:
        if row["name"].startswith("q1_"):
            copy(raw/row["name"], "records/"+row["name"])
    copy(numeric, "raw/"+numeric.name)
    links = re.findall(r"\]\(([^)]+)\)", (output/"README.md").read_text())
    require(all((output/p).resolve().is_relative_to(output) and (output/p).is_file() for p in links), "Broken README link")
    require(all(identity(r["path"]) == r for r in sources), "Source changed while copying")
    save(output/"evidence_audit.json", dict(status="PASSED", original_campaign_status="FAILED",
        saved_analysis_status=audit["status"], reused_policy_calls=936, new_policy_calls=0,
        new_physical_steps=0, new_optimizer_updates=0, six_panel_png=2, six_panel_pdf=2,
        new_videos=0, hardware_end_to_end_latency_ms=None, natural_teleoperation_qualified=False,
        user_accepted_checkpoint=None, readme_links_checked=links))
    members = [dict(relative=str(p.relative_to(output)), **{k:v for k,v in identity(p).items() if k != "path"})
               for p in sorted(output.rglob("*")) if p.is_file()]
    save(output/"bundle_manifest.json", dict(files=members))
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as stream:
        for p in sorted(output.rglob("*")):
            if p.is_file():
                stream.write(p, p.relative_to(output))
    with zipfile.ZipFile(archive) as stream:
        require(stream.testzip() is None and set(stream.namelist()) == {r["relative"] for r in members}|{"bundle_manifest.json"},
                "Archive CRC or exact members failed")
        for row in members:
            value = stream.read(row["relative"])
            require(len(value) == row["bytes"] and hashlib.sha256(value).hexdigest() == row["sha256"], "Archive member differs")
    return dict(archive=identity(archive), members=len(members)+1, readme_links=len(links), sources=len(sources))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.root, args.output)))
