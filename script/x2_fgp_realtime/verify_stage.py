"""Verify pinned sources, installed assets and the bounded ARM64 probe outputs."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import statistics
import tarfile

import numpy as np

from .assets import file_digest


SOURCES = (
    ("CLOTHO_FGP_REALTIME_SOURCE_1e719140_20261113.tgz", "clotho_fgp_realtime_20261113",
     "e6b0d59e67c4a8847b14d6c639c40af98b67db98c7874bf83c87229379232306"),
    ("CARTICULATE_UPSTREAM_ce5b2046_20261113.tgz", "carticulate_upstream_ce5b2046",
     "3bceae456878ff5555e3d053c9eb4f9f2d1eec47ca4746af4aa90379cc3194b0"),
    ("SOPHUS_12210_de0f8d3d_20261113.tgz", "sophus_de0f8d3d",
     "ca5d3f8955f7da6ec3c24b641d5a1a20f6f4cd9f031023e0eeaa4bb6bd9edf11"),
)
RUNS = ("neural_s1_t2_r2", "neural_s1_t1_r1", "full_s1_t4_r1", "full_cano_t4_r1")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing verification outputs")
    root = args.project.resolve()
    analysis = root / "data/analysis/X2_FGP_REALTIME_20261113"
    source_checks = []
    for archive, directory, expected in SOURCES:
        archive = root / "data/deliveries" / archive
        if file_digest(archive) != expected:
            raise ValueError("Pinned source archive changed: " + str(archive))
        checked = 0
        with tarfile.open(archive) as package:
            for member in package:
                if not member.isfile():
                    continue
                name = PurePosixPath(member.name)
                if name.is_absolute() or ".." in name.parts or len(name.parts) < 2:
                    raise ValueError("Unsafe archive path")
                target = root / "data/runtime" / directory / Path(*name.parts[1:])
                with package.extractfile(member) as stream:
                    expected_file = hashlib.sha256(stream.read()).hexdigest()
                if target.is_symlink() or file_digest(target) != expected_file:
                    raise ValueError("Original source changed: " + str(target))
                checked += 1
        source_checks.append(dict(archive=str(archive.relative_to(root)), sha256=expected,
                                  regular_files_sha_matched=checked))
    audit = json.loads((analysis / "asset_audit_orin.json").read_text())
    author_root = root / "data/runtime/clotho_fgp_realtime_20261113"
    for row in audit["files"]:
        path = author_root / row["path"]
        if path.stat().st_size != row["bytes"] or file_digest(path) != row["sha256"]:
            raise ValueError("Installed author asset changed: " + str(path))
    build = json.loads((analysis / "native_build_r1/build_receipt.json").read_text())
    module = author_root / "carticulate" / Path(build["module"]).name
    if file_digest(module) != build["sha256"]:
        raise ValueError("Installed native module differs from successful build")
    invariants = json.loads((analysis / "native_invariants_r1.json").read_text())
    if (invariants["status"] != "NATIVE_DYNAMICS_INVARIANTS_PASSED"
            or invariants["module_sha256"] != build["sha256"]):
        raise ValueError("Native invariant receipt mismatch")
    results, artifacts = [], []
    files = [analysis / "asset_audit_orin.json", analysis / "native_invariants_r1.json",
             analysis / "native_build_r1/build_receipt.json", module]
    for name in RUNS:
        report_path = analysis / name / "report.json"
        r = json.loads(report_path.read_text())
        expected = "RECORDED_FULL_PIPELINE_PASSED" if name.startswith("full_") else "RECORDED_NEURAL_STAGES_PASSED"
        if r["status"] != expected or r["input_frames"] != 61 or r["output_frames"] != 32:
            raise ValueError("Unexpected probe result")
        if r["hardware_commands"] != 0 or r["publish_allowed"]:
            raise ValueError("Probe was not advisory")
        fixture = root / "data/runtime/x2-fgp-viewer/recorded_fixture.npz"
        if r["fixture_sha256"] != file_digest(fixture):
            raise ValueError("Probe fixture changed")
        if r["manifest_sha256"] != file_digest(author_root / "models" / r["bundle"] / "manifest.json"):
            raise ValueError("Probe bundle changed")
        output = analysis / name / "outputs.npz"
        with np.load(output, allow_pickle=False) as saved:
            if saved["pose"].shape != (32, 24, 3, 3):
                raise ValueError("Invalid saved pose shape")
            if not all(np.isfinite(saved[k]).all() for k in saved.files):
                raise ValueError("Nonfinite saved outputs")
            if r["native_physics"] and saved["translation"].shape != (32, 3):
                raise ValueError("Native translation missing")
        result = {key: r[key] for key in ("status", "bundle", "threads", "native_physics",
                  "initialize_s", "timing_all_ms", "timing_after_first2_ms")}
        result["run"] = name
        if r["native_physics"]:
            stable = r["rows"][2:]
            stages = stable[0]["author_exclusive_stage_ms"]
            result["exclusive_stage_median_ms"] = {
                k: statistics.median(row["author_exclusive_stage_ms"][k] for row in stable) for k in stages}
            result["native_physics_median_ms"] = statistics.median(row["physics_ms"] for row in stable)
            result["statuses"] = sorted({row["status"] for row in r["rows"]})
        results.append(result)
        files.extend((report_path, output))
    for path in sorted(Path(__file__).parent.iterdir()):
        if path.is_file() and (path.suffix == ".py" or path.name == "CMakeLists.txt"):
            files.append(path.resolve())
    for path in files:
        artifacts.append(dict(path=str(path.relative_to(root)), bytes=path.stat().st_size,
                              sha256=file_digest(path)))
    receipt = dict(status="ARM64_STAGE_INTEGRITY_PASSED", machine=platform.machine(),
        source_checks=source_checks, original_assets_sha_matched=len(audit["files"]),
        installed_native_matches_build=True, native_invariants=invariants, runs=results,
        artifacts=artifacts, hardware_commands=0, publish_allowed=False,
        windows_binary_equivalence="NOT_ESTABLISHED", accuracy_ground_truth_evaluated=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
