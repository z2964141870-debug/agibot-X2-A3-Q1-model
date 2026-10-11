"""Read-only log/hash verification into a small diagnostic evidence manifest."""
import argparse
import hashlib
import json
from pathlib import Path
from script.a3.idle_reboot_watch import save_json

ROOT = Path(__file__).resolve().parents[2]


def record(case):
    logs = ROOT / ("logs/gpu_operator_compare_20261011_"+case)
    output = ROOT / ("data/experiments/gpu_operator_compare_20261011_"+case)
    summary = json.loads((ROOT / ("data/manifests/gpu_operator_compare_result_20261011_"+case+".json")).read_text())
    paths = list((logs / "phases").glob("*.jsonl"))
    paths += [logs/name for name in ("health.jsonl","kernel.jsonl","metadata.json","external_mirror.jsonl","external_tail.jsonl")]
    paths += [output/"worker_result.json"]
    files = {}
    for path in paths:
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda:stream.read(65536),b""):
                    digest.update(block)
            files[str(path.relative_to(ROOT))] = dict(bytes=path.stat().st_size,sha256=digest.hexdigest())
    phase_count, matmul_begin, matmul_end = 0, 0, 0
    for path in (logs / "phases").glob("*.jsonl"):
        for line in path.read_text().splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            phase_count += 1
            if item["phase"] == "MATMUL":
                matmul_begin += item["event"] == "BEGIN"
                matmul_end += item["event"] == "END"
    mirror = {}
    for name in ("external_mirror.jsonl","external_tail.jsonl"):
        path = logs / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            item = json.loads(line)
            if "byte_offset" in item:
                mirror[(item["file"],item["byte_offset"])] = item["record"]
    matched, phase_covered, mismatches = 0, 0, []
    for (file,offset),expected in mirror.items():
        path = ROOT / file
        with path.open("rb") as stream:
            stream.seek(offset)
            actual = json.loads(stream.readline())
        if actual == expected:
            matched += 1
            phase_covered += "/phases/" in file
        else:
            mismatches.append(dict(file=file,offset=offset))
    kernels = [json.loads(x) for x in (logs/"kernel.jsonl").read_text().splitlines()]
    result_path = str((output/"worker_result.json").relative_to(ROOT))
    result_verified = (result_path in files and files[result_path]["sha256"] == summary["worker_result_sha256"])
    evidence = dict(experiment=summary["experiment"], state=summary["state"], files=files,
                    phase_records=phase_count, matmul_begins=matmul_begin, matmul_completed=matmul_end,
                    mirrored_unique_records=len(mirror), mirror_records_matched=matched,
                    mirror_phase_records_covered=phase_covered, all_phase_records_mirrored=(phase_covered==phase_count),
                    mirror_mismatches=mismatches, worker_result_sha256_verified=result_verified,
                    kernel_queries=len(kernels), kernel_queries_successful=all(x["query"].get("returncode")==0 for x in kernels),
                    backup_status="LOCAL_ONLY", interpretation="BOUNDARY_EVIDENCE_NOT_ROOT_CAUSE_PROOF")
    save_json(ROOT / ("data/manifests/gpu_operator_compare_evidence_20261011_"+case+".json"),evidence)
    print(json.dumps(evidence),flush=True)
    if mismatches:
        raise ValueError("External mirror differs from host evidence")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("case",choices=["E27","E28","E29"])
    record(parser.parse_args().case)
