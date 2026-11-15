"""Package verified saved tables and raw evidence, without plots or model calls."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

from script.x2_joint_latency.timing import statistics


def stamp(path):
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ("root", "raw", "output", "archive"):
        p.add_argument("--"+name, required=True, type=Path)
    a = p.parse_args()
    if a.output.exists() or a.archive.exists():
        p.error("Preserve previous delivery")
    summary_path = a.root/"analysis_r1/summary.json"
    summary = json.loads(summary_path.read_text())
    if (summary["status"] != "HOST_OPT_SAVED_ANALYSIS_PASSED" or summary["hardware_commands"]
            or not summary["full_sequence_parity"]["passed"] or summary["preview_ms"] != 180):
        raise ValueError("Require complete saved analysis and pose/root parity")
    inputs = [summary_path, a.root/"analysis_r1/frames.csv", a.raw]
    for entry in summary["source_files"]:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = a.root.resolve().parents[2]/path
        actual = stamp(path)
        if any(actual[k] != entry[k] for k in ("bytes", "sha256")):
            raise ValueError("Saved analysis input changed")
        inputs.append(path)
    raw = stamp(a.raw)
    if raw["bytes"] != 7145510 or raw["sha256"] != "a19ac2dec250ba37b49906f0d0e7e19fc614d644749a13cccd2530606c59c28c":
        raise ValueError("Wrong verified raw archive")
    before = [stamp(path) for path in inputs]
    copies, stages, comparison = {}, [], []
    for variant, r in summary["variants"].items():
        profile = [json.loads(s) for s in (a.root/(variant+"_joint")/"profile.jsonl").read_text().splitlines()]
        selected = [x for x in profile if x["stage"] == "cuda_memcpy_host"]
        if len(selected) != 2700 or {x["direction"] for x in selected} != {1, 2}:
            raise ValueError("Wrong CUDA copy accounting")
        copies[variant] = {}
        for direction, name, count in ((1, "h2d", 1800), (2, "d2h", 900)):
            values = [x for x in selected if x["direction"] == direction]
            if len(values) != count:
                raise ValueError("Missing direction events")
            copies[variant][name] = {k: statistics([x[k] for x in values]) for k in ("wall_ms", "thread_cpu_ms")}
            copies[variant][name]["byte_sizes"] = sorted({x["bytes"] for x in values})
        for group in ("profile_stages", "fgp_stages_ms", "policy_stages_ms"):
            for stage, fields in r[group].items():
                if group != "profile_stages":
                    fields = {"wall_ms": fields}
                for field, v in fields.items():
                    stages.append([variant, group, stage, field, *[v[k] for k in ("count", "p50", "p95", "p99", "maximum")]])
        comparison.append([variant, r["outputs"], r["skipped"], r["human_hz"], r["source_to_ack_ms"]["p95"],
            r["output_interval_ms"]["p95"], r["output_interval_ms"]["over_budget"], r["rpc_ready_ms"]["p95"],
            r["rpc_ready_ms"]["p99"], r["rpc_ready_ms"]["maximum"], r["ready"], r["accepted"],
            *[r["deadline_exceed"][k] for k in ("20", "25", "30")], r["reference_age_upper_ms"]["p95"]])
    a.output.mkdir(parents=True)
    for name, header, rows in (
        ("stages.csv", ["variant", "group", "stage", "measure", "count", "p50", "p95", "p99", "maximum"], stages),
        ("comparison.csv", ["variant", "outputs", "skipped", "human_hz", "ack_p95_ms", "interval_p95_ms", "interval_over_33ms",
                            "rpc_p95_ms", "rpc_p99_ms", "rpc_max_ms", "ready", "accepted", "over20", "over25", "over30", "reference_upper_p95_ms"], comparison)):
        with (a.output/name).open("x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
    (a.output/"copy_direction.json").write_text(json.dumps(copies, indent=2)+"\n")
    (a.output/"sources.json").write_text(json.dumps(before, indent=2)+"\n")
    (a.output/"README.md").write_text("""# X2 performance comparison, 2026-11-15

Two new 64-output gates and two 900-output joint tests completed. Recorded IMU
was released at 30 Hz, with live read-only feedback. All 900 selected frame IDs
matched; pose/root differences were below 2e-5. Models, physics and 180 ms preview
were preserved. No motors, MC changes, live BLE reconnection, training or adoption.

- [Comparison](comparison.csv)
- [Stages](stages.csv)
- [CUDA direction timing](copy_direction.json)
- [Verified analysis](summary.json)
- [Frames](frames.csv)
- [Original device evidence](X2_JOINT_HOST_OPT_RAW_20261115.tgz)

Dedicated hosts: ACK p95 63.07 -> 59.73 ms; Morph RPC p95 15.19 -> 14.31 ms;
RPC >20 ms 10/1491 -> 4/1492. Mean information rate stayed about 29.98 Hz.
Spacing >33.33 ms worsened 536/899 -> 566/899; maximum RPC 31.21 -> 32.20 ms.
Short sequential equally instrumented tests do not certify stable 30 Hz or
causal allocation benefits. Historical uninstrumented R2 had two timeouts;
the new four do not beat that result. Whole-host counters include shared users.
CUDA stream spans include queue/dispatch gaps; nested timings must not be summed.
D2H host waiting is not a removable 12 ms copy cost. Reference ages include
180 ms preview and are not BLE-to-mechanical latency. Performance uses tables;
six-panel figures are required for training, per the latest user preference.
""")
    payload = {f.name: f for f in sorted(a.output.iterdir())}
    payload.update({"summary.json": summary_path, "frames.csv": a.root/"analysis_r1/frames.csv", a.raw.name: a.raw})
    manifest = [dict(name=n, bytes=stamp(f)["bytes"], sha256=stamp(f)["sha256"]) for n, f in payload.items()]
    manifest_path = a.output/"manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    payload["manifest.json"] = manifest_path
    with zipfile.ZipFile(a.archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name, path in payload.items():
            z.write(path, name)
    with zipfile.ZipFile(a.archive) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP CRC failed")
        for entry in manifest:
            data = z.read(entry["name"])
            if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError("ZIP content mismatch")
        links = re.findall(r"\]\(([^)]+)\)", z.read("README.md").decode())
        if any(link not in z.namelist() for link in links):
            raise ValueError("Broken portable relative link")
    if before != [stamp(path) for path in inputs]:
        raise ValueError("Source changed during delivery")
    receipt = dict(status="HOST_OPT_TABLE_DELIVERY_PASSED", archive=stamp(a.archive), members=len(payload),
                   verified_links=len(links), new_plots=0, new_model_calls=0, hardware_commands=0)
    (a.output/"delivery.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
