"""Package existing joint-run evidence with per-member hashes; no inference."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

from script.x2_fgp_realtime.assets import file_digest


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--project", type=Path, default=Path.cwd())
    a = p.parse_args()
    root = a.project.resolve()
    base = root/"data/analysis/X2_JOINT_LATENCY_20261113"
    target = root/"data/deliveries/X2_JOINT_LATENCY_EVIDENCE_20261113.zip"
    manifest_path = root/"data/manifests/x2_joint_latency_evidence_20261113.json"
    if target.exists() or manifest_path.exists():
        p.error("Preserve prior deliveries")
    files = {}
    for name in ("README_X2_JOINT_LATENCY_20261113.md", "README_X2_JOINT_LATENCY_RESULT_20261113.md"):
        files["reports/"+name] = (root/"reports"/name).read_bytes()
    for folder, prefix in ((base/"analysis_r3_v2", "fifo"), (base/"analysis_r4", "latest")):
        for name in ("six_panel.png", "six_panel.pdf", "summary.json", "fgp_timing.csv", "policy_timing.csv"):
            files[prefix+"/"+name] = (folder/name).read_bytes()
    for directory, label in ((base/"received_r3", "raw_r3"), (base/"received_r4", "raw_r4")):
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                files[label+"/"+str(path.relative_to(directory))] = path.read_bytes()
    for path in sorted((root/"script/x2_joint_latency").glob('*.py')):
        files['code/'+path.name] = path.read_bytes()
    files["fixture/fixture.json"] = (base/"fixture_r1/fixture.json").read_bytes()
    files["fixture/imu929.npz"] = (base/"fixture_r1/imu929.npz").read_bytes()
    brief = """# X2 FGP + Morph Joint Latency Evidence

Recorded clothing IMU released at ideal 30Hz; real X2 ROS feedback;
actual GPU/CPU FGP, CPU Morph, unchanged native observe backend.
Motor commands, SDK publishing, MC changes and new BLE connections: zero.

Latest-input run: 537 human outputs, 17.805Hz, 363/929 raw samples skipped.
Oldest human reference age at backend receive: median conservatively
296.93--310.46ms, p95 341.15--356.36ms, including 180ms preview span.
These are software reference ages, not measured BLE or mechanical latency.
FGP p50/p95 41.56/63.42ms. Morph all-ready RPC p50/p95 12.63/22.96ms;
133 of 1456 ready requests exceeded the original 20ms limit and were refused.
Skipping samples preserves freshness but does not establish pose quality
for the unchanged 30Hz recurrent and physics models.

The FIFO diagnostic processed all inputs and accumulated about 21s of lag.
The author's real-time entry already uses latest input; FIFO is not its latency.

- [Latest-input six-panel figure](latest/six_panel.png)
- [Latest-input numeric summary](latest/summary.json)
- [Latest-input per-request timing](latest/policy_timing.csv)
- [FIFO diagnostic figure](fifo/six_panel.png)
- [FIFO numeric summary](fifo/summary.json)

Raw JSONL, finite SMPL outputs, failure logs, source integrity checks,
fixture metadata and the exact experiment code are included. Weights and
engines are excluded. Reports retain original project-relative links;
use the links above when opening this extracted standalone archive.
This package contains no new training or video. Netdisk backup is unverified.
"""
    files["README.md"] = brief.encode()
    for link in re.findall(r'\]\(([^)]+)\)', brief):
        if link not in files:
            raise ValueError("Missing standalone delivery link: "+link)
    inventory = [dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                 for name, data in sorted(files.items())]
    files["manifest.json"] = (json.dumps(dict(files=inventory), indent=2)+"\n").encode()
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, data in sorted(files.items()):
            z.writestr(name, data)
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None or len(z.namelist()) != len(files):
            raise ValueError("ZIP CRC or member count failed")
        for row in inventory:
            data = z.read(row['path'])
            if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                raise ValueError("ZIP member mismatch")
    record = dict(status="JOINT_EVIDENCE_ZIP_VERIFIED", path=str(target.relative_to(root)),
        bytes=target.stat().st_size, sha256=file_digest(target), members=len(files),
        relative_links_checked=5, crc_passed=True, member_size_sha256_passed=True,
        generated_from_saved_results=True, new_inference_calls=0, hardware_commands=0,
        members_manifest=inventory, netdisk_backup="LOCAL_ONLY_NOT_VERIFIED")
    manifest_path.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({k:v for k,v in record.items() if k != "members_manifest"}))


if __name__ == "__main__":
    main()
