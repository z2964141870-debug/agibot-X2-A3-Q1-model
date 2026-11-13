"""Extract a bounded contiguous real IMU recording; no pose labels are consumed."""
import argparse
import json
from pathlib import Path
import numpy as np
from script.x2_fgp_realtime.assets import file_digest


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--recording", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source = a.recording / "frames.jsonl"
    calibration = a.recording / "tpose_calibration.json"
    raw, original_times = [], []
    with source.open() as stream:
        for i, line in enumerate(stream):
            if i == 929:
                break
            row = json.loads(line)
            imu = row["raw_imu"]
            raw.append(np.concatenate((imu["upper_ordered_6x7"], imu["lower_ordered_5x7"])))
            original_times.append(row["monotonic_s"])
    raw = np.asarray(raw, dtype=np.float32)
    cal = np.asarray(json.loads(calibration.read_text())["data"]["calibration_data"], dtype=np.float32)
    if raw.shape != (929, 10, 7) or cal.shape != (120,) or not np.isfinite(raw).all() or not np.isfinite(cal).all():
        raise ValueError("Invalid bounded recording")
    target = a.output / "imu929.npz"
    np.savez_compressed(target, raw=raw, calibration=cal, original_times=np.asarray(original_times))
    receipt = dict(status="CONTIGUOUS_IMU_FIXTURE_PREPARED", frames=929,
                   source_sha256=file_digest(source), calibration_sha256=file_digest(calibration),
                   fixture_sha256=file_digest(target), fixture_bytes=target.stat().st_size,
                   scheduling="Replay at ideal 30Hz; original arrival jitter is NOT reproduced",
                   inference_calls=0, hardware_commands=0)
    (a.output / "fixture.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
