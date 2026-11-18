"""Compare the new streaming adapter with saved audited latest-FGP output."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .engine import LatestFgp


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--clotho-root", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=4800)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.frames <= 128 or args.output.exists():
        parser.error("use 1..128 frames and a new output")
    began = time.perf_counter()
    engine = LatestFgp(args.clotho_root)
    calibration = json.loads((args.recording/"tpose_calibration.json").read_text())["data"]["calibration_data"]
    engine.reset(calibration)
    outputs=[]
    with (args.recording/"frames.jsonl").open() as stream:
        for index,line in enumerate(stream):
            if index < args.start_frame-29:
                continue
            if index >= args.start_frame+args.frames:
                break
            imu=json.loads(line)["raw_imu"]
            values=np.concatenate([np.array(imu["upper_ordered_6x7"], dtype=np.float32),
                                   np.array(imu["lower_ordered_5x7"], dtype=np.float32)])
            result=engine.step(values)
            if index>=args.start_frame:
                if result is None:
                    raise ValueError("unexpected history warmup")
                outputs.append(result)
    reference=np.load(args.reference,allow_pickle=False)
    if len(outputs)!=args.frames:
        raise ValueError("not enough raw frames")
    pose=np.stack([r["pose_mat"] for r in outputs])
    root=np.stack([r["root_translation_m"] for r in outputs])
    pose_diff=float(np.max(np.abs(pose-reference["pose_mat"][:len(pose)])))
    root_diff=float(np.max(np.abs(root-reference["root_translation_m"][:len(root)])))
    passed=pose_diff<2e-5 and root_diff<2e-5
    result=dict(status="STREAMING_FGP_PARITY_PASSED" if passed else "FAILED", frames=len(outputs),
                pose_matrix_max_abs_difference=pose_diff, root_m_max_abs_difference=root_diff,
                tolerance=2e-5, wall_s=time.perf_counter()-began,
                reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                torch_version=engine.torch.__version__, device="cpu", hardware_commands=0,
                globalpose_integration_dt_s=float(engine.gp.dt),
                interpretation="Recorded-input software parity only; no live BLE or Orin performance claim")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
    if not passed:
        raise SystemExit(1)


if __name__=="__main__":
    main()
