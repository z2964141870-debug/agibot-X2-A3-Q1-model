"""Create or verify a short private recorded-input CPU fixture; never opens BLE."""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--make",action="store_true")
    parser.add_argument("--recording",type=Path)
    parser.add_argument("--reference",type=Path)
    parser.add_argument("--fixture",type=Path,required=True)
    parser.add_argument("--clotho-root",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.make:
        rows=[]
        with (args.recording/"frames.jsonl").open() as stream:
            for index,line in enumerate(stream):
                if index<4771:
                    continue
                if index>=4832:
                    break
                imu=json.loads(line)["raw_imu"]
                rows.append(np.concatenate([np.array(imu["upper_ordered_6x7"],dtype=np.float32),
                                            np.array(imu["lower_ordered_5x7"],dtype=np.float32)]))
        cal=json.loads((args.recording/"tpose_calibration.json").read_text())["data"]["calibration_data"]
        ref=np.load(args.reference,allow_pickle=False)
        args.fixture.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.fixture,raw=np.stack(rows),calibration=np.array(cal),
                            pose=ref["pose_mat"][:32],root=ref["root_translation_m"][:32])
        return
    if not args.clotho_root or not args.output or args.output.exists():
        parser.error("verify needs --clotho-root and a new --output")
    from .engine import LatestFgp
    fixture=np.load(args.fixture,allow_pickle=False)
    began=time.perf_counter()
    engine=LatestFgp(args.clotho_root)
    engine.reset(fixture["calibration"])
    results=[]
    for row in fixture["raw"]:
        result=engine.step(row)
        if result is not None:
            results.append(result)
    pose=np.stack([row["pose_mat"] for row in results])
    roots=np.stack([row["root_translation_m"] for row in results])
    pose_error=float(np.max(np.abs(pose-fixture["pose"])))
    root_error=float(np.max(np.abs(roots-fixture["root"])))
    times=np.array([row["inference_ms"] for row in results])
    passed=len(results)==32 and pose_error<2e-5 and root_error<2e-5
    report=dict(status="RECORDED_CPU_FIXTURE_PASSED" if passed else "FAILED",frames=len(results),
                pose_matrix_max_abs_difference=pose_error,root_m_max_abs_difference=root_error,tolerance=2e-5,
                inference_ms_all=dict(p50=float(np.median(times)),p95=float(np.percentile(times,95)),max=float(times.max())),
                inference_ms_after_first2=dict(p50=float(np.median(times[2:])),p95=float(np.percentile(times[2:],95)),max=float(times[2:].max())),
                wall_s=time.perf_counter()-began,torch_version=engine.torch.__version__,device="cpu",
                ble_connections=0,hardware_commands=0,feedback_subscriptions=0,
                interpretation="Recorded inputs only; excludes BLE, mesh generation, HTTP, browser and hardware latency")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))
    if not passed:
        raise SystemExit(1)


if __name__=="__main__":
    main()
