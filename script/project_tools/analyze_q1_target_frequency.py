"""Read saved trajectories only; separate target bandwidth from measured joint motion."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

FIXED = ("walk_ff_stop_180_R_002__A095", "walk_ff_start_180_R_002__A071", "triumph_two_handed_001__A058")


def identity(path):
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def bandwidth(value):
    if len(value) < 4:
        raise ValueError("Too few physical samples")
    value = value.astype(np.float64)
    centered = (value-value.mean(0)) * np.hanning(len(value))[:, None]
    power = np.abs(np.fft.rfft(centered, axis=0))**2
    frequencies = np.fft.rfftfreq(len(value), .02)
    total = power[frequencies > 0].sum()
    return {str(hz)+"_hz_fraction": float(power[frequencies >= hz].sum()/max(total, 1e-20))
            for hz in (5, 10, 15)}


def main(args):
    cases = [
        (args.phase_root, "phase_baseline", "baseline_repeat{repeat}.npz"),
        (args.rf15_root/"evaluation", "rf15_static", "initial_static075_repeat{repeat}.npz"),
        (args.rf15_root/"evaluation", "rf15_head", "stage_01_repeat{repeat}.npz"),
    ]
    rows, sources = [], []
    for root, label, pattern in cases:
        library = json.loads((root/"evaluation_library.json").read_text())
        names = [row["name"] for row in library["splits"]["train"]]
        sources.append(identity(root/"evaluation_library.json"))
        for repeat in range(3):
            path = root/pattern.format(repeat=repeat)
            sources.append(identity(path))
            with np.load(path, allow_pickle=False) as trace:
                for name in FIXED:
                    index = names.index(name)
                    length = int(trace["steps"][index])
                    targets = trace["targets"][:length, index, :12]
                    measured = trace["states"][:length, index, 13:25]
                    if not np.isfinite(targets).all() or not np.isfinite(measured).all():
                        raise FloatingPointError("Nonfinite saved target or state")
                    changes = np.diff(targets, axis=0)
                    rows.append(dict(label=label, repeat=repeat, name=name, physical_steps=length,
                        target_delta_rms_rad=float(np.sqrt(np.mean(changes**2))),
                        target_delta_abs_p99_rad=float(np.quantile(np.abs(changes), .99)),
                        fraction_joint_target_changes_over_point1rad=float(np.mean(np.abs(changes)>.1)),
                        target_poststep_joint_rmse_rad=float(np.sqrt(np.mean((targets-measured)**2))),
                        target_bandwidth=bandwidth(targets), actual_joint_bandwidth=bandwidth(measured)))
    result = dict(status="SAVED_TARGET_BANDWIDTH_DIAGNOSIS", captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rows=rows, sources=sources, new_physical_steps=0, policy_inference_calls=0, optimizer_updates=0,
        limitations=[
            "PSD is demeaned Hann-windowed50Hz joint positions pooled over12 legs; fractions are not force spectra.",
            "PD error uses post20ms state, not internal5ms actuator error or actual motor torque.",
            "Closed-loop target/state association cannot identify causal actuator delay.",
            "Same checkpoint repeated; existing validation/search has no independent held-out claim.",
        ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        print(json.dumps(result, indent=2, allow_nan=False), file=stream)
    print(json.dumps({"status":result["status"], "rows":rows}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase-root", type=Path, required=True)
    p.add_argument("--rf15-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
