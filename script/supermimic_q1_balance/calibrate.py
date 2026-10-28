"""Calibrate fallback only on saved RF11 training clips, with no new physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_ppo"))
from resources import identity, save, stamp
from control import risk_features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    source = json.loads(a.library.read_text())
    rows = [dict(row, split=split) for split in ("train", "validation")
            for row in source["splits"][split]]
    selected = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    if len(selected) != 300:
        raise ValueError("Calibration requires only the original 300 training clips")
    samples = []
    for repeat in range(3):
        path = a.baseline / f"RF11_baseline_repeat{repeat}.npz"
        with np.load(path, allow_pickle=False) as saved:
            active = saved["alive"][:, selected].astype(bool)
            values = [torch.from_numpy(saved[key][:, selected][active]) for key in
                      ("states", "reference_p", "reference_quat", "force", "slip")]
            samples.append(risk_features(*values).numpy())
    sample = np.concatenate(samples)
    if not np.isfinite(sample).all():
        raise ValueError("Nonfinite RF11 calibration sample")
    floors = np.array([.15, .20, 2., .15])
    high = np.maximum(np.quantile(sample, .95, axis=0), floors)
    high[0] = min(high[0], .25)
    save(a.output, {"status": "TRAINING_ONLY_FALLBACK_CALIBRATED", "created_utc": stamp(),
        "thresholds": high.tolist(), "risk_order": ["root_XY_L2_m", "tilt_rad", "root_angular_XY_rad_s", "max_loaded_slip_m_s"],
        "train_clips": len(selected), "validation_clips_used": 0, "samples": len(sample),
        "quantile": .95, "minimum_scales": floors.tolist(), "root_trigger_ceiling_m": .25,
        "empirical_quantiles": {str(q): np.quantile(sample, q, axis=0).tolist() for q in (.5, .9, .95, .99)},
        "baseline_traces": [identity(a.baseline / f"RF11_baseline_repeat{r}.npz") for r in range(3)],
        "library": identity(a.library), "optimizer_steps": 0, "new_physical_steps": 0,
        "protocol": "Equal sampled control frames from saved alive training trajectories. EMA tau0.1s, high1/low0.7, cooldown10steps; alpha down1.5/s, up0.25/s; heuristic, not safety certification."})
    print(a.output.read_text())


if __name__ == "__main__":
    main()
