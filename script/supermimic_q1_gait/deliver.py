"""RF14 six-panel dashboard and fixed54s videos from saved physical traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_ppo"))
from resources import save, stamp, identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "trial", "video-suite"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    status = a.trial / "delivery_status.json"
    save(status, {"status": "RUNNING", "phase": "dashboard", "updated_utc": stamp()})
    subprocess.run([sys.executable, str(HERE.parent / "supermimic_q1_hp/plot_trial.py"),
                    "--run", str(a.trial / "run"), "--output", str(a.trial / "dashboard")], check=True)
    sys.path[:0] = [str(a.runtime / "src"), str(a.runtime), str(HERE.parent / "supermimic_q1_hp")]
    import torch
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
    from video_suite import render_suite
    torch.set_num_threads(8)
    evaluation = a.trial / "evaluation"
    receipt = json.loads((evaluation / "receipt.json").read_text())
    selection = json.loads((a.trial / "run/checkpoint_selection.json").read_text())
    rows = receipt["rows"]
    kin = Q1Kinematics(a.runtime / "q1_assets/scene.xml", "cpu")
    motion = MotionLib([row["path"] for row in rows], kin, "cpu", src_fps=30.)
    env = SimpleNamespace(supermimic_mdp=SimpleNamespace(motion=motion), device=torch.device("cpu"))
    suite = json.loads(a.video_suite.read_text())
    media = {}
    for label, source in (("initial", receipt["initial_traces"][0]),
                          ("final", receipt["final_traces"][0]),
                          ("selected", selection["selected"]["traces"][0])):
        save(status, {"status": "RUNNING", "phase": label + "_video", "updated_utc": stamp()})
        with np.load(source, allow_pickle=False) as saved:
            trace = {key:saved[key] for key in ("initial", "states", "steps", "failed")}
        media[label] = render_suite(a.runtime, trace, env, rows, suite,
                                    evaluation / (label + ".mp4"), "RF14 " + label)
        save(evaluation / (label + "_media.json"), media[label])
    save(status, {"status": "COMPLETED", "updated_utc": stamp(), "new_optimizer_steps": 0,
                  "new_physical_steps": 0, "media": media,
                  "dashboard": identity(a.trial / "dashboard/receipt.json")})


if __name__ == "__main__":
    main()
