"""Run the complete owner and private stand controller from matched crouches."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from balanced_policy import BalancedTeacher
from common import HERE
from limbs import add_limb_contacts
from low_catch import MidCrouchCatch
from policy import export
import eval_core


class SensedStart(BalancedTeacher):
    """Initialize the rise reference from the measured joint posture."""

    def forward(self, rec_obs_hist, rec_state):
        frame = rec_obs_hist[:, -1]
        joints = frame[:, :22] + self.stand
        distance = (self.reference_q[0] - joints[:, None]).square().mean(-1)
        phase = torch.argmin(distance, dim=1).to(joints.dtype)
        zero = phase * 0
        initial = torch.cat((torch.stack((zero + 60, zero + 4, zero, phase), 1), joints), 1)
        state = torch.where(rec_state[:, :1] == 0, initial, rec_state)
        return super().forward(rec_obs_hist, state)


def run(out, gate):
    out.mkdir(parents=True, exist_ok=False)
    dataset = HERE / "prepared_balanced_02"
    reference = np.load(dataset / "references.npz")
    starts = np.load(dataset / "starts.npz")
    records = starts["valid_meta"]
    order = np.argsort(records[:, 1], kind="stable")
    chosen = order[np.linspace(0, len(order) - 1, 24).round().astype(int)]
    policy = SensedStart(reference).eval()
    expert = out / "expert.onnx"
    export(policy, expert)
    controller = MidCrouchCatch if gate == "early" else eval_core.Q1AutonomousRecovery

    class Owned(controller):
        def reset(self):
            super().reset()
            self.owner = "recovery"
            self.entered = 0.
            self.rise.expert.gate.mode = "recover"

    eval_core.Q1AutonomousRecovery = Owned
    eval_core.bench.scene = add_limb_contacts
    eval_core.bench.poses = lambda *_: [
        ("balanced_crouch", starts["valid"][i, :29].copy()) for i in chosen
    ]
    eval_core.simulate("static", 94003, out / "replay", expert.resolve(), 500, 11, True, 2)
    rows = json.loads((out / "replay/results.json").read_text())
    for i, row in zip(chosen, rows, strict=True):
        row["validation_index"] = int(i)
        row["initial_reference_phase"] = int(records[i, 1])
        row["stage_pass"] = bool(
            row["first_confirmation_s"] is not None
            and row["first_confirmation_s"] <= 15
            and row["solo_calm_s"] >= .5
            and len(row["releases"]) == 1 and not row["takeovers"]
            and row["wrong_release"] == 0 and row["phases"][0]["post_1s"]
        )
    stats = dict(gate=gate, n=len(rows), stage_pass=sum(r["stage_pass"] for r in rows),
                 wrong_releases=sum(r["wrong_release"] for r in rows),
                 releases=sum(len(r["releases"]) for r in rows),
                 successful_phases=[r["initial_reference_phase"] for r in rows if r["stage_pass"]],
                 reference_sha256=hashlib.sha256((dataset/"references.npz").read_bytes()).hexdigest(),
                 starts_sha256=hashlib.sha256((dataset/"starts.npz").read_bytes()).hexdigest())
    (out / "stage_results.json").write_text(json.dumps(rows, indent=2))
    (out / "summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--gate", choices=("standard", "early"), default="standard")
    args = parser.parse_args()
    run(args.out, args.gate)
