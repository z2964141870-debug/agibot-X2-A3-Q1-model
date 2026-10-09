"""Invoke the pinned simulator and optionally collect actual policy inputs."""

import argparse
from pathlib import Path
import sys

import numpy as np

from script.a3.reference_contract import load_sim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--capture-count", type=int, default=100)
    parser.add_argument("sim_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    sim = load_sim()
    observations, actions = [], []
    original = sim.A3Policy.act

    def capture(policy, encoder_input, actor_obs, device):
        action = original(policy, encoder_input, actor_obs, device)
        if len(observations) < args.capture_count:
            # Official ABI flattens each tokenizer feature, then appends proprioception.
            observations.append(np.concatenate([encoder_input[:, :58].reshape(-1),
                                                encoder_input[:, 58:].reshape(-1),
                                                actor_obs.reshape(-1)]).astype(np.float32))
            actions.append(action.copy())
        return action

    if args.capture:
        sim.A3Policy.act = capture
    sys.argv = [str(sim.__file__), *([*args.sim_args[1:]] if args.sim_args[:1] == ["--"] else args.sim_args)]
    try:
        return sim.main()
    finally:
        if args.capture and observations:
            args.capture.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.capture, observations=np.asarray(observations), actions=np.asarray(actions),
                                input_order=np.asarray("command_multi_future_then_ori6d_multi_future_then_proprioception"),
                                usage=np.asarray("ACTUAL_MUJOCO_POLICY_INPUTS"))


if __name__ == "__main__":
    raise SystemExit(main())
