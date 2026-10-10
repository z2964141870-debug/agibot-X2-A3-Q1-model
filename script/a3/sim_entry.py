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
    parser.add_argument("--reference-buffer-ms", type=float, default=0)
    parser.add_argument("--buffer-trace", type=Path)
    parser.add_argument("--joint-reference-mode", choices=("oracle_aligned", "hold", "cv", "smooth_cv", "E12", "E14"))
    parser.add_argument("--joint-predictor", type=Path)
    parser.add_argument("--joint-noise-std", type=float, default=0)
    parser.add_argument("--joint-noise-seed", type=int, default=0)
    parser.add_argument("--joint-trace", type=Path)
    parser.add_argument("sim_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.reference_buffer_ms and args.buffer_trace is None:
        parser.error("--reference-buffer-ms requires --buffer-trace")
    if args.joint_reference_mode and (args.joint_trace is None or args.reference_buffer_ms):
        parser.error("Joint ablation needs its own trace and cannot combine with buffering")
    sim = load_sim()
    replay = None
    if args.reference_buffer_ms:
        from script.a3.causal_policy import CausalPolicyReplay
        tokens = args.sim_args
        fps = float(tokens[tokens.index("--csv-source-fps") + 1])
        stride = int(tokens[tokens.index("--csv-frame-stride") + 1])
        replay = CausalPolicyReplay(sim, fps, stride, args.reference_buffer_ms)
        replay.install()
    if args.joint_reference_mode:
        from script.a3.joint_forecast_replay import JointForecastReplay
        replay = JointForecastReplay(sim, args.joint_reference_mode, args.joint_predictor,
                                     args.joint_noise_std, args.joint_noise_seed)
        replay.install()
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
        if replay and replay.rows:
            replay.save(args.joint_trace if args.joint_reference_mode else args.buffer_trace)
        if args.capture and observations:
            args.capture.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.capture, observations=np.asarray(observations), actions=np.asarray(actions),
                                input_order=np.asarray("command_multi_future_then_ori6d_multi_future_then_proprioception"),
                                usage=np.asarray("ACTUAL_MUJOCO_POLICY_INPUTS"))


if __name__ == "__main__":
    raise SystemExit(main())
