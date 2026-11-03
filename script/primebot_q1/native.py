"""Verify the actual Primebot V3 simulator and ten-substep torque callback."""

import argparse
import json
from pathlib import Path
import inspect
import sys

from common import require, save


def verify(config, dataset, output):
    import torch
    import numpy as np
    import warp as wp
    from supermimic.data import Q1Motions
    from supermimic.robot import Q1Robot
    from supermimic.simulator import Q1Simulator, torque_speed_limits
    torch.manual_seed(0)
    robot = Q1Robot(Path(config["xml"]), reward_scaling="q1_proportional",
                    modeled_action_delay=True).to("cuda:0")
    paths = sorted((Path(dataset) / "accepted").glob("*/*.npz"))
    motions = Q1Motions(robot, paths[:1], device="cuda:0", storage="device-fp16")
    reference = motions.load(0)[:1].expand(16, -1).contiguous()
    simulator = Q1Simulator(robot, 16, "cuda:0", .5, use_tn=True, physics_timestep=.002,
                            capture_reset=True)
    ids = torch.arange(16, device="cuda:0")
    simulator.reset(ids, reference, randomize=False)
    require(simulator.substeps == 10, "Q1 must execute ten physics substeps per control step")
    simulator.context[:, simulator.layout.delay_s] = (ids % 10)[:, None] * .002
    source_lines, first = inspect.getsourcelines(Q1Simulator.step.__wrapped__)
    audit_line = first + next(i for i, line in enumerate(source_lines)
                             if "if self._step_graph is None:" in line)
    audit = dict(calls=0, max_torque_error_nm=0., max_target_error_rad=0.)

    def trace(frame, event, arg):
        if event == "line" and frame.f_code.co_filename == inspect.getfile(Q1Simulator) and frame.f_lineno == audit_line:
            values = frame.f_locals
            substep = values["substep"]
            positions = simulator.qpos[:, 7:].cpu().numpy().copy()
            velocities = simulator.qvel[:, 6:].cpu().numpy().copy()
            context = simulator.context.cpu().numpy()
            zero = robot.default_q.cpu().numpy() + context[:, simulator.layout.motor_offset]
            target = zero + .5 * values["actions"].cpu().numpy()
            previous = zero + .5 * simulator.last_action.cpu().numpy()
            delay = np.rint(context[:, simulator.layout.delay_s] / .002).astype(int)
            expected_target = np.where(substep >= delay, target, previous)
            pd = (simulator.kp.cpu().numpy() * context[:, simulator.layout.kp_factor]
                  * (expected_target - positions) - simulator.kd.cpu().numpy()
                  * context[:, simulator.layout.kd_factor] * velocities)
            peaks = simulator.torque_peaks.cpu().numpy()
            knees = simulator.torque_knees.cpu().numpy()
            cutoffs = simulator.torque_cutoffs.cpu().numpy()
            high = np.minimum(peaks, np.maximum(0, peaks * (cutoffs - velocities) / (cutoffs - knees)))
            low = -np.minimum(peaks, np.maximum(0, peaks * (cutoffs + velocities) / (cutoffs - knees)))
            expected = np.clip(pd, low, high)[:, robot.actuator_qindex.cpu().numpy()]
            expected /= robot.actuator_gear.cpu().numpy()
            expected = np.clip(expected, robot.ctrl_limits[:, 0].cpu().numpy(), robot.ctrl_limits[:, 1].cpu().numpy())
            actual = wp.to_torch(simulator.warp_data.ctrl).cpu().numpy()
            torque_error = float(np.max(np.abs(actual - expected)))
            target_error = float(np.max(np.abs(values["applied"].cpu().numpy() - expected_target)))
            audit["calls"] += 1
            audit["max_torque_error_nm"] = max(audit["max_torque_error_nm"], torque_error)
            audit["max_target_error_rad"] = max(audit["max_target_error_rad"], target_error)
            require(torque_error < 1e-4 and target_error < 1e-6,
                    "Actual substep PD, speed-limited torque or delay oracle failed")
        return trace

    samples = []
    prior_trace = sys.gettrace()
    try:
        sys.settrace(trace)
        for step in range(24):
            action = torch.full((16, 22), .1 if step % 2 else -.1, device="cuda:0")
            before = simulator.state().clone()
            after = simulator.step(action)
            require(after.shape == (16, 79) and bool(torch.isfinite(after).all()),
                    "Actual simulator returned an invalid physical/action state")
            require(torch.equal(simulator.last_action, action), "Action memory update failed")
            require(torch.equal(simulator.context[:, simulator.layout.previous_executed_action], action),
                    "Context action memory update failed")
            observation = robot.observation(after, simulator.context)
            require(observation.shape == (16, 72) and bool(torch.isfinite(observation).all()),
                    "Actual Primebot observation contract failed")
            samples.append(dict(step=step, max_state_change=float((after - before).abs().max())))
    finally:
        sys.settrace(prior_trace)
    require(audit["calls"] == 240, "Did not audit every actual physics substep")
    # Independent directional motor-curve oracle on every native joint.
    velocity = torch.linspace(-20, 20, 128, device="cuda:0")[:, None].expand(-1, 22)
    low, high = torque_speed_limits(velocity, simulator.torque_peaks,
                                   simulator.torque_knees, simulator.torque_cutoffs)
    peak, knee, cutoff = simulator.torque_peaks, simulator.torque_knees, simulator.torque_cutoffs
    expected_high = torch.minimum(peak, (peak * (cutoff - velocity) / (cutoff - knee)).clamp_min(0))
    expected_low = -torch.minimum(peak, (peak * (cutoff + velocity) / (cutoff - knee)).clamp_min(0))
    error = max(float((low - expected_low).abs().max()), float((high - expected_high).abs().max()))
    require(error <= 1e-4, "Primebot torque-speed curve differs from independent oracle")
    torch.cuda.synchronize()
    save(output, dict(status="PASSED", robot="q1_v3", environments=16, control_steps=24,
         environment_transitions=384, physics_substep_batches=240, new_optimizer_updates=0,
         motor_curve_max_error_nm=error, actual_substep_control=audit,
         samples=samples, asset_sha256=robot.asset_sha256,
         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
         peak_reserved_bytes=torch.cuda.max_memory_reserved(),
         quality_claim="native execution only, not trained gait"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "dataset", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    verify(json.loads(args.config.read_text()), args.dataset, args.output)
