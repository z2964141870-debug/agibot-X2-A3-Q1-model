"""Fixed-method TEST_ONLY arrival replay; no policy, fitting or physical latency.

The 40 ms interpolation buffer and other method parameters are fixed in source,
not selected on these recordings. A synthetic missing-packet interval is clearly
separate from recorded transport. No recorded sample becomes training data.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import time

import numpy as np

from causal_resampler import ArrivedPose, CausalResampler, IntakeSource
from causal_tail import multiply, rotation_quaternion, rotation_vector, unit
from recorded_intake_audit import describe, validated_times
from stream_contract import DT_NS, HumanPose


SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
CONFIGURATIONS = (("hold_now", "hold", 0), ("cv_now", "cv", 0),
                  ("damped_now", "damped_cv", 0), ("linear_buffer40", "delayed_linear", 40_000_000))
SCENARIOS = (("recorded_arrivals", None), ("synthetic_drop_30s_250ms", (30_000_000_000, 250_000_000)))


def distribution(values):
    if not values:
        return None
    return dict(zip(("p50", "p95", "p99", "maximum"),
                    (float(value) for value in (*np.percentile(values, (50, 95, 99)), max(values)))))


class RecordedInput:
    def __init__(self, path):
        self.path = Path(path)
        self.input_identity = describe(self.path)
        self.inspection_path = self.path.with_name("inspection.json")
        self.inspection_identity = describe(self.inspection_path)
        inspection = json.loads(self.inspection_path.read_text())
        if (inspection.get("usage") != "TEST_ONLY" or inspection.get("training_allowed") is not False
                or inspection.get("bundle", {}).get("sha256") != self.input_identity["sha256"]):
            raise ValueError("Require unchanged TEST_ONLY bundle and extraction inspection")
        with np.load(self.path, allow_pickle=False) as data:
            if data["usage"].item() != "TEST_ONLY" or data["training_allowed"].item() is not False:
                raise ValueError("This replay cannot relabel test data")
            if tuple(data["parents"].tolist()) != SMPL_PARENTS:
                raise ValueError("Require the recorded SMPL24 parent topology")
            self.host_s = validated_times(data["host_monotonic_s"])
            self.payload_s = validated_times(data["source_time_s"])
            self.sequence = data["frame_index"].copy()
            self.positions = data["joints"].astype(float)
            self.rotvecs = data["axis_angle"].astype(float).reshape(-1, 24, 3)
            self.translation_all_zero = bool(np.all(data["trans"] == 0))
        frames = len(self.host_s)
        if (len(self.payload_s) != frames or self.sequence.shape != (frames,)
                or self.sequence.dtype.kind not in "iu" or np.any(np.diff(self.sequence) <= 0)
                or self.positions.shape != (frames, 24, 3) or self.rotvecs.shape != (frames, 24, 3)
                or not np.isfinite(self.positions).all() or not np.isfinite(self.rotvecs).all()):
            raise ValueError("Unexpected recorded pose/frame schema")
        self.arrivals = [int(round(float(value)*1e9)) for value in self.host_s]
        self.payloads = [int(round(float(value)*1e9)) for value in self.payload_s]
        self.source = IntakeSource("egolocate_fgp", self.path.parent.name,
            "recorded_payload_time:"+self.path.parent.name,
            "recorded_host_monotonic:"+self.path.parent.name)

    def packet(self, index):
        # This per-frame conversion has no temporal filter or future dependency.
        local = [rotation_quaternion(vector) for vector in self.rotvecs[index]]
        world = [local[0]]
        for joint, parent in enumerate(SMPL_PARENTS[1:], 1):
            world.append(unit(multiply(world[parent], local[joint])))
        return ArrivedPose(self.source, int(self.sequence[index]), self.payloads[index],
                           self.arrivals[index], HumanPose(self.positions[index], world))

    def verify(self):
        if (describe(self.path) != self.input_identity
                or describe(self.inspection_path) != self.inspection_identity):
            raise RuntimeError("Recorded input changed during replay")


def replay_recording(recording, drop_window=None):
    configurations = {name: (CausalResampler(recording.source), method, buffer_ns)
                      for name, method, buffer_ns in CONFIGURATIONS}
    rows = {name: dict(status=Counter(), provenance=Counter(), ages=[], support_wait=[],
        sample_ms=[], steps_mm=[], steps_rad=[], generated=0, capped=0, available=0,
        causal_violations=0, examples=[]) for name in configurations}
    previous = {name: None for name in configurations}
    origin, last = recording.arrivals[0], recording.arrivals[-1]
    packet_index, skipped, conversion_ms = 0, 0, []
    ticks = (last-origin)//DT_NS+1
    for tick in range(ticks):
        control = origin+tick*DT_NS
        while packet_index < len(recording.arrivals) and recording.arrivals[packet_index] <= control:
            offset = recording.arrivals[packet_index]-origin
            if drop_window and drop_window[0] <= offset < sum(drop_window):
                skipped += 1
            else:
                started = time.perf_counter_ns()
                packet = recording.packet(packet_index)
                conversion_ms.append((time.perf_counter_ns()-started)/1e6)
                for resampler, _, _ in configurations.values():
                    resampler.ingest(packet)
            packet_index += 1
        for name, (resampler, method, buffer_ns) in configurations.items():
            row = rows[name]
            started = time.perf_counter_ns()
            value = resampler.sample(target_ns=max(0, control-buffer_ns), control_ns=control,
                receiver_clock_id=recording.source.receiver_clock_id, method=method)
            row["sample_ms"].append((time.perf_counter_ns()-started)/1e6)
            row["status"][value.status] += 1
            row["provenance"][value.provenance] += 1
            timing = value.timing()
            row["causal_violations"] += int(not timing["locally_causal"])
            row["generated"] += int(value.generated)
            row["capped"] += int(value.speed_cap_applied)
            if timing["latest_support_arrival_age_ns"] is not None:
                row["ages"].append(timing["latest_support_arrival_age_ns"]/1e6)
                row["support_wait"].append(timing["interpolation_right_support_ns"]/1e6)
            if value.can_step:
                row["available"] += 1
                prior = previous[name]
                if prior is not None:
                    row["steps_mm"].append(float(np.linalg.norm(np.asarray(value.pose.positions)
                        -np.asarray(prior.positions), axis=-1).max()*1000))
                    angles = [rotation_vector(multiply((a[0], -a[1], -a[2], -a[3]), b))
                              for a, b in zip(prior.orientations, value.pose.orientations)]
                    row["steps_rad"].append(float(np.linalg.norm(angles, axis=-1).max()))
                previous[name] = value.pose
            else:
                previous[name] = None
                if len(row["examples"]) < 3:
                    row["examples"].append(timing)
    output = {}
    for name, (resampler, method, buffer_ns) in configurations.items():
        row = rows[name]
        output[name] = dict(method=method, scheduled_buffer_ms=buffer_ns/1e6,
            ticks=ticks, available_ticks=row["available"], availability=row["available"]/ticks,
            status=dict(row["status"]), provenance=dict(row["provenance"]),
            generated_ticks=row["generated"], speed_capped_ticks=row["capped"],
            local_causality_violations=row["causal_violations"],
            latest_used_arrival_age_ms=distribution(row["ages"]),
            interpolation_right_arrival_support_ms=distribution(row["support_wait"]),
            resample_compute_ms=distribution(row["sample_ms"]),
            maximum_joint_position_step_mm=distribution(row["steps_mm"]),
            maximum_joint_rotation_step_rad=distribution(row["steps_rad"]),
            input_counters=resampler.counters, unavailable_examples=row["examples"])
    if any(row["local_causality_violations"] for row in output.values()):
        raise RuntimeError("A resampled value used an unarrived packet")
    return dict(methods=output, deliberately_dropped_packets=skipped,
        synthetic_drop_window_ns=None if drop_window is None else list(drop_window),
        per_packet_axis_angle_to_world_compute_ms=distribution(conversion_ms),
        trailing_silence_added=False)


def run(paths, output):
    paths, output = [Path(path) for path in paths], Path(output)
    if output.exists() or not paths or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Require distinct TEST_ONLY inputs and a new output")
    started = time.perf_counter()
    recordings = []
    for path in paths:
        recording = RecordedInput(path)
        scenarios = {name: replay_recording(recording, window) for name, window in SCENARIOS}
        recording.verify()
        recordings.append(dict(input=recording.input_identity,
            inspection=recording.inspection_identity, source_frames=len(recording.arrivals),
            usage="TEST_ONLY", training_allowed=False, input_integrity="PASSED",
            pose_frame="recorded_SMPL_native_world_axes_not_hardware_calibrated",
            orientation_conversion="per_packet_recorded_local_axis_angle_parent_composition",
            root_translation_all_zero=recording.translation_all_zero,
            payload_clock_synchronized_to_receiver=False,
            upstream_dependency_end_ns=None, scenarios=scenarios))
    code = [Path(__file__), Path(__file__).with_name("causal_resampler.py"),
            Path(__file__).with_name("causal_tail.py"), Path(__file__).with_name("stream_contract.py")]
    result = dict(stage="FIXED_TEST_ONLY_RECORDED_CAUSAL_RESAMPLER_REPLAY", status="COMPLETED",
        recordings=recordings, code=[describe(path) for path in code],
        runtime=dict(python=platform.python_version(), numpy=np.__version__),
        elapsed_seconds=time.perf_counter()-started,
        fixed_parameters=dict(output_hz=50, capacity=256, history_frames=5, tau_s=.12,
            max_linear_speed_m_s=3., max_angular_speed_rad_s=8., max_arrival_age_ms=200,
            max_extrapolation_ms=80, max_support_gap_ms=200),
        fitting_or_hyperparameter_selection=False, training_samples_used=0, optimizer_updates=0,
        new_policy_inference=0, new_physics_steps=0, new_cuda_jobs=0,
        measured_end_to_end_latency_ms=None, measured_pose_tracking_accuracy=None,
        limitations=["Receiver-arrival timeline, not calibrated physical capture",
            "Upstream FGP causality and hardware axes remain unverified",
            "Geometric step metrics are not tracking accuracy or robot naturalness",
            "TEST_ONLY recordings never enter training or parameter selection",
            "250ms packet omission is synthetic, not a recorded network fault",
            "No policy bridge to measured-exact H4/H5 slots was enabled"],
        user_accepted_checkpoint=None)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=describe(output), recordings=len(recordings),
                          elapsed_seconds=result["elapsed_seconds"])))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    run(options.input, options.output)
