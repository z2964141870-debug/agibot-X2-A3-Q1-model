"""Audit recorded TEST_ONLY mocap timing without claiming physical latency.

Payload timestamps and receiver timestamps are separate clock domains. Offline
linear/Slerp interpolation waits for its right support sample. This diagnostic
quantifies that dependency in the payload clock, without synchronizing clocks,
fitting any model or feeding resampled data to the robot policy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

def describe(path):
    content = Path(path).read_bytes()
    return dict(path=str(Path(path).resolve()), bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest())


def validated_times(values):
    result = np.asarray(values, dtype=np.float64)
    if (result.ndim != 1 or len(result) < 2 or not np.isfinite(result).all()
            or np.any(np.diff(result) <= 0)):
        raise ValueError("Require finite strictly increasing recording times")
    return result


def distribution_ms(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(zip(("p50_ms", "p95_ms", "p99_ms", "maximum_ms"),
                    (float(value * 1000) for value in
                     (*np.percentile(values, (50, 95, 99)), np.max(values)))))


def cadence(values):
    values = validated_times(values)
    dt = np.diff(values)
    return dict(frames=len(values), duration_s=float(values[-1] - values[0]),
        average_hz=float((len(values) - 1) / (values[-1] - values[0])),
        interval=distribution_ms(dt), gaps_over_100ms=int(np.count_nonzero(dt > .1)),
        gaps_over_200ms=int(np.count_nonzero(dt > .2)))


def interpolation_dependency(values, hz=50):
    values = validated_times(values)
    if not isinstance(hz, (int, float)) or not np.isfinite(hz) or hz <= 0:
        raise ValueError("Require a finite positive resampling frequency")
    relative = values - values[0]
    tolerance = max(1e-9, 4 * np.spacing(np.max(np.abs(values))))
    targets = np.arange(int(np.floor((relative[-1] + tolerance) * hz)) + 1, dtype=np.float64) / hz
    right = np.minimum(np.searchsorted(relative, targets, side="left"), len(relative) - 1)
    left = np.searchsorted(relative, targets, side="right") - 1
    lookahead, age = np.maximum(relative[right] - targets, 0.), targets - relative[left]
    return dict(output_hz=hz, output_frames=len(targets),
        timestamp_roundoff_tolerance_ns=float(tolerance * 1e9),
        future_support_fraction=float(np.mean(lookahead > tolerance)),
        offline_interpolation_future_dependency=distribution_ms(lookahead),
        causal_hold_data_age=distribution_ms(age),
        timing_note="Same-clock source-grid dependencies only; neither is measured end-to-end latency")


def audit_bundle(path):
    path = Path(path)
    before = describe(path)
    inspection_path = path.with_name("inspection.json")
    inspection_identity = describe(inspection_path)
    inspection = json.loads(inspection_path.read_text())
    if (inspection.get("usage") != "TEST_ONLY" or inspection.get("training_allowed") is not False
            or inspection.get("bundle", {}).get("sha256") != before["sha256"]):
        raise ValueError("Require the unchanged extracted TEST_ONLY bundle and its inspection manifest")
    with np.load(path, allow_pickle=False) as data:
        if data["usage"].item() != "TEST_ONLY" or data["training_allowed"].item() is not False:
            raise ValueError("Recorded test data cannot be relabelled as training data")
        host = validated_times(data["host_monotonic_s"])
        elapsed = validated_times(data["elapsed_s"])
        source = validated_times(data["source_time_s"])
        frames = data["frame_index"]
        if (len(host) != len(source) or len(host) != len(elapsed) or len(host) != len(frames)
                or frames.dtype.kind not in "iu" or np.any(np.diff(frames) <= 0)
                or data["joints"].shape != (len(host), 24, 3)
                or data["pose_matrix"].shape != (len(host), 24, 9)
                or data["trans"].shape != (len(host), 3)):
            raise ValueError("Recording frame arrays or body schema differ")
        root_translation_all_zero = bool(np.all(data["trans"] == 0))
        source_sha256 = str(data["source_sha256"].item())
        missing = int(np.maximum(np.diff(frames) - 1, 0).sum())
    row = dict(input=before, inspection=inspection_identity,
        raw_source_sha256=source_sha256, usage="TEST_ONLY", training_allowed=False,
        receiver_clock=cadence(host), source_payload_clock=cadence(source),
        source_payload_to_50hz=interpolation_dependency(source),
        receiver_record_to_50hz=interpolation_dependency(host),
        source_payload_capture_semantics="NOT_VERIFIED_AS_HARDWARE_CAPTURE_TIME",
        host_stamp_semantics="RECORDED_RECEIVER_HOST_TIME; internal perception/dependency time unknown",
        source_receiver_clocks_synchronized=False,
        receiver_duration_minus_payload_duration_s=float((host[-1] - host[0]) - (source[-1] - source[0])),
        duration_difference_is_clock_skew_measurement=False,
        source_frame_index_missing_count=missing,
        root_translation_all_zero=root_translation_all_zero,
        world_locomotion_following_assessable=not root_translation_all_zero,
        measured_end_to_end_latency_ms=None,
        preprocessing_causality="UNKNOWN",
        online_resampler_required=True,
        resampled_pose_is_an_exact_measured_capture=False,
        training_samples_used=0, model_updates=0, new_physics_steps=0)
    if describe(path) != before or describe(inspection_path) != inspection_identity:
        raise RuntimeError("Recorded source changed during the timing audit")
    row["input_integrity"] = "PASSED"
    return row


def run(paths, output):
    paths, output = [Path(path) for path in paths], Path(output)
    if output.exists() or not paths or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Require distinct input bundles and a new output file")
    started = time.perf_counter()
    rows = [audit_bundle(path) for path in paths]
    result = dict(status="COMPLETED", stage="RECORDED_TEST_ONLY_INTAKE_TIMING_AUDIT", recordings=rows,
        audit_code=describe(Path(__file__)),
        training_allowed=False, training_samples_used=0, optimizer_updates=0,
        new_physics_steps=0, cuda_initialized=False, measured_end_to_end_latency_ms=None,
        elapsed_seconds=time.perf_counter() - started,
        limitations=["No clock calibration or hardware capture timestamp proof",
            "No policy inference, physical replay, learning or parameter tuning",
            "Two recordings with no root translation cannot validate locomotion",
            "Offline interpolation lookahead is not total robot latency"],
        user_accepted_checkpoint=None)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=describe(output), recordings=len(rows),
                          elapsed_seconds=result["elapsed_seconds"])))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    run(options.input, options.output)
