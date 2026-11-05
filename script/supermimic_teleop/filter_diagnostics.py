"""Fixed scalar analytic filter checks, not human-data tuning or robot control.

One Euro comes from the unmodified author-maintained BSD wheel. These radian
signals expose phase/event tradeoffs; they do not implement a quaternion filter.
"""

import argparse
from collections import deque
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

from recorded_intake_audit import describe
from stream_contract import nonnegative_ns


WHEEL_SHA256 = "a81db3155ea51c4f920762de04e888909d17e2af87d9e544947d20740f70fae8"
METHODS = ("raw", "mean5", "one_euro_beta0", "one_euro_beta01", "slew2", "one_euro_beta01_slew2")


def load_one_euro(wheel):
    wheel = Path(wheel).resolve()
    identity = describe(wheel)
    if identity["sha256"] != WHEEL_SHA256:
        raise ValueError("Require the exact pinned author OneEuroFilter0.2.1 wheel")
    with zipfile.ZipFile(wheel) as archive:
        members = {}
        for name in ("OneEuroFilter/OneEuroFilter.py", "OneEuroFilter-0.2.1.dist-info/LICENSE",
                     "OneEuroFilter-0.2.1.dist-info/METADATA"):
            content = archive.read(name)
            members[name] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    sys.path.insert(0, str(wheel))
    module = importlib.import_module("OneEuroFilter")
    if not str(module.__file__).startswith(str(wheel)+"/"):
        raise RuntimeError("A different OneEuroFilter module was already imported")
    return module.OneEuroFilter, dict(wheel=identity, members=members,
        version="0.2.1", license="BSD-3-Clause; copyright retained in original wheel",
        author="Gery Casiez / Nicolas Roussel / Inria",
        upstream="https://github.com/casiez/OneEuroFilter",
        package="https://pypi.org/project/OneEuroFilter/0.2.1/",
        unmodified=True, installed_into_system_environment=False,
        implementation_detail="0.2.1 estimates derivative relative to last filtered value")


class ScalarFilter:
    def __init__(self, method, one_euro_type=None, *, max_gap_ns=200_000_000):
        if method not in METHODS:
            raise ValueError("Unknown fixed filter method")
        nonnegative_ns(max_gap_ns, "Maximum filter gap")
        if max_gap_ns == 0:
            raise ValueError("Filter gap budget must be positive")
        self.method, self.max_gap = method, max_gap_ns
        self.last_ns, self.last_output = None, None
        self.values = deque(maxlen=5)
        self.euro = None
        if method.startswith("one_euro"):
            if one_euro_type is None:
                raise ValueError("The pinned OneEuroFilter dependency is required")
            beta = 0. if method == "one_euro_beta0" else .1
            self.euro = one_euro_type(freq=50., mincutoff=1., beta=beta, dcutoff=1.)

    def __call__(self, value, timestamp_ns):
        nonnegative_ns(timestamp_ns, "Filter timestamp")
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("Filter input must be finite scalar radians")
        if self.last_ns is not None:
            if timestamp_ns <= self.last_ns:
                raise ValueError("Filter timestamps must be strictly increasing")
            if timestamp_ns-self.last_ns > self.max_gap:
                raise ValueError("Stale segment; caller must create a fresh filter after fallback")
        seconds = .02 if self.last_ns is None else (timestamp_ns-self.last_ns)/1e9
        output = float(value)
        if self.method == "mean5":
            self.values.append(output)
            output = sum(self.values)/len(self.values)
        if self.euro is not None:
            # Explicit frequency also handles the upstream truthiness check at t=0.
            self.euro.setFrequency(1./seconds)
            output = self.euro(output, timestamp_ns/1e9)
        if self.method.endswith("slew2") and self.last_output is not None:
            maximum = 2.*seconds
            output = max(self.last_output-maximum, min(self.last_output+maximum, output))
        if not math.isfinite(output):
            raise RuntimeError("Nonfinite scalar filter output")
        self.last_ns, self.last_output = timestamp_ns, output
        return output


def filter_signal(method, times, values, one_euro_type=None):
    times, values = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    if (times.ndim != 1 or values.shape != times.shape or len(times) < 2
            or not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times) <= 0)
            or not np.isfinite(values).all()):
        raise ValueError("Require finite increasing times and matching scalar inputs")
    filter_ = ScalarFilter(method, one_euro_type)
    return np.asarray([filter_(float(value), int(round(float(stamp)*1e9)))
                       for stamp, value in zip(times, values)])


def first_time(times, condition, after):
    indices = np.flatnonzero((times >= after)&condition)
    return None if not len(indices) else float(times[indices[0]]-after)


def harmonic_response(times, values, frequency, *, after=5., input_amplitude=.25):
    selected = times >= after
    phase = 2*np.pi*frequency*times[selected]
    basis = np.column_stack((np.sin(phase), np.cos(phase), np.ones(len(phase))))
    coefficients = np.linalg.lstsq(basis, values[selected], rcond=None)[0]
    if not np.isfinite(coefficients).all():
        raise RuntimeError("Nonfinite harmonic fit coefficients")
    prediction = np.sum(basis*coefficients, axis=1)
    phase_radians = float(np.arctan2(coefficients[1], coefficients[0]))
    return dict(frequency_hz=frequency,
        fundamental_gain=float(np.hypot(*coefficients[:2])/input_amplitude),
        fundamental_phase_lag_ms=-phase_radians/(2*np.pi*frequency)*1000,
        harmonic_residual_rms_rad=float(np.sqrt(np.mean((prediction-values[selected])**2))),
        phase_is_measured_end_to_end_delay=False)


def scenarios():
    time_short = np.arange(301, dtype=float)/50
    step = np.where(time_short >= 2., .5, 0.)
    stop = np.clip(time_short-2., 0., 1.)
    reversal = np.where(time_short < 3., np.clip(time_short-2., 0., 1.),
                        np.clip(4.-time_short, 0., 1.))
    spike = np.zeros_like(time_short)
    spike[100] = .5
    rng = np.random.default_rng(42)
    time_long = np.arange(1001, dtype=float)/50
    cases = [("step_05rad", time_short, step, None),
             ("ramp_stop_1rad_s", time_short, stop, None),
             ("ramp_reversal_1rad_s", time_short, reversal, None),
             ("one_sample_spike_05rad", time_short, spike, None),
             ("stationary_noise_std001rad", time_long, rng.normal(0., .01, len(time_long)), None)]
    cases += [("sine_"+str(frequency)+"hz", time_long,
               .25*np.sin(2*np.pi*frequency*time_long), frequency) for frequency in (.5, 1., 2., 5.)]
    return cases


def verify_library(one_euro_type):
    times = np.arange(6, dtype=float)*.02
    values = np.asarray([0., 1., 1., 1., 1., 1.])
    actual = filter_signal("one_euro_beta0", times, values, one_euro_type)
    alpha = 1./(1.+1./(2*np.pi*.02))
    expected = 1.-(1.-alpha)**np.arange(6)
    error = float(np.max(np.abs(actual-expected)))
    if error > 1e-12:
        raise RuntimeError("Pinned library fails the analytic beta0 low-pass check")
    return dict(beta0_analytic_max_error=error, tolerance=1e-12, status="PASSED")


def run(wheel, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Require a new analytic filter output")
    one_euro_type, dependency = load_one_euro(wheel)
    library_check = verify_library(one_euro_type)
    started = time.perf_counter()
    results = []
    for name, times, values, frequency in scenarios():
        for method in METHODS:
            filtered = filter_signal(method, times, values, one_euro_type)
            speed = np.diff(filtered)/np.diff(times)
            row = dict(scenario=name, method=method, samples=len(times),
                output_peak_abs_rad=float(np.max(np.abs(filtered))),
                maximum_output_step_rad=float(np.max(np.abs(np.diff(filtered)))),
                maximum_output_speed_rad_s=float(np.max(np.abs(speed))))
            if name == "step_05rad":
                row.update(t50_after_event_s=first_time(times, filtered >= .25, 2.),
                    t90_after_event_s=first_time(times, filtered >= .45, 2.),
                    overshoot_rad=float(max(0., np.max(filtered)-.5)))
            elif name == "ramp_stop_1rad_s":
                selected = times >= 3.
                row.update(stop_remaining_motion_rad=float(np.abs(np.diff(filtered[selected])).sum()),
                    stop_position_error_at_event_rad=float(1.-filtered[np.flatnonzero(selected)[0]]),
                    stop_error_after200ms_rad=float(1.-filtered[160]))
            elif name == "ramp_reversal_1rad_s":
                derivative = np.r_[0., speed]
                reaction = first_time(times, derivative < -1e-9, 3.)
                row.update(first_negative_motion_after_reversal_s=reaction,
                    additional_reversal_delay_vs_raw_ms=None if reaction is None else (reaction-.02)*1000,
                    peak_position_rad=float(np.max(filtered)))
            elif name == "stationary_noise_std001rad":
                selected = times >= 5.
                row.update(output_noise_std_rad=float(np.std(filtered[selected])),
                           input_noise_std_rad=float(np.std(values[selected])))
            if frequency is not None:
                row["harmonic"] = harmonic_response(times, filtered, frequency)
            results.append(row)
    result = dict(stage="FIXED_ANALYTIC_SCALAR_FILTER_TRADEOFF", status="COMPLETED",
        code=describe(Path(__file__)), dependency=dependency, library_check=library_check,
        fixed_parameters=dict(input_hz=50, units="radians", mean_window=5,
            one_euro_min_cutoff_hz=1., one_euro_beta=[0., .1], derivative_cutoff_hz=1.,
            slew_limit_rad_s=2., stale_gap_ms=200, noise_std_rad=.01, noise_seed=42),
        cases=results, source_human_recordings_used=0, training_samples_used=0,
        model_updates=0, new_policy_inference=0, new_physics_steps=0, new_cuda_jobs=0,
        measured_end_to_end_latency_ms=None, elapsed_seconds=time.perf_counter()-started,
        limitations=["Scalar analytic signals only; no quaternion or contact constraints",
            "Fundamental phase lag is frequency-specific and not full robot latency",
            "OneEuro adaptive filtering is nonlinear; harmonic residual is reported",
            "Fixed settings are not tuned on human test data and not a deployment recommendation",
            "Slew rate cap does not bound acceleration or guarantee physical torque limits"],
        user_accepted_checkpoint=None)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=describe(output), cases=len(results),
                          elapsed_seconds=result["elapsed_seconds"])))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--one-euro-wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    run(options.one_euro_wheel, options.output)
