"""Causal position/SO(3) baselines with explicit measured-versus-forecast slots.

These independent point forecasts are not bone/contact constrained. They need
data, policy and physics validation before any claim about natural teleoperation.
"""

from dataclasses import dataclass
import math

from preview_policy import validate_window
from stream_contract import DT_NS, HumanPose, PreviewWindow, SLOTS

METHODS = ("hold", "cv", "damped_cv")
TAIL_METHODS = ("masked",) + METHODS


def norm(vector):
    return math.sqrt(sum(value*value for value in vector))


def scale(vector, factor):
    return tuple(value*factor for value in vector)


def cap(vector, maximum):
    length = norm(vector)
    return scale(vector, maximum/length) if maximum is not None and length > maximum else vector


def unit(quaternion):
    length = norm(quaternion)
    if length < 1e-12:
        raise ValueError("Zero quaternion")
    return scale(quaternion, 1/length)


def multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return (w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v)


def rotation_vector(q):
    q = unit(q)
    if q[0] < 0:
        q = scale(q, -1)
    length = norm(q[1:])
    return scale(q[1:], 2 if length < 1e-10 else 2*math.atan2(length, q[0])/length)


def rotation_quaternion(vector):
    angle = norm(vector)
    if angle < 1e-10:
        return unit((1., *(value/2 for value in vector)))
    return (math.cos(angle/2), *scale(vector, math.sin(angle/2)/angle))


def anchored_slope(values):
    """Fit velocity with the last measurement as a fixed intercept, at 50 Hz."""
    dt = DT_NS/1e9
    times = [(i-len(values)+1)*dt for i in range(len(values))]
    denominator = sum(t*t for t in times)
    return tuple(sum(t*(row[j]-values[-1][j]) for t, row in zip(times, values))/denominator
                 for j in range(len(values[-1])))


@dataclass(frozen=True)
class TailForecast:
    observed: PreviewWindow
    poses: tuple
    measured_mask: tuple
    generated_mask: tuple
    provenance: tuple
    method: str
    tau_s: float
    max_linear_speed_m_s: object
    max_angular_speed_rad_s: object

    def timing(self):
        return dict(self.observed.timing(), policy_prefix_frames=SLOTS,
            measured_preview_budget_ns=self.observed.requested_frames*DT_NS,
            generated_slots=sum(self.generated_mask), method=self.method,
            tau_s=self.tau_s, max_linear_speed_m_s=self.max_linear_speed_m_s,
            max_angular_speed_rad_s=self.max_angular_speed_rad_s,
            target_offsets_ns=[(i+1)*DT_NS for i in range(SLOTS)],
            upstream_causality=self.observed.source.preprocessing,
            bone_contact_constraints_applied=False)


def forecast_window(window, method="damped_cv", *, tau_s=.12,
                    max_linear_speed_m_s=None, max_angular_speed_rad_s=None):
    count = validate_window(window)
    if method not in METHODS or not math.isfinite(tau_s) or tau_s <= 0:
        raise ValueError("Require a known method and finite positive damping tau")
    for maximum in (max_linear_speed_m_s, max_angular_speed_rad_s):
        if maximum is not None and (not math.isfinite(maximum) or maximum <= 0):
            raise ValueError("Speed caps must be finite and positive")
    terminal = window.terminal_capture_ns is not None and \
        window.latest_used_capture_ns == window.terminal_capture_ns
    measured = list(window.poses[:count])
    if count < SLOTS and method != "hold" and not terminal and count < 2:
        raise ValueError("Velocity forecast requires at least two measured target poses")
    last = measured[-1]
    linear, angular = [], []
    if count < SLOTS and method != "hold" and not terminal:
        for point in range(24):
            linear.append(cap(anchored_slope([pose.positions[point] for pose in measured]),
                              max_linear_speed_m_s))
            q = last.orientations[point]
            inverse = (q[0], -q[1], -q[2], -q[3])
            relative = [rotation_vector(multiply(inverse, pose.orientations[point])) for pose in measured]
            angular.append(cap(anchored_slope(relative), max_angular_speed_rad_s))
    poses, provenance = list(window.poses[:count]), list(window.provenance[:count])
    for offset in range(1, SLOTS-count+1):
        seconds = offset*DT_NS/1e9
        travel = tau_s*(-math.expm1(-seconds/tau_s)) if method == "damped_cv" else seconds
        if method == "hold" or terminal:
            predicted = last
        else:
            positions = [tuple(value + velocity*travel for value, velocity in
                               zip(last.positions[point], linear[point])) for point in range(24)]
            orientations = [unit(multiply(last.orientations[point],
                rotation_quaternion(scale(angular[point], travel)))) for point in range(24)]
            predicted = HumanPose(positions, orientations)
        poses.append(predicted)
        provenance.append("declared_clip_terminal_hold" if terminal else "causal_forecast:"+method)
    generated = tuple(i >= count and not terminal for i in range(SLOTS))
    return TailForecast(window, tuple(poses), window.measured_mask, generated, tuple(provenance),
        "terminal_hold" if terminal else method, tau_s, max_linear_speed_m_s, max_angular_speed_rad_s)


def validate_forecast(forecast):
    if not isinstance(forecast, TailForecast):
        raise ValueError("Require an explicit measured/forecast goal")
    count = validate_window(forecast.observed)
    if type(forecast.tau_s) not in (int, float) or not math.isfinite(forecast.tau_s) or forecast.tau_s <= 0:
        raise ValueError("Invalid forecast damping tau")
    for value in (forecast.max_linear_speed_m_s, forecast.max_angular_speed_rad_s):
        if value is not None and (not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value <= 0):
            raise ValueError("Invalid forecast speed cap")
    if len(forecast.poses) != SLOTS or not all(isinstance(pose, HumanPose) for pose in forecast.poses):
        raise ValueError("Forecast requires ten valid human poses")
    if forecast.measured_mask != forecast.observed.measured_mask or forecast.poses[:count] != forecast.observed.poses[:count]:
        raise ValueError("Forecast changed the measured/terminal prefix")
    terminal = forecast.method == "terminal_hold"
    if forecast.method not in (*METHODS, "terminal_hold") or forecast.generated_mask != tuple(
            i >= count and not terminal for i in range(SLOTS)):
        raise ValueError("Invalid generated-slot identity")
    expected_tail = ("declared_clip_terminal_hold" if terminal else "causal_forecast:"+forecast.method,)*(SLOTS-count)
    if forecast.provenance != forecast.observed.provenance[:count] + expected_tail:
        raise ValueError("Forecast provenance differs from its measured prefix and method")
    declared_end = forecast.observed.terminal_capture_ns is not None and \
        forecast.observed.latest_used_capture_ns == forecast.observed.terminal_capture_ns
    if terminal != declared_end:
        raise ValueError("Cannot invent or ignore an end-of-clip event")
    if (terminal or forecast.method == "hold") and any(
            pose != forecast.observed.poses[count-1] for pose in forecast.poses[count:]):
        raise ValueError("A held tail must repeat the last available pose")
    return forecast
