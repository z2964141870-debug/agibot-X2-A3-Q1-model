"""Scalar offline motion scoring; no model, FK, tensor library or sensor access.

All distances use meters. Nearest-time pairing may select a future sample and
is explicitly NOT an online fusion algorithm. Metrics never fill missing poses.
The caller must establish dataset rights, provenance and experiment admission.
"""
from bisect import bisect_left
from dataclasses import dataclass
from itertools import combinations
import math


MATCH_LIMIT_NS = 50_000_000
LOCAL_EQUALITY_TOLERANCE_M = 1e-6
KNOWN_FRAMES = {"synthetic_world_z_up", "optitrack_world_z_up"}
KNOWN_CLOCKS = {"synthetic_capture", "utc"}


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


@dataclass(frozen=True)
class PoseSeries:
    timestamps_ns: tuple
    joints_m: tuple
    joint_names: tuple
    root_index: int
    coordinate_frame: str
    clock_domain: str

    @classmethod
    def from_mapping(cls, data):
        required = {"timestamps_ns", "joints_m", "joint_names", "root_index",
                    "coordinate_frame", "clock_domain", "position_unit", "timestamp_unit"}
        if not isinstance(data, dict) or not required.issubset(data):
            raise ValueError("explicit pose, unit, frame, clock and root metadata required")
        if data["position_unit"] != "m" or data["timestamp_unit"] != "ns":
            raise ValueError("expected meters and integer nanoseconds; no implicit conversion")
        if data["coordinate_frame"] not in KNOWN_FRAMES:
            raise ValueError("unknown world coordinate frame")
        if data["clock_domain"] not in KNOWN_CLOCKS:
            raise ValueError("unknown clock domain")
        if ((data["coordinate_frame"] == "synthetic_world_z_up")
                != (data["clock_domain"] == "synthetic_capture")):
            raise ValueError("synthetic and physical frame/clock declarations cannot be mixed")
        names = data["joint_names"]
        if (not isinstance(names, (list, tuple)) or not 1 <= len(names) <= 22
                or not all(isinstance(n, str) and n for n in names)
                or len(set(names)) != len(names)):
            raise ValueError("expected 1..22 explicitly ordered unique joint names")
        root = data["root_index"]
        if not _integer(root) or not 0 <= root < len(names) or names[root] != "pelvis":
            raise ValueError("root index must identify the declared pelvis joint")
        times, poses = data["timestamps_ns"], data["joints_m"]
        if not isinstance(times, (list, tuple)) or not isinstance(poses, (list, tuple)):
            raise ValueError("timestamps and poses must be sequences")
        if len(times) != len(poses):
            raise ValueError("pose/time lengths differ; truncation is prohibited")
        if (not all(_integer(t) and t >= 0 for t in times)
                or any(b <= a for a, b in zip(times, times[1:]))):
            raise ValueError("timestamps must be nonnegative, strictly increasing integer ns")
        frozen = []
        for frame in poses:
            if not isinstance(frame, (list, tuple)) or len(frame) != len(names):
                raise ValueError("joint count differs from the declared order")
            points = []
            for point in frame:
                if (not isinstance(point, (list, tuple)) or len(point) != 3
                        or not all(_finite(x) for x in point)):
                    raise ValueError("joint positions must contain three finite scalars")
                points.append(tuple(float(x) for x in point))
            frozen.append(tuple(points))
        return cls(tuple(times), tuple(frozen), tuple(names), root,
                   data["coordinate_frame"], data["clock_domain"])

    def contract(self):
        return (self.joint_names, self.root_index, self.coordinate_frame, self.clock_domain)


def _matches(gt_times, predicted_times):
    """Return nearest indices within the fixed gate; ties choose earlier time."""
    result, offsets = [], []
    for t in gt_times:
        if not predicted_times:
            result.append(None)
            offsets.append(None)
            continue
        right = bisect_left(predicted_times, t)
        candidates = [i for i in (right - 1, right) if 0 <= i < len(predicted_times)]
        i = min(candidates, key=lambda k: (abs(predicted_times[k] - t), predicted_times[k]))
        dt = predicted_times[i] - t
        offsets.append(dt)
        result.append(i if abs(dt) <= MATCH_LIMIT_NS else None)
    return tuple(result), tuple(offsets)


def _norm2(v):
    return math.fsum(x * x for x in v)


def _score(gt, prediction, match, gt_indices):
    if not gt_indices:
        return None
    root_sq, world_sq, local_sq, cross, world_lengths, local_lengths = [], [], [], [], [], []
    for i in gt_indices:
        j = match[i]
        if j is None:
            raise ValueError("cannot score a missing observation")
        truth, pred = gt.joints_m[i], prediction.joints_m[j]
        r = tuple(pred[gt.root_index][k] - truth[gt.root_index][k] for k in range(3))
        root_sq.append(_norm2(r))
        for p, q in zip(pred, truth):
            error = tuple(p[k] - q[k] for k in range(3))
            local = tuple(error[k] - r[k] for k in range(3))
            e2, l2 = _norm2(error), _norm2(local)
            world_sq.append(e2)
            local_sq.append(l2)
            cross.append(2.0 * math.fsum(r[k] * local[k] for k in range(3)))
            world_lengths.append(math.sqrt(e2))
            local_lengths.append(math.sqrt(l2))
    n = len(world_sq)
    root_mse = math.fsum(root_sq) / len(root_sq)
    world_mse, local_mse, cross_term = (math.fsum(v) / n for v in (world_sq, local_sq, cross))
    if not all(math.isfinite(v) for v in (root_mse, world_mse, local_mse, cross_term)):
        raise ValueError("non-finite derived metric; input magnitude exceeds scalar arithmetic")
    return {
        "gt_frames_scored": len(gt_indices),
        "joints_per_frame_including_pelvis": len(gt.joint_names),
        "world_mpjpe_m": math.fsum(world_lengths) / n,
        "world_joint_mse_m2": world_mse,
        "root_position_rmse_m": math.sqrt(root_mse),
        "root_position_mse_m2": root_mse,
        "root_relative_mpjpe_m": math.fsum(local_lengths) / n,
        "root_relative_joint_mse_m2": local_mse,
        "root_local_cross_m2": cross_term,
        "decomposition_residual_m2": world_mse - (root_mse + local_mse + cross_term),
        "gt_root_replacement_counterfactual_mpjpe_m": math.fsum(local_lengths) / n,
        "gt_root_replacement_is_deployable": False,
        "gt_root_replacement_is_guaranteed_error_bound": False,
    }


def _local_equivalence(a, b, matches_a, matches_b, common_indices):
    if not common_indices:
        return {"status": "NO_COMMON_FRAMES", "root_only_attribution_supported": False}
    max_error = 0.0
    same_times = True
    for i in common_indices:
        ia, ib = matches_a[i], matches_b[i]
        same_times &= a.timestamps_ns[ia] == b.timestamps_ns[ib]
        pa, pb = a.joints_m[ia], b.joints_m[ib]
        for ja, jb in zip(pa, pb):
            delta = tuple((ja[k] - pa[a.root_index][k]) - (jb[k] - pb[b.root_index][k])
                          for k in range(3))
            max_error = max(max_error, math.sqrt(_norm2(delta)))
    equal = max_error <= LOCAL_EQUALITY_TOLERANCE_M
    return {"status": "CHECKED_ON_COMMON_GT_TIMELINE", "gt_frames": len(common_indices),
            "same_source_capture_times": same_times,
            "max_root_relative_joint_distance_m": max_error,
            "local_equality_tolerance_m": LOCAL_EQUALITY_TOLERANCE_M,
            "sampled_local_poses_equal": equal,
            "root_only_attribution_supported": equal and same_times,
            "causal_source_provenance_verified": False}


def evaluate(ground_truth, methods):
    """Score admitted mappings on a fixed GT axis, exposing missing coverage.

    This function checks declared contracts, not whether source calibrations or
    source labels are actually accurate. It never claims data/model admission.
    """
    gt = PoseSeries.from_mapping(ground_truth)
    if not gt.timestamps_ns:
        raise ValueError("empty ground-truth timeline")
    if (not isinstance(methods, dict) or not 1 <= len(methods) <= 8
            or not all(isinstance(name, str) and name for name in methods)):
        raise ValueError("expected one to eight named prediction methods")
    series = {name: PoseSeries.from_mapping(value) for name, value in methods.items()}
    if any(s.contract() != gt.contract() for s in series.values()):
        raise ValueError("joint order/root, world frame or clock contract mismatch")
    matched = {name: _matches(gt.timestamps_ns, s.timestamps_ns) for name, s in series.items()}
    valid = {name: {i for i, j in enumerate(m[0]) if j is not None}
             for name, m in matched.items()}
    common = sorted(set.intersection(*valid.values()))
    results = {}
    for name, s in series.items():
        mapping, offsets = matched[name]
        indices = sorted(valid[name])
        used = [mapping[i] for i in indices]
        results[name] = {
            "source_frames": len(s.timestamps_ns),
            "gt_frames_matched": len(indices),
            "gt_frames_missing": len(gt.timestamps_ns) - len(indices),
            "gt_frame_coverage": len(indices) / len(gt.timestamps_ns),
            "matched_gt_indices": indices,
            "missing_gt_indices": [i for i in range(len(gt.timestamps_ns)) if i not in valid[name]],
            "matched_source_indices": used,
            "unique_source_frames_used": len(set(used)),
            "reused_source_assignments": len(used) - len(set(used)),
            "source_minus_gt_ns_for_matched_frames": [offsets[i] for i in indices],
            "max_abs_match_error_ns": max((abs(offsets[i]) for i in indices), default=None),
            "all_available_metrics": _score(gt, s, mapping, indices),
            "common_gt_metrics": _score(gt, s, mapping, common),
        }
    equivalence = []
    for a, b in combinations(series, 2):
        pair_common = sorted(valid[a] & valid[b])
        equivalence.append({"methods": [a, b], **_local_equivalence(
            series[a], series[b], matched[a][0], matched[b][0], pair_common)})
    return {
        "schema_version": 1,
        "matching_mode": "OFFLINE_NEAREST_NOT_CAUSAL",
        "max_match_error_ns": MATCH_LIMIT_NS,
        "coverage_denominator": "ground_truth_frame_count_not_wall_time",
        "gt_frames": len(gt.timestamps_ns),
        "joint_names": list(gt.joint_names),
        "root_index": gt.root_index,
        "coordinate_frame": gt.coordinate_frame,
        "clock_domain": gt.clock_domain,
        "common_gt_frames": len(common),
        "common_gt_indices": common,
        "common_gt_frame_coverage": len(common) / len(gt.timestamps_ns),
        "methods": results,
        "pairwise_local_equivalence": equivalence,
        "coverage_warning": "Conditional errors do not penalize missing frames; report coverage alongside them.",
        "data_provenance_and_rights_verified": False,
        "robot_or_headband_performance_verified": False,
    }
