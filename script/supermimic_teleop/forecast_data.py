"""Frozen whole-source splits and causal windows for an independent forecaster.

Only the pinned, already authorized 393-clip corpus is accepted by the CLI.
Planning reads metadata, not NPZ bodies. Sidecar audit is an explicit operation.
The 50 Hz sidecars have offline interpolation dependencies, not live timestamps.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np


ADMISSION_SHA = "896738c54a039160334312bf43158ea2d159eed250a588594ceed0a2e066c167"
SOURCES_SHA = "82f0615d72651958bf57e917fefa186e1f0384696569005a1702958781f27745"
HUMAN_ROOT = ("/opt/data/private/ZHY/data/supermimic_for_q1/"
    "MORPHMIMIC_Q1_SOMA_FULL1H_V01_20261102/converted/npz50/human50")
DT_NS = 20_000_000
HUMAN_NAMES = (
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee", "Spine2",
    "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot", "Neck",
    "Left_Collar", "Right_Collar", "Head", "Left_Shoulder", "Right_Shoulder",
    "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist", "Left_Hand", "Right_Hand",
)
SPLITS = ("train", "validation", "test")


def identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return dict(bytes=path.stat().st_size, sha256=digest.hexdigest())


def read_pinned(path, expected):
    before = identity(path)
    if before["sha256"] != expected:
        raise ValueError("Input differs from the pinned authorized manifest")
    value = json.loads(Path(path).read_text())
    if identity(path) != before:
        raise ValueError("Manifest changed during reading")
    return value, before


def relative_clip(value):
    if not isinstance(value, str):
        raise ValueError("Require a string relative clip path")
    path = PurePosixPath(value)
    if (path.is_absolute() or len(path.parts) != 2
            or any(p in (".", "..") for p in path.parts) or path.suffix != ".npz"
            or str(path) != value or "\\" in value):
        raise ValueError("Require an exact source-directory/clip.npz relative path")
    return path


def checked_identity(value):
    digest, size = value.get("sha256"), value.get("bytes")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or type(size) is not int or size <= 0):
        raise ValueError("Require a complete positive-size content identity")
    return dict(bytes=size, sha256=digest)


def source_support(frame):
    """Exact last nonzero support of the pinned 120 -> 50 Hz interpolator."""
    if type(frame) is not int or frame < 0:
        raise ValueError("Frame must be a nonnegative integer")
    source_index = (frame * 12 + 4) // 5
    dependency_ns = (source_index * 1_000_000_000 + 119) // 120
    return dict(nominal_ns=frame*DT_NS, source_index=source_index,
                dependency_end_ns=dependency_ns,
                right_support_ns=dependency_ns-frame*DT_NS)


def policy_slot_mapping(buffer_frames):
    if type(buffer_frames) is not int or not 0 <= buffer_frames <= 10:
        raise ValueError("Buffer must be an integer in 0..10")
    return [dict(policy_slot=slot, nominal_offset_ns=slot*DT_NS,
                 measured=slot <= buffer_frames,
                 forecast_horizon=None if slot <= buffer_frames else slot-buffer_frames)
            for slot in range(1, 11)]


def make_plan(admission, sources, *, seed=20261106, history_frames=25, future_frames=10):
    if (admission.get("status") != "FROZEN" or sources.get("status") != "COMPLETED_CPU_SOURCE_EXTRACTION"
            or admission.get("original_sources_edited") is not False):
        raise ValueError("Require the frozen unchanged paired corpus and completed source extraction")
    if (type(seed) is not int or type(history_frames) is not int or history_frames < 2
            or type(future_frames) is not int or future_frames != 10):
        raise ValueError("Require a fixed integer seed, at least two history frames and ten targets")
    extracted = {}
    for row in sources["extraction_result"]["outputs"]:
        name = str(relative_clip(row["clip"]))
        if name in extracted:
            raise ValueError("Duplicate source extraction row")
        if row["archive_member"] != "soma_uniform/bvh/" + str(PurePosixPath(name).with_suffix(".bvh")):
            raise ValueError("Original BVH member does not match its paired clip")
        extracted[name] = row
    rows = admission["rows"]
    if len(rows) != admission["admitted"] or len({r["relative"] for r in rows}) != len(rows):
        raise ValueError("Frozen admission row count or unique paths differ")
    fixed = set(admission["fixed_motion_names"])
    found, clips = set(), []
    parents = {}

    def find(value):
        parents.setdefault(value, value)
        while value != parents[value]:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    def union(a, b):
        a, b = find(a), find(b)
        parents[max(a, b)] = min(a, b)

    seen = {}
    for row in sorted(rows, key=lambda r: r["relative"]):
        path = relative_clip(row["relative"])
        if str(path) not in extracted or type(row["frames"]) is not int or row["frames"] < 1:
            raise ValueError("Missing original source or invalid frame count")
        source = extracted[str(path)]
        group = path.parts[0]
        find(group)
        human = checked_identity(row["human"])
        original = checked_identity(source)
        for key in (("human", human["sha256"]), ("bvh", original["sha256"])):
            if key in seen:
                union(group, seen[key])
            seen[key] = group
        is_fixed = path.stem in fixed
        if is_fixed:
            found.add(path.stem)
        clips.append(dict(relative=str(path), frames=row["frames"], source_directory=group,
            human=human, original_bvh=original, archive_member=source["archive_member"],
            fixed_evaluation_clip=is_fixed,
            windows=max(0, row["frames"]-history_frames-future_frames+1)))
    if found != fixed or len(fixed) != 3:
        raise ValueError("All three fixed evaluation sources must be retained")
    components = {}
    for directory in parents:
        components.setdefault(find(directory), []).append(directory)
    group_for = {d: tuple(sorted(v)) for v in components.values() for d in v}
    test_groups = {group_for[c["source_directory"]] for c in clips if c["fixed_evaluation_clip"]}
    for clip in clips:
        group = group_for[clip["source_directory"]]
        score = int(hashlib.sha256((str(seed)+"|"+"|".join(group)).encode()).hexdigest(), 16)
        bucket = score % 100
        clip["group_id"] = "|".join(group)
        clip["split"] = ("test" if group in test_groups or bucket >= 85 else
                         "validation" if bucket >= 70 else "train")
    summary = {}
    for split in SPLITS:
        selected = [c for c in clips if c["split"] == split]
        summary[split] = dict(clips=len(selected), groups=len({c["group_id"] for c in selected}),
            frames=sum(c["frames"] for c in selected), windows=sum(c["windows"] for c in selected),
            seconds=sum((c["frames"]-1)/50 for c in selected),
            human_bytes=sum(c["human"]["bytes"] for c in selected))
        if not selected or not summary[split]["windows"]:
            raise ValueError("Each whole-source split must have actual eligible windows")
    return dict(schema="q1_causal_forecast_split_v1", seed=seed, history_frames=history_frames,
        future_frames=future_frames, fps=50,
        grouping="whole original source directory, unioned by identical BVH or human content",
        subject_disjoint_verified=False, action_family_disjoint_verified=False,
        original_policy_trained_on_all_393=True,
        unseen_end_to_end_policy_generalization_claim_allowed=False,
        preprocessing=dict(type="offline_120_to_50_linear_slerp", source_fps=120,
            right_support_cycle_ns=[source_support(i)["right_support_ns"] for i in range(5)],
            upstream_mocap_and_capture_clock="UNKNOWN", live_capture_causal_claim_allowed=False),
        terminal_padding=False, test_only_recordings_included=False,
        test_used_for_normalization_or_model_selection=False,
        summary=summary, clips=clips)


def load_plan(admission_path, sources_path, plan_path=None):
    admission, admission_id = read_pinned(admission_path, ADMISSION_SHA)
    sources, sources_id = read_pinned(sources_path, SOURCES_SHA)
    stored = None if plan_path is None else json.loads(Path(plan_path).read_text())
    options = {} if stored is None else {k: stored[k] for k in ("seed", "history_frames", "future_frames")}
    plan = make_plan(admission, sources, **options)
    plan["manifest_identities"] = dict(admission=admission_id, original_sources=sources_id)
    if stored is not None and stored != plan:
        raise ValueError("Stored split differs from its pinned source-derived plan")
    return plan


@dataclass(frozen=True)
class ForecastWindow:
    relative: str
    split: str
    cutoff: int
    history_indices: tuple
    target_indices: tuple
    input_positions: np.ndarray
    input_quaternions: np.ndarray
    target_positions: np.ndarray
    target_quaternions: np.ndarray
    latest_input_dependency_ns: int


class ForecastCorpus:
    """One sidecar cached at a time; no cross-clip or synthetic terminal windows."""

    def __init__(self, plan, human_root):
        self.plan = plan
        self.root = Path(human_root).resolve(strict=True)
        self.rows = {c["relative"]: c for c in plan["clips"]}
        self.cached = None

    def load(self, relative):
        row = self.rows[str(relative_clip(relative))]
        if self.cached is not None and self.cached[0] == relative:
            return self.cached[1:]
        path = (self.root / relative).resolve(strict=True)
        if not path.is_relative_to(self.root):
            raise ValueError("Sidecar escaped the explicitly selected human root")
        before = identity(path)
        if before != row["human"]:
            raise ValueError("Sidecar content differs from the frozen corpus")
        with np.load(path, allow_pickle=False) as data:
            if "training_allowed" in data and not bool(data["training_allowed"].item()):
                raise ValueError("TEST_ONLY recordings cannot enter training")
            if (int(data["fps"]) != 50 or tuple(str(n) for n in data["human_joint_names"]) != HUMAN_NAMES
                    or data["source_frames"].shape != () or int(data["source_frames"]) < 2):
                raise ValueError("Sidecar fps, original frame count or SMPL24 order differs")
            source_frames = int(data["source_frames"])
            positions = data["human_pos_w"].astype(np.float64)
            quaternions = data["human_quat_w"].astype(np.float64)
        frames = row["frames"]
        if ((source_frames-1)*5//12+1 != frames or positions.shape != (frames, 24, 3)
                or quaternions.shape != (frames, 24, 4)
                or not np.isfinite(positions).all() or not np.isfinite(quaternions).all()
                or np.max(np.abs(np.sum(quaternions**2, axis=-1)-1)) > 1e-4):
            raise ValueError("Sidecar shape, interpolation count or finite unit poses differ")
        if identity(path) != before:
            raise ValueError("Sidecar changed while loading")
        positions.setflags(write=False)
        quaternions.setflags(write=False)
        self.cached = (relative, positions, quaternions)
        return positions, quaternions

    def cutoffs(self, relative):
        return range(self.plan["history_frames"]-1,
                     self.rows[relative]["frames"]-self.plan["future_frames"])

    def window(self, relative, cutoff, *, split):
        if split not in SPLITS or self.rows[relative]["split"] != split:
            raise ValueError("Requested split does not own the whole source")
        if type(cutoff) is not int or cutoff not in self.cutoffs(relative):
            raise ValueError("History and all ten targets must be inside this single clip")
        positions, quaternions = self.load(relative)
        history = tuple(range(cutoff-self.plan["history_frames"]+1, cutoff+1))
        targets = tuple(range(cutoff+1, cutoff+self.plan["future_frames"]+1))
        arrays = [values[np.asarray(indices)].copy() for values, indices in
                  ((positions, history), (quaternions, history),
                   (positions, targets), (quaternions, targets))]
        for values in arrays:
            values.setflags(write=False)
        return ForecastWindow(relative, split, cutoff, history, targets, *arrays,
                              source_support(cutoff)["dependency_end_ns"])


class TrainingMoments:
    """Online population moments; fitting accepts training examples only."""

    def __init__(self):
        self.count, self.mean, self.m2 = 0, None, None

    def update(self, values, *, split):
        values = np.asarray(values, dtype=np.float64)
        if split != "train":
            raise ValueError("Normalization fitting is restricted to training sources")
        if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Require finite nonempty [samples,features] training values")
        if self.mean is not None and values.shape[1:] != self.mean.shape:
            raise ValueError("Training feature dimensions changed")
        count, mean = len(values), values.mean(axis=0)
        m2 = np.sum((values-mean)**2, axis=0)
        if not self.count:
            self.count, self.mean, self.m2 = count, mean, m2
        else:
            delta = mean-self.mean
            total = self.count+count
            self.m2 += m2 + delta**2*self.count*count/total
            self.mean += delta*count/total
            self.count = total

    def parameters(self):
        if not self.count:
            raise ValueError("No training normalization samples")
        return dict(count=self.count, mean=self.mean.tolist(),
                    scale=np.maximum(np.sqrt(self.m2/self.count), 1e-6).tolist(),
                    fitted_split="train")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "audit-sidecars"))
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--human-root", type=Path, default=Path(HUMAN_ROOT))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; preserve previous evidence")
    if args.command == "audit-sidecars" and args.plan is None:
        parser.error("Audit requires a published frozen --plan")
    plan = load_plan(args.admission, args.sources, args.plan)
    result = plan
    if args.command == "audit-sidecars":
        corpus = ForecastCorpus(plan, args.human_root)
        for row in plan["clips"]:
            corpus.load(row["relative"])
        result = dict(status="VERIFIED_FROZEN_SIDECARS", clips=len(plan["clips"]),
            human_bytes=sum(c["human"]["bytes"] for c in plan["clips"]),
            plan_identity=identity(args.plan), summary=plan["summary"],
            policy_inference=0, physics_steps=0, optimizer_updates=0,
            upstream_capture_causal_claim_allowed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(output=str(args.output), identity=identity(args.output),
                          summary=plan["summary"]), allow_nan=False))


if __name__ == "__main__":
    main()
