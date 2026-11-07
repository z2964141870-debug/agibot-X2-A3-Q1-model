"""Select source-owned collection clips and retain actual common-plant traces."""

import hashlib
from pathlib import Path
import time

import numpy as np


MAX_STEPS = 256
STRIDE = 2
SPLIT_SHA = "f40c1d5a856a08d89f1a8c23dfe6d990fc7439232b9da9a658d9860456353145"


def collection_jobs(plan):
    if (plan.get("original_policy_trained_on_all_393") is not True
            or plan.get("test_used_for_normalization_or_model_selection") is not False):
        raise ValueError("Missing original-teacher and split-selection provenance")
    jobs = []
    ownership, relatives = {}, set()
    for row in plan["clips"]:
        if row["relative"] in relatives or row["split"] not in ("train", "validation", "test"):
            raise ValueError("Duplicated source or unknown split")
        relatives.add(row["relative"])
        previous = ownership.setdefault(row["group_id"], row["split"])
        if previous != row["split"]:
            raise ValueError("One original source group crosses dataset splits")
    for split, count, seed in (("train", 12, 2026110701), ("validation", 6, 2026110702)):
        rows = [row for row in plan["clips"] if row["split"] == split and row["frames"] >= 40]
        rows.sort(key=lambda row: hashlib.sha256((str(seed)+":"+row["relative"]).encode()).hexdigest())
        selected, groups = [], set()
        for row in rows:
            if row["group_id"] not in groups:
                selected.append(row)
                groups.add(row["group_id"])
                if len(selected) == count:
                    break
        if len(selected) != count:
            raise ValueError("Insufficient distinct source groups for bounded collection")
        for index, row in enumerate(selected):
            frames = min(MAX_STEPS, row["frames"]-10)
            first = max(1, min(frames-3, int(frames*.4)))
            duration = max(2, min(8, frames-first-2))
            jobs.append(dict(relative=row["relative"], clip=Path(row["relative"]).stem,
                split=split, group_id=row["group_id"], human=row["human"],
                maximum_steps=frames, record_stride=STRIDE,
                packet_event=dict(kind=("nominal", "dropout", "late_burst")[index % 3],
                    first_frame=first, duration_frames=duration, transport_frames=index % 3),
                physical_reference_event="nominal", seed=0))
    if len({job["clip"] for job in jobs}) != len(jobs):
        raise ValueError("Ambiguous clip names across collection sources")
    return jobs


def paired_source(item, job):
    human = item["human"]
    return (Path(human["path"]).as_posix().endswith("/"+job["relative"])
            and all(human[key] == job["human"][key] for key in ("bytes", "sha256")))


def validate_records(data, split, jobs, traces):
    shapes = dict(full_goal=(2254,), history=(930,), f0=(10, 226),
                  teacher_action=(29,), actual_state=(79,), prefix=(5,))
    size = len(data["frame"])
    if not size or set(data) != {*shapes, "frame", "clip_id"}:
        raise ValueError("Missing or empty collected dataset")
    if data["frame"].shape != (size,) or data["clip_id"].shape != (size,):
        raise ValueError("Collection row identities differ")
    if any(not np.issubdtype(data[name].dtype, np.integer) for name in ("frame", "clip_id")):
        raise ValueError("Frame and clip identities must be integers")
    for name, shape in shapes.items():
        if data[name].shape != (size, *shape) or not np.isfinite(data[name]).all():
            raise ValueError("Invalid collection tensor: "+name)
    if (not np.issubdtype(data["prefix"].dtype, np.integer)
            or np.any(data["prefix"] < 0) or np.any(data["prefix"] > np.array((10, 5, 4, 2, 1)))):
        raise ValueError("Invalid available preview-prefix counts")
    keys = set()
    for index in range(size):
        clip_id, frame = int(data["clip_id"][index]), int(data["frame"][index])
        if (clip_id, frame) in keys or not 0 <= clip_id < len(jobs):
            raise ValueError("Duplicate or unknown physical sample identity")
        keys.add((clip_id, frame))
        job, trace = jobs[clip_id], traces[clip_id]
        if (job["split"] != split or not 9 <= frame < job["maximum_steps"]
                or frame % STRIDE or frame >= len(trace["canonical_action"])):
            raise ValueError("Collection split, history or future boundary crossed")
        if (not np.array_equal(data["actual_state"][index], trace["actual"][frame])
                or not np.allclose(data["teacher_action"][index], trace["canonical_action"][frame],
                                   atol=2e-5, rtol=2e-5)):
            raise ValueError("A training state or teacher action does not match the actual rollout")
    return dict(records=size, source_clips=len({int(i) for i in data["clip_id"]}),
                physical_state_trace_match=True, teacher_action_trace_match=True)


def collect(args, config, runtime, plan, frozen):
    from common import identity, require, save
    from delivery import read
    from gpu_preview_worker import GpuPolicyProcess
    from interim_review import load_physical_robot
    from paused_preview import checkpoint, gpu_command
    from preview_review import require_plant

    torch = require_plant(runtime)
    from supermimic.data import Q1Motions
    from supermimic.simulator import Q1Simulator

    selected, baseline = checkpoint(args, config)
    robot, settings = load_physical_robot(baseline["path"])
    jobs = read(args.output/"collection_plan.json")["jobs"]
    rows = {Path(row["q1_primebot"]["path"]).stem: row for row in frozen["rows"]}
    command = gpu_command(args, config, runtime, plan, selected, baseline)
    command[2] = str(Path(__file__).with_name("student_training_worker.py"))
    command.extend(("--output", str(args.output), "--student-plan", str(args.student_plan),
                    "--deadline-ns", str(args.deadline_ns)))
    actor = GpuPolicyProcess(command, args.output/"native_worker.log", config["expected_gpu"])
    clips, traces = [], {}
    try:
        require(actor.ready["checkpoint"] == selected and actor.ready["metadata"]["iteration"] == 32000,
                "Collector loaded another teacher checkpoint")
        for clip_id, job in enumerate(jobs):
            item = rows[job["clip"]]
            require(paired_source(item, job), "Collection source differs from frozen paired reference")
            reference = Q1Motions(robot, [Path(item["q1_primebot"]["path"])], device="cpu").load(0)
            actor.request(dict(op="start", clip=job["clip"], seed=0, interface="human", clip_id=clip_id))
            simulator = Q1Simulator(robot, 1, "cpu", settings["action_scale"],
                use_tn=settings.get("use_tn", False), physics_timestep=settings.get("physics_timestep"))
            simulator.reset(torch.tensor([0]), reference[:1], randomize=False)
            state = simulator.state()
            actual, actions, failure = [state[0].numpy().copy()], [], None
            for frame in range(job["maximum_steps"]):
                require(time.monotonic_ns() < args.deadline_ns-60_000_000_000,
                        "Collection exhausted the bounded training wall budget")
                reply = actor.action(frame, state[0])
                action = torch.tensor(reply["native_action"], dtype=state.dtype)[None]
                canonical = np.asarray(reply["canonical_action"], dtype=np.float32)
                require(action.shape == (1, 22) and canonical.shape == (29,)
                        and torch.isfinite(action).all().item() and np.isfinite(canonical).all(),
                        "Invalid actual teacher action")
                state = simulator.step(action)
                require(torch.isfinite(state).all().item(), "Nonfinite collected physical state")
                actual.append(state[0].numpy().copy())
                actions.append(canonical)
                if robot.terminated(state, reference[frame+1:frame+2]).item():
                    failure = frame+1
                    break
            trace = dict(actual=np.asarray(actual), canonical_action=np.asarray(actions))
            traces[clip_id] = trace
            path = args.output/f"collection_{clip_id:02d}.npz"
            with path.open("xb") as stream:
                np.savez(stream, **trace)
            clips.append(dict(clip_id=clip_id, split=job["split"], clip=job["clip"],
                actual_steps=len(actions), requested_steps=job["maximum_steps"],
                failure_frame=failure, prefix_complete=failure is None, trace=identity(path)))
            save(args.output/"collection_progress.json", dict(completed_clips=len(clips),
                physical_steps=sum(row["actual_steps"] for row in clips), latest=clips[-1]))
        data_receipt = actor.request(dict(op="finish_data"))
        verified = {}
        for split in ("train", "validation"):
            path = args.output/(split+"_data.npz")
            with np.load(path, allow_pickle=False) as source:
                verified[split] = validate_records(dict(source), split, jobs, traces)
            require(verified[split]["source_clips"] >= 2 and verified[split]["records"] >= 32,
                    "Insufficient real source coverage for "+split)
        save(args.output/"collection_result.json", dict(stage="ACTUAL_TEACHER_STATE_COLLECTION",
            status="PASSED", clips=clips, dataset_receipt=data_receipt, data_validation=verified,
            actual_physical_steps=sum(row["actual_steps"] for row in clips),
            ideal_reference_states_used_as_actual=False, test_sources_read=False))
        remaining = (args.deadline_ns-time.monotonic_ns())/1e9
        require(remaining >= 60, "Insufficient reserved time for bounded fit and checkpoint cleanup")
        actor.timeout = remaining+5
        result = actor.request(dict(op="train"))
        actor.close()
        require(actor.child.returncode == 0, "Training worker did not exit cleanly")
        return result
    finally:
        if not actor.closed:
            actor.close(graceful=False)
