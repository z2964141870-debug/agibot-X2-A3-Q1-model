"""Full CPU plant snapshots, distinct from lossy79-dimensional observations."""

import copy
import random

import numpy as np

from common import require


TENSORS = ("context", "startup_context", "startup_sampled", "push_enabled", "push_timer",
    "pushed", "lagged", "last_action", "transition_state")


def snapshot(simulator):
    import torch
    require(simulator.device == "cpu" and simulator.num_envs == 1, "Snapshot only the common single CPU plant")
    return dict(data=copy.copy(simulator.cpu_data[0]),
        tensors={name:getattr(simulator, name).clone() for name in TENSORS},
        torch_rng=torch.get_rng_state().clone(), numpy_rng=copy.deepcopy(np.random.get_state()),
        python_rng=random.getstate(), state=simulator.state().clone())


def restore(simulator, value):
    import mujoco
    import torch
    require(simulator.device == "cpu" and simulator.num_envs == 1, "Restore only the admitted CPU plant")
    mujoco.mj_copyData(simulator.cpu_data[0], simulator.cpu_models[0], value["data"])
    for name in TENSORS:
        getattr(simulator, name).copy_(value["tensors"][name])
    simulator._sync_cpu()
    torch.set_rng_state(value["torch_rng"])
    np.random.set_state(value["numpy_rng"])
    random.setstate(value["python_rng"])
    require(torch.equal(simulator.state(), value["state"]), "Full plant snapshot did not restore the exact observation")
    return simulator.state()


def checked_state(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape == (79,) and np.isfinite(actual).all() and np.isfinite(expected).all(),
        "Invalid restored state shape or finite values")
    difference = float(np.max(np.abs(actual-expected)))
    require(np.allclose(actual, expected, atol=2e-6, rtol=2e-6), "Saved-action physical prefix differs")
    return difference


def fixed_jobs(collection_jobs):
    require(len(collection_jobs) == 9, "Require the original nine owned sources")
    jobs = []
    for clip_id in (0, 2, 6, 8):
        item = collection_jobs[clip_id]
        require(item["split"] == ("train" if clip_id < 6 else "validation"), "No test or new held-out source")
        for anchor in (64, 96):
            jobs.append(dict(clip_id=clip_id, clip=item["clip"], split=item["split"], anchor=anchor,
                maximum_steps=80, arm_order=["student_f0", "privileged_teacher_h10"] if len(jobs)%2 == 0
                    else ["privileged_teacher_h10", "student_f0"]))
    return jobs
