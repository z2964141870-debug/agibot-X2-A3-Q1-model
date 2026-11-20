"""Check modal byte parity against the archived implementation and time a reference step."""

import importlib.util
import json
import time
from pathlib import Path

from bench import np
from q1_kinematics_np import quat_xyzw_to_mat, rotz_np
from reference import HandoffReference
from restart_profile import PROFILES, RestartProfile


def validate_offsets():
    damped = PROFILES["damped_moderate"]
    for t in (0, 29, 30, 90, 200):
        assert damped.ankle_offset(t, 10.0, [0, 0]) is None
    for t in range(31, 90):
        for rate in (-100.0, -0.12, 0.0, 0.12, 100.0):
            a = damped.ankle_offset(t, rate, [0.0, 0.04])
            b = damped.ankle_offset(t, -rate, [0.04, 0.0])
            assert abs(a).max() <= 0.03000001 and a[5] == -b[11]
            assert np.count_nonzero(a) == int(abs(rate) > 0.12) and a[11] == 0
        assert PROFILES["published"].ankle_offset(t, 1.0, [0, 0]) is None
    assert damped.ankle_offset(50, np.nan, [0, 0]) is None
    assert damped.ankle_offset(50, 0.3, [np.inf, 0]) is None
    for args in ((0, 0.0), (2.5, 0.0), (30, -0.1), (30, np.inf)):
        try:
            RestartProfile(*args)
        except ValueError:
            continue
        raise AssertionError(args)


def test_reference(root):
    spec = importlib.util.spec_from_file_location(
        "archived_reference", root / "previous_study/reference.py"
    )
    archived = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archived)
    clip = np.loadtxt(root / "formal_03/input_tracks/loop_figure8_walk.csv", delimiter=",").astype(
        np.float32
    )
    pos = clip[:, :3]
    rot = quat_xyzw_to_mat(clip[:, 3:7])
    joints = clip[:, 7:]
    samples = {"archived": [], "cached": []}
    exact = 0
    for offset in (0.0, 1.8, -np.pi + 0.0001):
        for ramp in (15, 20, 30):
            anchor = rotz_np(offset) @ rot[0]
            old = archived.HandoffReference(pos[0], anchor, joints[0], resume_ticks=ramp)
            new = HandoffReference(pos[0], anchor, joints[0], resume_ticks=ramp)
            for i in range(250):
                args = (pos[i], rot[i], joints[i])
                # Alternate which implementation runs first on this CPU.
                order = (("cached", new), ("archived", old))
                outputs = {}
                for key, instance in order if i % 2 else order[::-1]:
                    start = time.perf_counter_ns()
                    outputs[key] = instance.step(*args)
                    elapsed = time.perf_counter_ns() - start
                    if i >= 30:
                        samples[key].append(elapsed / 1000)
                for a, b in zip(outputs["cached"], outputs["archived"], strict=True):
                    assert (
                        a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
                    ), (offset, ramp, i)
                exact += 1
    return dict(
        exact_ticks=exact,
        timed_ticks_per_implementation=len(samples["cached"]),
        reference_step_us={
            k: dict(p50=float(np.median(v)), p95=float(np.percentile(v, 95)))
            for k, v in samples.items()
        },
    )


if __name__ == "__main__":
    validate_offsets()
    result = test_reference(Path.cwd())
    Path("reference_check.json").write_text(json.dumps(result, indent=2))
