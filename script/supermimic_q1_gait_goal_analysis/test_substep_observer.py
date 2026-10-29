"""Exercise the real backend control-loop methods on nonphysical stand-ins."""

import ast
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any
import unittest

import numpy as np
import torch

from substep_observer import SubstepObserver, WarpSampler

BACKEND_SOURCE = Path("/opt/data/private/ZHY/data/environments/supermimic_q1_hp_20261026/lib/python3.11/site-packages/unisim/backend/mjwarp/backend.py")


class Sampler:
    def __init__(self, backend, fail_after=None):
        self.backend, self.fail_after = backend, fail_after
        self.events = []

    def before(self, control, substep):
        self.events.append(("before", control, substep, self.backend.value))
        return self.backend.value

    def after(self, control, substep, before):
        self.events.append(("after", control, substep, before, self.backend.value))
        if self.fail_after == substep:
            raise ValueError("Synthetic observer failure")


class Backend:
    def __init__(self, graph=False):
        self.value, self.launches, self.force_seen, self.uploads = 0., [], [], []
        self._cuda_graph_enabled = graph
        self._step_graph = object()
        self._device_model = object()
        self._device_data = SimpleNamespace(xfrc_applied=np.zeros(1))
        self._warp = SimpleNamespace(capture_launch=lambda graph: self.kernel("graph"))
        self._mujoco_warp = SimpleNamespace(step=lambda model, data: self.kernel("eager"))
        self._num_envs, self._nu, self._sim_dt = 1, 22, .005
        self._pre_step_control_fn = None
        self._xfrc_pending, self._device_xfrc_pending = True, False
        self._xfrc_staging = np.array([2.])
        self._time_cache = np.zeros(1, np.float32)
        self._tracked_body_names = ("foot",)
        self._entity_faulted = False
        self.ctrl = torch.zeros(1, 22)

    def kernel(self, mode):
        # Arithmetic stand-in only: no native physics module or environment exists.
        self.launches.append(mode)
        self.force_seen.append(float(self._device_data.xfrc_applied[0]))
        self.value += float(self.ctrl.sum()) + self.force_seen[-1]

    def _require_entity_healthy(self):
        if self._entity_faulted:
            raise RuntimeError("Faulted")

    def _validate_nsteps(self, value):
        if type(value) is not int or value < 1:
            raise ValueError("Bad step count")
        return value

    def _stable_tensor_field_view(self, name):
        if name != "ctrl":
            raise ValueError("Unexpected field")
        return self.ctrl

    def _validate_torch_operand(self, name, value, shape):
        if tuple(value.shape) != shape:
            raise ValueError("Bad shape")
        return value

    def _torch_execution_stream(self, device):
        return "synthetic_stream"

    def _upload(self, target, source):
        self.uploads.append(source.copy())
        np.copyto(target, source)

    def _step_completion(self, stream):
        return ("synthetic_completion", stream)


def source_backend():
    if not BACKEND_SOURCE.is_file():
        raise FileNotFoundError("Run this exact-source suite in the existing server interpreter")
    tree = ast.parse(BACKEND_SOURCE.read_text())
    methods = [node for cls in tree.body if isinstance(cls, ast.ClassDef) for node in cls.body
               if isinstance(node, ast.FunctionDef) and node.name in ("step_tensor", "_execute_device_steps")]
    if len(methods) != 2:
        raise ValueError("Backend method contract changed")
    space = dict(Any=Any, time=time, np=np)
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(BACKEND_SOURCE), "exec"), space)
    return type("SourceMethodStandIn", (Backend,), {name: space[name] for name in
        ("step_tensor", "_execute_device_steps")})


class SubstepObserverTests(unittest.TestCase):
    def test_real_source_eager_and_graph_keep_outer_control_and_force_lifecycle(self):
        Runtime = source_backend()
        action = torch.full((1, 22), .25)
        for graph in (False, True):
            with self.subTest(graph=graph):
                baseline, watched = Runtime(graph), Runtime(graph)
                expected = baseline.step_tensor(action, 4)
                sampler = Sampler(watched)
                with SubstepObserver(watched, sampler, 1) as observer:
                    actual = watched.step_tensor(action, 4)
                    self.assertEqual(observer.control_calls, 1)
                    self.assertEqual(observer.completed_substeps, 4)
                    self.assertEqual(observer.attempted_native_substeps, 4)
                    self.assertEqual(observer.returned_native_substeps, 4)
                    self.assertFalse(observer.faulted)
                self.assertEqual(watched.value, baseline.value)
                self.assertEqual(watched.launches, baseline.launches)
                self.assertEqual(watched.force_seen, [2.] * 4)
                self.assertEqual(len(watched.uploads), 2)
                np.testing.assert_array_equal(watched._time_cache, baseline._time_cache)
                self.assertFalse(watched._xfrc_pending)
                self.assertTrue(watched._host_cache_stale)
                self.assertTrue(watched._tracked_body_state_dirty)
                self.assertEqual(actual["completion"], expected["completion"])
                self.assertEqual([e[0] for e in sampler.events], ["before", "after"] * 4)
                self.assertNotIn("_execute_device_steps", vars(watched))

    def test_invalid_substep_count_and_control_bound_prevent_extra_launches(self):
        backend = source_backend()()
        with SubstepObserver(backend, Sampler(backend), 1) as observer:
            for count in (1, 3, 5, True):
                with self.assertRaises(ValueError):
                    backend._execute_device_steps(count)
            self.assertEqual(len(backend.launches), 0)
            backend._execute_device_steps(4)
            with self.assertRaises(RuntimeError):
                backend._execute_device_steps(4)
            self.assertEqual(observer.completed_substeps, 4)
            self.assertEqual(len(backend.launches), 4)

    def test_failed_sampler_stops_and_cannot_resume_and_restores_method(self):
        backend = source_backend()()
        with SubstepObserver(backend, Sampler(backend, fail_after=1), 2) as observer:
            with self.assertRaises(ValueError):
                backend._execute_device_steps(4)
            self.assertEqual(observer.completed_substeps, 1)
            self.assertEqual(observer.attempted_native_substeps, 2)
            self.assertEqual(observer.returned_native_substeps, 2)
            self.assertTrue(observer.faulted)
            with self.assertRaises(RuntimeError):
                backend._execute_device_steps(4)
            self.assertEqual(len(backend.launches), 2)
        self.assertNotIn("_execute_device_steps", vars(backend))

    def test_existing_instance_method_is_restored_exactly(self):
        backend = source_backend()()
        original = backend._execute_device_steps
        backend._execute_device_steps = original
        with SubstepObserver(backend, Sampler(backend), 1):
            backend._execute_device_steps(4)
        self.assertIs(backend._execute_device_steps, original)

    def test_observer_lifetime_is_single_use_and_rejects_bad_budgets(self):
        backend = source_backend()()
        for maximum in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                SubstepObserver(backend, Sampler(backend), maximum)
        observer = SubstepObserver(backend, Sampler(backend), 2)
        with observer:
            backend._execute_device_steps(4)
        with self.assertRaises(RuntimeError):
            observer.__enter__()
        self.assertNotIn("_execute_device_steps", vars(backend))

    def test_other_hook_ownership_is_preserved(self):
        backend = source_backend()()
        observer = SubstepObserver(backend, Sampler(backend), 1).__enter__()
        other = lambda count: None
        backend._execute_device_steps = other
        with self.assertRaises(RuntimeError):
            observer.__exit__(None, None, None)
        self.assertIs(backend._execute_device_steps, other)

    def test_inner_native_failure_marks_uncertainty_without_retry(self):
        backend = source_backend()()
        def fail(count):
            backend.kernel("partial_launch_then_error")
            raise ValueError("Synthetic native launch failure")
        backend._execute_device_steps = fail
        with SubstepObserver(backend, Sampler(backend), 1) as observer:
            with self.assertRaises(ValueError):
                backend._execute_device_steps(4)
            self.assertTrue(observer.faulted)
            self.assertEqual(observer.completed_substeps, 0)
            self.assertEqual(observer.attempted_native_substeps, 1)
            self.assertEqual(observer.returned_native_substeps, 0)
            self.assertEqual(len(backend.launches), 1)
        self.assertIs(backend._execute_device_steps, fail)

    def test_returned_async_launch_is_not_a_verified_completed_substep(self):
        backend = source_backend()()
        with SubstepObserver(backend, Sampler(backend, fail_after=0), 1) as observer:
            with self.assertRaises(ValueError):
                backend._execute_device_steps(4)
            self.assertEqual(observer.completed_substeps, 0)
            self.assertEqual(observer.attempted_native_substeps, 1)
            self.assertEqual(observer.returned_native_substeps, 1)
            self.assertTrue(observer.faulted)
            self.assertEqual(len(backend.launches), 1)


class SamplerTests(unittest.TestCase):
    def sampler(self):
        sample = object.__new__(WarpSampler)
        sample.wp = SimpleNamespace(synchronize=lambda: None, to_torch=torch.from_numpy)
        sample.data = SimpleNamespace(**{key: np.full((1, 1), 2., dtype=np.float32)
            for key in WarpSampler.STATE_FIELDS + WarpSampler.SOLVE_FIELDS})
        sample.data.time = np.array([.005], dtype=np.float32)
        sample.data.xpos = np.zeros((1, 2, 3), dtype=np.float32)
        sample.data.xmat = np.broadcast_to(np.eye(3), (1, 2, 3, 3)).copy()
        sample.geometry = SimpleNamespace(heights=lambda p, r: p[..., 2] - .1)
        sample.foot_ids, sample.records = [0, 1], []
        sample.contact_summary = lambda: {"contact_count": np.array(1)}
        before = sample.read(sample.data, WarpSampler.STATE_FIELDS)
        before["time"][:] = 0
        before["qpos"][:] = 1
        return sample, before

    def test_pre_solve_post_timestamps_and_geometry_are_separate(self):
        sample, before = self.sampler()
        sample.after(3, 2, before)
        archive = sample.archive()
        self.assertEqual(archive["pre_time"][0, 0], 0.)
        self.assertAlmostEqual(float(archive["post_time"][0, 0]), .005)
        self.assertEqual(archive["pre_qpos"][0, 0, 0], 1.)
        self.assertEqual(archive["post_qpos"][0, 0, 0], 2.)
        self.assertEqual(archive["solve_actuator_force"][0, 0, 0], 2.)
        self.assertEqual(archive["control_index"].tolist(), [3])
        self.assertEqual(archive["substep_index"].tolist(), [2])
        np.testing.assert_array_equal(archive["solve_collision_sole_z_m"], np.full((1, 1, 2), -.1, dtype=np.float32))

    def test_bad_clock_changed_control_and_contact_mutation_fail(self):
        for field in ("clock", "control", "mutation"):
            with self.subTest(field=field):
                sample, before = self.sampler()
                if field == "clock":
                    sample.data.time[:] = .02
                elif field == "control":
                    sample.data.ctrl[:] = 3
                else:
                    def mutate():
                        sample.data.qvel[:] = 3
                        return {"contact_count": np.array(1)}
                    sample.contact_summary = mutate
                with self.assertRaises((ValueError, AssertionError)):
                    sample.after(0, 0, before)
                self.assertEqual(sample.records, [])


if __name__ == "__main__":
    unittest.main()
