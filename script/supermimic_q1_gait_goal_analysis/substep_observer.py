"""Observe the existing inner Warp loop without splitting outer control steps."""

import numpy as np


class SubstepObserver:
    def __init__(self, backend, sampler, maximum_controls):
        if type(maximum_controls) is not int or maximum_controls <= 0:
            raise ValueError("Expected a positive control-step bound")
        self.backend, self.sampler = backend, sampler
        self.maximum_controls = maximum_controls
        self.control_calls = self.completed_substeps = 0
        self.attempted_native_substeps = self.returned_native_substeps = 0
        self.faulted = self.installed = self.used = False

    def __enter__(self):
        if self.used:
            raise RuntimeError("A single-use observer cannot be installed again")
        self.used = True
        self.had_local = "_execute_device_steps" in vars(self.backend)
        self.original = self.backend._execute_device_steps

        def observed(nsteps):
            if self.faulted:
                raise RuntimeError("Failed observation cannot be resumed")
            if type(nsteps) is not int or nsteps != 4:
                raise ValueError("Expected the unchanged four-substep control call")
            if self.control_calls >= self.maximum_controls:
                raise RuntimeError("Single-use observation control bound reached")
            control = self.control_calls
            self.control_calls += 1
            try:
                for substep in range(nsteps):
                    before = self.sampler.before(control, substep)
                    self.attempted_native_substeps += 1
                    self.original(1)
                    self.returned_native_substeps += 1
                    self.sampler.after(control, substep, before)
                    self.completed_substeps += 1
            except BaseException:
                self.faulted = True
                raise

        self.wrapper = observed
        self.backend._execute_device_steps = observed
        self.installed = True
        return self

    def __exit__(self, *error):
        if self.backend._execute_device_steps is not self.wrapper:
            raise RuntimeError("Inner hook ownership changed; refusing to overwrite another observer")
        if self.had_local:
            self.backend._execute_device_steps = self.original
        else:
            del self.backend._execute_device_steps
        self.installed = False


class WarpSampler:
    """Copy native arrays after each solve/integration; never call forward or step."""

    STATE_FIELDS = ("qpos", "qvel", "time", "qacc_warmstart", "qacc", "ctrl",
                    "qfrc_applied", "xfrc_applied")
    SOLVE_FIELDS = ("actuator_length", "actuator_velocity", "actuator_force", "qfrc_actuator",
                    "qfrc_constraint", "qfrc_bias", "qfrc_passive")
    MODEL_FIELDS = ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange")

    def __init__(self, env):
        import torch
        from evaluation import NativeContacts
        from sole_geometry import SoleGeometry

        self.torch = torch
        self.env, self.backend = env, env.backend
        self.contacts = NativeContacts(env)
        self.wp, self.data = self.contacts.wp, self.contacts.data
        self.model, self.device_model = self.contacts.model, self.contacts.gpu_model
        if env.num_envs != 1 or self.model.nu != 22 or self.model.nq != 29 or self.model.nv != 28:
            raise ValueError("Only one native Q1 world with22 actuators is supported")
        self.geometry = SoleGeometry.from_model(self.model)
        self.foot_ids = [self.model.body(foot.body).id for foot in self.geometry.feet]
        from saved_pd_demand import model_contract
        from unisupermimic.tasks.motion_tracking.q1.supermimic.robot import Q1_JOINT_NAMES
        self.joint_names = list(Q1_JOINT_NAMES)
        self.actuation = model_contract(self.model, self.joint_names)
        self.wp.synchronize()
        self.coefficients = self.read(self.device_model, self.MODEL_FIELDS)
        self.records = []

    def read(self, obj, fields):
        result = {name: self.wp.to_torch(getattr(obj, name)).detach().cpu().numpy().copy()
                  for name in fields}
        if not all(np.isfinite(value).all() for value in result.values()):
            raise FloatingPointError("Nonfinite native observation")
        return result

    def before(self, control, substep):
        self.wp.synchronize()
        actual = self.read(self.device_model, self.MODEL_FIELDS)
        if any(not np.array_equal(actual[name], expected) for name, expected in self.coefficients.items()):
            raise ValueError("Native actuator coefficients changed during a fixed diagnostic")
        return self.read(self.data, self.STATE_FIELDS)

    def contact_summary(self):
        import mujoco_warp as mw

        wp, data, binding, torch = self.wp, self.data, self.contacts, self.torch
        count = int(wp.to_torch(data.nacon)[0])
        if not 0 <= count <= data.naconmax:
            raise ValueError("Native contact count is outside allocated capacity")
        binding.forces.zero_()
        mw.contact_force(self.device_model, data, binding.contact_ids, True, binding.forces)
        wp.synchronize()
        worlds = wp.to_torch(data.contact.worldid)[:count].long()
        bodies = binding.geom_body[wp.to_torch(data.contact.geom)[:count].long()]
        points = wp.to_torch(data.contact.pos)[:count]
        vertical = wp.to_torch(binding.forces)[:count, 2].abs()
        cvel, com = wp.to_torch(data.cvel), wp.to_torch(data.subtree_com)
        force = torch.zeros(1, 2, device=self.env.device)
        weighted_speed = torch.zeros_like(force)
        for side in range(2):
            first = (binding.foot_owner[bodies[:, 0]] == side) & (bodies[:, 1] == 0)
            second = (binding.foot_owner[bodies[:, 1]] == side) & (bodies[:, 0] == 0)
            keep = first | second
            world, point = worlds[keep], points[keep]
            body = torch.where(first[keep], bodies[keep, 0], bodies[keep, 1])
            velocity = cvel[world, body, 3:] + torch.linalg.cross(
                cvel[world, body, :3], point - com[world, binding.root_ids[body]])
            force[:, side].index_add_(0, world, vertical[keep])
            weighted_speed[:, side].index_add_(0, world, vertical[keep] * velocity[:, :2].norm(dim=-1))
        return dict(contact_count=np.array(count, dtype=np.int32),
            foot_force_world_z_abs_N=force.detach().cpu().numpy().copy(),
            foot_force_weighted_slip_mps=(weighted_speed / force.clamp(min=1e-8)).detach().cpu().numpy().copy())

    def after(self, control, substep, before):
        self.wp.synchronize()
        after = self.read(self.data, self.STATE_FIELDS + self.SOLVE_FIELDS)
        if not np.allclose(after["time"] - before["time"], .005, rtol=0, atol=2e-6):
            raise ValueError("Native substep did not advance the expected5ms")
        if not np.array_equal(after["ctrl"], before["ctrl"]):
            raise ValueError("Control changed within the native substep")
        contact = self.contact_summary()
        preserved = self.read(self.data, self.STATE_FIELDS + self.SOLVE_FIELDS)
        if any(not np.array_equal(after[name], preserved[name]) for name in after):
            raise AssertionError("Read-only contact observation changed native dynamic/solve arrays")
        positions = self.wp.to_torch(self.data.xpos)[:, self.foot_ids].detach().cpu().numpy().copy()
        rotations = self.wp.to_torch(self.data.xmat)[:, self.foot_ids].detach().cpu().numpy().copy()
        record = {"control_index": np.array(control), "substep_index": np.array(substep),
                  **{"pre_" + key: value for key, value in before.items()},
                  **{"post_" + key: after[key] for key in self.STATE_FIELDS},
                  **{"solve_" + key: after[key] for key in self.SOLVE_FIELDS}, **contact,
                  "solve_foot_position_m": positions, "solve_foot_rotation": rotations,
                  "solve_collision_sole_z_m": self.geometry.heights(positions, rotations)}
        if not all(np.isfinite(value).all() for value in record.values()):
            raise FloatingPointError("Nonfinite recorded substep")
        self.records.append(record)

    def archive(self):
        if not self.records:
            raise ValueError("No completed native substeps were recorded")
        return {key: np.stack([row[key] for row in self.records]) for key in self.records[0]}

    def contract(self):
        return dict(physics_dt_s=.005, control_dt_s=.02, substeps_per_control=4,
            foot_body_ids=self.foot_ids, collision_geometry=self.geometry.identity(),
            native_actuator_ids_in_q1_joint_order=self.actuation["actuator_ids"].tolist(),
            q1_joint_names=self.joint_names,
            qpos_ids_in_q1_joint_order=self.actuation["qpos_ids"].tolist(),
            qvel_ids_in_q1_joint_order=self.actuation["qvel_ids"].tolist(),
            actuator_coefficients={name: value.tolist() for name, value in self.coefficients.items()},
            solve_time="pre_time of this5ms step; geometry/contact/effort from native forward/solve before integration",
            post_time="post_time and post_qpos/qvel after this5ms integration; no extra forward to refresh geometry",
            contact_scope="Solved world-Z force magnitudes on foot/floor, weighted contact-point speed; not20ms impulse",
            inner_hook="Only _execute_device_steps is wrapped; original(1) called four times, same graph/eager path",
            outer_control_upload_force_cleanup_and_time_cache="Original step_tensor called once and unchanged",
            mutation_guard="Contact extraction preserves copied state and solved-effort arrays exactly",
            no_extra_forward_or_solver=True, trajectory_equivalence_physically_verified=False)
