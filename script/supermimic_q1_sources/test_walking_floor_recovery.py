"""Signed restoring bounds, real Q1 distance derivatives, and an actual recovery QP."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np

from probe_walking_floor_recovery import RecoveringFloorLimit, restoring_upper_bound
from probe_walking_contact_floor import floor_limit


class RecoveryTests(unittest.TestCase):
    def test_signed_bound_and_dt_convention(self):
        for dt in (.02, .005):
            for distance in (-.04, -.001, .0005, .04):
                actual = restoring_upper_bound(distance, .0005, .85 * dt, dt)
                self.assertAlmostEqual(actual, .85 * (distance - .0005))
                self.assertEqual(actual < 0, distance < .0005)

    def test_invalid_numeric_contract_rejected(self):
        for values in ((0, .0005, 1, 0), (np.nan, .0005, 1, .02),
                       (0, -.1, 1, .02), (0, .0005, -1, .02), (0, .0005, 1, .02, .1)):
            with self.assertRaises(ValueError):
                restoring_upper_bound(*values)

    def test_penetrated_spheres_recover_and_positive_clearance_is_preserved(self):
        import mink
        import mujoco as mj

        xml = """<mujoco><worldbody><geom name="floor" type="plane" size="1 1 .1"/>
        <body name="pelvis"><freejoint/><inertial mass="1" pos="0 0 0" diaginertia=".1 .1 .1"/>
        <body name="left_ankle_roll_link" pos="0 .1 .07"><joint name="left_z" type="slide" axis="0 0 1"/>
        <geom type="sphere" size=".03" mass="1"/></body>
        <body name="right_ankle_roll_link" pos="0 -.1 .07"><joint name="right_z" type="slide" axis="0 0 1"/>
        <geom type="sphere" size=".03" mass="1"/></body></body></worldbody></mujoco>"""
        for floor in (0., -.04):
            for dt in (.02, .005):
                for start_gap in (-.01, .04):
                    model = mj.MjModel.from_xml_string(xml)
                    model.geom_pos[model.geom("floor").id, 2] = floor
                    configuration = mink.Configuration(model)
                    q = configuration.data.qpos.copy()
                    q[7:] = floor + start_gap - .04
                    configuration.update(q)
                    root = mink.FrameTask("pelvis", "body", 1., 1.)
                    root.set_target(mink.SE3.from_translation(np.zeros(3)))
                    tasks = [mink.FrameTask(side + "_ankle_roll_link", "body", 200., 0.) for side in ("left", "right")]
                    for task, y in zip(tasks, (.1, -.1)):
                        task.set_target(mink.SE3.from_translation(np.array([0, y, floor - .1])))
                    bound = RecoveringFloorLimit(floor_limit(mink, model, dt), mj)
                    con = bound.compute_qp_inequalities(configuration, dt)
                    if start_gap < 0:
                        self.assertTrue(np.any(con.h < 0))
                        self.assertFalse(np.all(con.G @ np.zeros(model.nv) <= con.h))
                    for _ in range(30):
                        velocity = mink.solve_ik(configuration, tasks, dt, "quadprog", damping=.001,
                            limits=[bound], constraints=[root])
                        configuration.integrate_inplace(velocity, dt)
                    feet = [gid for gid in range(model.ngeom) if model.geom_bodyid[gid]]
                    low = float((configuration.data.geom_xpos[feet, 2] - .03 - floor).min())
                    self.assertGreaterEqual(low, .0005 - 1e-8)

    def test_real_q1_signed_distance_derivatives_after_root_changes(self):
        import mink
        import mujoco as mj

        root = Path("/opt/data/private/YuanQi_Q1")
        sys.path.insert(0, str(root / "script/supermimic_q1_phuma"))
        from retarget import ContactCorrectedAssets
        from guard_phuma_batch import MemoryBoundBudget

        memory = MemoryBoundBudget(None, int(28.75 * 1024 ** 3), 14)
        memory.memory()
        assets = ContactCorrectedAssets(root)
        model = assets.model
        report = json.loads((root / "data/supermimic_for_q1/PHUMA_P14_walking_root_offset_20261025/probe/receipt.json").read_text())
        rng = np.random.default_rng(2026102515)
        configuration = mink.Configuration(model)
        bound = RecoveringFloorLimit(floor_limit(mink, model, .005), mj)
        floor = model.geom("floor").id
        self.assertEqual(len(bound.geom_id_pairs), 10)
        for row in report["rows"]:
            memory.memory()
            self.assertEqual(row["source_split"], "train")
            with np.load(row["prior_arrays"]["path"], allow_pickle=False) as saved:
                start = saved["qpos"][0].copy()
            start[2] += row["decision"]["root_delta_z_m"]
            with np.load(row["arrays"]["path"], allow_pickle=False) as saved:
                worst = int(np.argmin(saved["all_frame_foot_height_m"].min(axis=1)))
                states = [start, saved["qpos"][worst].copy()]
            for state in states:
                configuration.update(state)
                constraint = bound.compute_qp_inequalities(configuration, .005)
                active = np.isfinite(constraint.h)
                self.assertTrue(np.any(constraint.h < 0))
                self.assertFalse(np.all(constraint.G @ np.zeros(model.nv) <= constraint.h))
                foot_ids = [second if first == floor else first for first, second in bound.geom_id_pairs]
                self.assertTrue(np.all(model.geom_type[foot_ids] == mj.mjtGeom.mjGEOM_SPHERE))
                for _ in range(8):
                    direction = rng.normal(size=model.nv)
                    direction /= np.linalg.norm(direction)
                    distance = []
                    for sign in (1, -1):
                        moved = state.copy()
                        mj.mj_integratePos(model, moved, direction, sign * 1e-6)
                        configuration.update(moved)
                        distance.append(configuration.data.geom_xpos[foot_ids, 2]
                            - model.geom_size[foot_ids, 0] - model.geom_pos[floor, 2])
                    derivative = (distance[0] - distance[1]) / 2e-6
                    np.testing.assert_allclose((constraint.G @ direction)[active], -derivative[active], rtol=0, atol=2e-7)
        memory.memory()


if __name__ == "__main__":
    unittest.main()
