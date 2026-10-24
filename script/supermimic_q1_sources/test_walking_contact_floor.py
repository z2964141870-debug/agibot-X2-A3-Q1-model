"""Native foot-floor distance bounds, including the common ground-origin shift."""

import unittest

import numpy as np

from probe_walking_contact_floor import floor_limit


class FloorConstraintTests(unittest.TestCase):
    def test_both_feet_registered_and_constraint_blocks_penetration(self):
        import mink
        import mujoco

        xml = """<mujoco><worldbody><geom name="floor" type="plane" size="1 1 .1"/>
          <body name="left_ankle_roll_link" pos="0 .1 .07"><joint name="left_z" type="slide" axis="0 0 1"/>
          <geom type="sphere" size=".03" mass="1"/></body>
          <body name="right_ankle_roll_link" pos="0 -.1 .07"><joint name="right_z" type="slide" axis="0 0 1"/>
          <geom type="sphere" size=".03" mass="1"/></body></worldbody></mujoco>"""
        for floor in (0., -.04):
            model = mujoco.MjModel.from_xml_string(xml)
            model.geom_pos[model.geom("floor").id, 2] = floor
            configuration = mink.Configuration(model)
            tasks = [mink.FrameTask(side + "_ankle_roll_link", "body", 200., 0.) for side in ("left", "right")]
            for task, y in zip(tasks, (.1, -.1)):
                task.set_target(mink.SE3.from_translation(np.array([0., y, floor - .1])))
            for _ in range(30):
                bound = floor_limit(mink, model, .005)
                self.assertEqual(len(bound.geom_id_pairs), 2)
                velocity = mink.solve_ik(configuration, tasks, .005, "quadprog", limits=[bound])
                configuration.integrate_inplace(velocity, .005)
                gids = [gid for gid in range(model.ngeom) if model.geom_bodyid[gid]]
                self.assertGreaterEqual(float((configuration.data.geom_xpos[gids, 2] - .03 - floor).min()), -1e-8)


if __name__ == "__main__":
    unittest.main()
