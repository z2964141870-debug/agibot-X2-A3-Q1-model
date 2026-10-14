"""Analytical positive and negative controls for proposed reference contracts."""
import math
import unittest
from script.headband.reference_contract import (Pose, pelvis_from_camera,
    camera_in_continuous_odom, velocity, ClockAlignment, observation_age)


class ReferenceContractTests(unittest.TestCase):
    def assertPose(self, a, b):
        for x, y in zip((a.x, a.y, a.yaw), (b.x, b.y, b.yaw)):
            self.assertAlmostEqual(x, y, places=10)

    def test_turning_head_keeps_pelvis_fixed(self):
        # Camera rotates around the head with a known 10 cm mounting offset.
        camera = Pose(0, .1, math.pi / 2)
        result = pelvis_from_camera(camera, Pose(0, 0, math.pi / 2), Pose(.1, 0, 0))
        self.assertPose(result, Pose(0, 0, 0))

    def test_fixed_head_transform_invents_motion_negative_control(self):
        wrong = pelvis_from_camera(Pose(0, .1, math.pi / 2), Pose(0, 0, 0), Pose(.1, 0, 0))
        self.assertAlmostEqual(wrong.yaw, math.pi / 2)
        # Full fixed pelvis-camera extrinsic also retains apparent head yaw.
        self.assertNotAlmostEqual(wrong.yaw, 0)

    def test_neck_translation_requires_dynamic_compensation(self):
        camera = Pose(.3, 0, 0)
        self.assertPose(pelvis_from_camera(camera, Pose(.2, 0, 0), Pose(.1, 0, 0)), Pose(0, 0, 0))
        wrong = pelvis_from_camera(camera, Pose(0, 0, 0), Pose(.1, 0, 0))
        self.assertAlmostEqual(wrong.x, .2)

    def test_actual_body_motion_is_not_removed(self):
        p = pelvis_from_camera(Pose(1, .1, math.pi / 2), Pose(0, 0, math.pi / 2), Pose(.1, 0, 0))
        self.assertPose(p, Pose(1, 0, 0))

    def test_world_coordinates_change_consistently(self):
        g = Pose(7, -3, math.pi / 3); p = Pose(1, 2, .3)
        ph, hc = Pose(.1, -.2, .5), Pose(.1, 0, -.1)
        camera = p.compose(ph).compose(hc)
        self.assertPose(pelvis_from_camera(g.compose(camera), ph, hc), g.compose(p))

    def test_map_relocalization_does_not_invent_odom_motion(self):
        odom_camera = Pose(.5, .2, .1); map_odom = Pose(9, -2, .8)
        corrected = camera_in_continuous_odom(map_odom.compose(odom_camera), map_odom)
        self.assertPose(corrected, odom_camera)
        self.assertEqual(tuple(round(x, 8) for x in velocity(odom_camera, corrected, .02, 'o1', 'o1')), (0, 0))

    def test_unknown_map_epoch_rejected(self):
        with self.assertRaises(ValueError): velocity(Pose(0, 0, 0), Pose(9, 0, 0), .02, 'a', 'b')

    def test_clock_delay_not_erased_by_receipt_time(self):
        r = observation_age(10, 10.1, 10.12)
        self.assertAlmostEqual(r['capture_to_use_s'], .12)
        self.assertAlmostEqual(r['host_buffer_to_use_s'], .02)

    def test_explicit_offset_and_drift_mapping(self):
        c = ClockAlignment('boot1', 1.00005e-6, 20, 'independent_sync_events')
        self.assertAlmostEqual(c.host_time(2_000_000, 'boot1'), 22.0001)

    def test_receipt_only_calibration_rejected(self):
        c = ClockAlignment('boot1', 1e-6, 20, 'host_receipt')
        with self.assertRaises(ValueError): c.host_time(2_000_000, 'boot1')

    def test_device_reset_mapping_rejected(self):
        c = ClockAlignment('boot1', 1e-6, 20, 'independent_sync_events')
        with self.assertRaises(ValueError): c.host_time(200, 'boot2')

    def test_missing_and_invalid_intervals_rejected(self):
        for dt in (0, -1, .201, float('nan')):
            with self.assertRaises(ValueError): velocity(Pose(0, 0, 0), Pose(1, 0, 0), dt, 'o', 'o')
        with self.assertRaises(ValueError): observation_age(10.2, 10.1, 10.3)


if __name__ == '__main__': unittest.main()
