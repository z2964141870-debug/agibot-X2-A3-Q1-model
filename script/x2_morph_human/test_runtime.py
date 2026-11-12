"""Offline integration/causality tests. Uses the real released ONNX where stated."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace as S
import numpy as np

from . import artifacts
from .core import (Calibration, DT_NS, HumanBuffer, NotReady, SmplAdapter,
                   axis_angle, fk, inv, mul, pelvis_from_sensor, rotate)


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = artifacts.activate()
        cls.a = SmplAdapter(cls.root/"upstream/deploy/vr_teleop/human_skeleton.json", Calibration("test"))

    def packets(self, n=12):
        from .__main__ import demo_stream
        return list(demo_stream(self.a, n))

    def test_local_rotation_chain_and_metric_root(self):
        a = SmplAdapter(self.root/"upstream/deploy/vr_teleop/human_skeleton.json",
                        Calibration("zup", (1.,0,0,0)))
        v = np.zeros((24,3));v[0,2] = np.pi/2
        p, q = a.convert(v, [1.,2.,3.])
        np.testing.assert_allclose(p[0], [1,2,3], atol=1e-12)
        bx, by, bz = a.bones[1]
        np.testing.assert_allclose(p[1]-p[0], [-by,bx,bz], atol=1e-12)
        v[1,0] = np.pi/2
        p2, q2 = a.convert(v, [1.,2.,3.])
        self.assertGreater(np.linalg.norm(p2[4]-p[4]), .1)
        np.testing.assert_allclose(p2[2], p[2], atol=1e-12)  # other hip unaffected

    def test_zero_and_pi_axis_angles(self):
        q = axis_angle(np.array([[0.,0,0],[np.pi,0,0]]))
        np.testing.assert_allclose(rotate(q, np.array([[0.,1,0],[0.,1,0]])), [[0,1,0],[0,-1,0]], atol=1e-12)

    def test_original_interleaved_layout(self):
        from deploy.sim2sim.human import join_frames
        packets = self.packets(10)
        b = HumanBuffer(self.a.skeleton_id)
        for p in packets: b.push(p)
        w, timing = b.window(9*DT_NS,"replay_monotonic")
        np.testing.assert_allclose(w[0], join_frames(np.array(packets[0]["positions"]),np.array(packets[0]["quaternions"])))
        self.assertEqual(timing["slot_capture_ns"], list(range(0,10*DT_NS,DT_NS)))
        self.assertEqual(timing["measured_preview_span_ms"], 180.)
        self.assertIsNone(timing["hardware_end_to_end_latency_ms"])

    def test_real_future_does_not_arrive_early(self):
        b = HumanBuffer(self.a.skeleton_id)
        packets=self.packets(10)
        for p in packets: b.push(p)
        with self.assertRaises(NotReady): b.window(8*DT_NS, "replay_monotonic")
        b.window(9*DT_NS,"replay_monotonic")
        with self.assertRaises(NotReady): b.window(9*DT_NS,"other_clock")
        with self.assertRaises(NotReady): b.window(9*DT_NS+3_000_000_001,"replay_monotonic")

    def test_interpolation_records_right_raw_support(self):
        packets = self.packets(20)
        b = HumanBuffer(self.a.skeleton_id)
        for i,p in enumerate(packets):
            p["capture_ns"] = i*12_000_000;p["arrival_ns"] = p["capture_ns"]+50_000_000
            b.push(p)
        w,t=b.window(packets[-1]["arrival_ns"],"replay_monotonic")
        self.assertEqual(w.shape,(10,168))
        self.assertTrue(any(s>t for s,t in zip(t["raw_support_end_ns"],t["slot_capture_ns"])))
        self.assertTrue(max(t["raw_support_end_ns"]) <= packets[-1]["capture_ns"])

    def test_schema_and_reset_boundaries(self):
        for change in [dict(kind="legacy_smpl"),dict(valid=[True]*23),dict(calibration_id=""),
                       dict(skeleton_id="different"),dict(capture_ns=-1),dict(sequence=1.0)]:
            b=HumanBuffer(self.a.skeleton_id);p=self.packets(1)[0];p.update(change)
            with self.assertRaises(ValueError): b.push(p)
        b=HumanBuffer(self.a.skeleton_id);p=self.packets(1)[0];b.push(p)
        with self.assertRaises(ValueError): b.push(p)
        p=self.packets(2)[1];p["calibration_id"]="new"
        with self.assertRaises(ValueError): b.push(p)
        b.reset();b.push(p)

    def test_resampling_phase_survives_ring_eviction(self):
        b=HumanBuffer(self.a.skeleton_id)
        p=self.packets(1)[0]
        for i in range(520):
            packet=dict(p,sequence=i,capture_ns=i*12_000_000,arrival_ns=i*12_000_000)
            b.push(packet)
        _,t=b.window(519*12_000_000,"replay_monotonic")
        self.assertEqual(t["slot_capture_ns"][0]%DT_NS,0)

    def test_nonfinite_and_invalid_quaternion(self):
        for value in (float("nan"),float("inf")):
            p=self.packets(1)[0];p["positions"][3][2]=value
            with self.assertRaises(ValueError): HumanBuffer(self.a.skeleton_id).push(p)
        p=self.packets(1)[0];p["quaternions"][3]=[0,0,0,0]
        with self.assertRaises(ValueError): HumanBuffer(self.a.skeleton_id).push(p)


class RealPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from .runtime import Runtime
        cls.r=Runtime()

    def setUp(self):
        self.r.reset()

    def fill(self):
        from .__main__ import demo_stream,adapter
        for p in demo_stream(adapter(self.r.root), 10): self.r.push(p)

    def test_real_model_native_runner_parity_and_head(self):
        from deploy.sim2sim.policy_node import PolicyRunner
        from .runtime import synthetic_state
        self.fill();now=9*DT_NS
        out=self.r.state(synthetic_state(self.r,0,now),now,"replay_monotonic")
        window=self.r.runner.last_ref.copy()
        class Ref:
            def window(self,*_): return window
            def reset(self): pass
        native=PolicyRunner(self.r.root/"assets/morphmimic_policy","x2",Ref(),"human","full")
        target=native.step(0,np.array([1.,0,0,0]),np.zeros(3),native.robot.default_joint_pos,np.zeros(31))
        np.testing.assert_array_equal(out["q_target_rad"],target)
        self.assertFalse(out["publish_allowed"])
        np.testing.assert_array_equal(target[-2:], native.robot.default_joint_pos[-2:])

    def test_joint_name_reordering(self):
        from .runtime import synthetic_state
        self.fill();p=synthetic_state(self.r,0,9*DT_NS)
        first=self.r.state(p,9*DT_NS,"replay_monotonic")
        self.r.reset();self.fill()
        p["joint_names"].reverse();p["q_rad"].reverse();p["dq_rad_s"].reverse()
        second=self.r.state(p,9*DT_NS,"replay_monotonic")
        np.testing.assert_array_equal(first["q_target_rad"],second["q_target_rad"])
        p["sequence"]+=1;p["joint_names"][0]="wrong_joint"
        with self.assertRaises(ValueError): self.r.state(p,9*DT_NS,"replay_monotonic")

    def test_single_scale_and_wrist_sign(self):
        r=self.r.robot;a=np.zeros(29);a[19]=1
        delta=r.targets(a)-r.default_joint_pos
        i=r.native_joint_names.index("left_wrist_yaw_joint")
        self.assertAlmostEqual(delta[i],-.25)
        self.assertEqual(np.count_nonzero(delta),1)

    def test_torso_gyro_reconstruction_finite_difference(self):
        r=self.r.robot;q=r.default_joint_pos.copy();dq=np.zeros(31)
        q[12:15]=[.3,-.2,.1];dq[12:15]=[.4,.5,-.3]
        root=axis_angle(np.array([.1,.2,-.3]));gyro_root=np.array([.3,-.1,.2])
        ext=axis_angle(np.array([.2,-.1,.15]))
        dt=1e-7
        torso=mul(root,r._tree_quats(q)[3]);sensor=mul(torso,ext)
        root_next=mul(root,axis_angle(gyro_root*dt))
        sensor_next=mul(mul(root_next,r._tree_quats(q+dq*dt)[3]),ext)
        relative=mul(inv(sensor),sensor_next)
        sensor_gyro=2*relative[1:]/dt
        got_q,got_g=pelvis_from_sensor(r,q,dq,sensor,sensor_gyro,"torso_link",ext)
        np.testing.assert_allclose(got_q,root,atol=1e-10)
        np.testing.assert_allclose(got_g,gyro_root,atol=1e-7)

    def test_no_target_before_full_window_and_no_duplicate_state(self):
        from .runtime import synthetic_state
        self.assertFalse(self.r.handle(synthetic_state(self.r,0,0),0,"replay_monotonic")["ready"])
        self.fill();p=synthetic_state(self.r,0,9*DT_NS)
        self.r.state(p,9*DT_NS,"replay_monotonic")
        with self.assertRaises(ValueError):self.r.state(p,9*DT_NS,"replay_monotonic")


class FeedbackTests(unittest.TestCase):
    def test_readonly_feedback_named_assembly_and_ages(self):
        from .shadow_reader import Feedback
        names=[f"j{i}" for i in range(31)]
        f=Feedback(names,"torso_link",[1,0,0,0])
        self.assertIsNone(f.snapshot(0,100))
        for group,indices in zip(("leg","waist","arm","head"),(range(12),range(12,15),range(15,29),range(29,31))):
            f.joints(group,S(joints=[S(name=names[i],position=i,velocity=-i) for i in reversed(indices)]),50)
        f.inertial(S(orientation=S(w=1,x=0,y=0,z=0),angular_velocity=S(x=1,y=2,z=3)),40)
        p=f.snapshot(3,100)
        self.assertEqual(p["q_rad"],list(range(31)))
        self.assertEqual(p["imu_gyro_rad_s"],[1,2,3])
        self.assertEqual(p["source_receive_ages_s"]["imu"],60e-9)

    def test_feedback_duplicate_joint_refused(self):
        from .shadow_reader import Feedback
        f=Feedback(["j"],"pelvis",[1,0,0,0])
        with self.assertRaises(ValueError):f.joints("leg",S(joints=[S(name="j",position=0,velocity=0)]*2),0)




class LatestFgpAndAsyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = artifacts.activate()
        cls.cal_path = artifacts.PACKAGE / "config" / "clotho_fgp_yup.json"
        cls.cal = Calibration(**json.loads(cls.cal_path.read_text()))
        cls.a = SmplAdapter(cls.root / "upstream/deploy/vr_teleop/human_skeleton.json", cls.cal)

    def test_matrix_to_quat_parity_and_so3_rejection(self):
        from .core import matrix_to_quat
        from .fgp_bridge import quat_to_matrix

        rng = np.random.default_rng(20261112)
        aa = rng.normal(scale=0.6, size=(24, 3))
        aa[0] = [0.0, np.pi, 0.0]  # include 180-deg boundary
        q_aa = axis_angle(aa)
        mats = quat_to_matrix(q_aa)
        q_mat = matrix_to_quat(mats)
        dots = np.abs(np.sum(q_aa * q_mat, axis=-1))
        np.testing.assert_allclose(dots, np.ones(24), atol=1e-10)

        p1, g1 = self.a.convert(aa, [0.1, -0.2, 0.3])
        p2, g2 = self.a.convert_matrices(mats, [0.1, -0.2, 0.3])
        np.testing.assert_allclose(p1, p2, atol=1e-10)
        np.testing.assert_allclose(np.abs(np.sum(g1 * g2, axis=-1)), np.ones(24), atol=1e-10)

        bad_reflect = mats.copy()
        bad_reflect[3] = np.diag([1.0, 1.0, -1.0])
        with self.assertRaises(ValueError):
            matrix_to_quat(bad_reflect)

        bad_scale = mats.copy()
        bad_scale[5] *= 1.05
        with self.assertRaises(ValueError):
            matrix_to_quat(bad_scale)

    def test_async_30hz_to_50hz_and_dropout_hold_recovery(self):
        dt_30_ns = 33_333_333  # ~30 Hz BLE garment stream
        b = HumanBuffer(self.a.skeleton_id, max_age_ns=120_000_000, max_interp_gap_ns=100_000_000)
        ident = np.broadcast_to(np.eye(3), (24, 3, 3)).copy()

        # Stream 10 frames at 30 Hz (0..300 ms), enough to cover 180 ms preview
        t_ns = 0
        seq = 0
        for _ in range(10):
            pkt = self.a.packet_from_matrices(
                ident, [0.0, 0.0, 0.0],
                sequence=seq, capture_ns=t_ns, capture_clock="fgp_ble",
                arrival_ns=t_ns + 5_000_000, arrival_clock="replay_monotonic",
            )
            b.push(pkt)
            seq += 1
            t_ns += dt_30_ns

        last_arr = (seq - 1) * dt_30_ns + 5_000_000
        w, timing = b.window(last_arr, "replay_monotonic")
        self.assertEqual(w.shape, (10, 168))
        self.assertAlmostEqual(timing["max_source_gap_ms"], dt_30_ns / 1e6, places=3)
        self.assertLessEqual(timing["max_interp_gap_ms"], 100.0)

        # Advance robot clock past available future -> NotReady (zero extrapolation)
        with self.assertRaises(NotReady):
            b.window(last_arr + 200_000_000, "replay_monotonic")

        # Inject a 160 ms packet stall (> 100 ms max_interp_gap_ns)
        stall_cap = (seq - 1) * dt_30_ns + 160_000_000
        b.push(self.a.packet_from_matrices(
            ident, [0.0, 0.0, 0.0],
            sequence=seq, capture_ns=stall_cap, capture_clock="fgp_ble",
            arrival_ns=stall_cap + 5_000_000, arrival_clock="replay_monotonic",
        ))
        seq += 1
        # Slots spanning the 160 ms gap must raise NotReady
        with self.assertRaises(NotReady):
            b.window(stall_cap + 5_000_000, "replay_monotonic")

        # Once post-stall 30 Hz stream refills >180 ms cleanly, HumanBuffer recovers
        for k in range(1, 8):
            cap = stall_cap + k * dt_30_ns
            b.push(self.a.packet_from_matrices(
                ident, [0.0, 0.0, 0.0],
                sequence=seq, capture_ns=cap, capture_clock="fgp_ble",
                arrival_ns=cap + 5_000_000, arrival_clock="replay_monotonic",
            ))
            seq += 1
        rec_now = stall_cap + 7 * dt_30_ns + 5_000_000
        w_rec, t_rec = b.window(rec_now, "replay_monotonic")
        self.assertEqual(w_rec.shape, (10, 168))
        self.assertLessEqual(t_rec["max_source_gap_ms"], 40.0)

    def test_full_vs_five_human_points_and_synth_bridge(self):
        import tempfile
        from .fgp_bridge import convert_fgp_npz, generate_synth_smpl_npz
        from .runtime import Runtime, synthetic_state

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            npz_path = tmp_dir / "wave.npz"
            jsonl_path = tmp_dir / "wave.jsonl"
            generate_synth_smpl_npz("turn_and_wave", 80, npz_path)
            summary = convert_fgp_npz(npz_path, self.cal_path, jsonl_path, align_initial_ground=True)
            self.assertEqual(summary["frames"], 80)

            packets = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
            r_full = Runtime(self.root, human_points="full")
            r_five = Runtime(self.root, human_points="five")
            diffs = []
            for idx, pkt in enumerate(packets):
                r_full.push(pkt)
                r_five.push(pkt)
                if idx >= 9:
                    now = pkt["arrival_ns"]
                    s_full = r_full.state(synthetic_state(r_full, idx - 9, now), now, "replay_monotonic")
                    s_five = r_five.state(synthetic_state(r_five, idx - 9, now), now, "replay_monotonic")
                    q_full = np.asarray(s_full["q_target_rad"])
                    q_five = np.asarray(s_five["q_target_rad"])
                    np.testing.assert_array_equal(q_full[-2:], r_full.robot.default_joint_pos[-2:])
                    np.testing.assert_array_equal(q_five[-2:], r_five.robot.default_joint_pos[-2:])
                    diffs.append(float(np.max(np.abs(q_full - q_five))))
            self.assertGreater(max(diffs), 1e-2)


if __name__ == "__main__":
    unittest.main()
