"""Optional ROS2 READ-ONLY feedback -> local advisory inference -> JSONL.

Run with the robot's ROS Python, not necessarily the policy's Python3.11. No command
message type, publisher, service client, MC handoff or process-control operation exists.
Mac only validates the pure message assembler; actual ROS/AimDK ABI remains field work.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .client import Client


class Feedback:
    def __init__(self, names, imu_body, body_from_sensor):
        self.names, self.imu_body, self.extrinsic = names, imu_body, body_from_sensor
        self.groups, self.imu = {}, None

    def joints(self, group, msg, now):
        rows = {j.name: (float(j.position), float(j.velocity)) for j in msg.joints}
        if len(rows) != len(msg.joints) or not set(rows).issubset(self.names):
            raise ValueError("duplicate/unexpected X2 feedback joint")
        self.groups[group] = (rows, now)

    def inertial(self, msg, now):
        q, g = msg.orientation, msg.angular_velocity
        self.imu = ([float(q.w),float(q.x),float(q.y),float(q.z)],
                    [float(g.x),float(g.y),float(g.z)], now)

    def snapshot(self, seq, now):
        if self.imu is None or set(self.groups) != {"leg","waist","arm","head"}:
            return None
        joints={}
        for values,_ in self.groups.values():
            if joints.keys() & values.keys():raise ValueError("overlapping joint topics")
            joints.update(values)
        if set(joints) != set(self.names):raise ValueError("feedback does not cover all exported X2 joints")
        quat,gyro,imu_time=self.imu
        return dict(kind="x2_measurement",version=1,sequence=seq,joint_names=self.names,
                    q_rad=[joints[n][0] for n in self.names],dq_rad_s=[joints[n][1] for n in self.names],
                    imu_quat_wxyz=quat,imu_gyro_rad_s=gyro,imu_body=self.imu_body,
                    body_from_sensor_wxyz=self.extrinsic,
                    source_receive_ages_s={**{k:(now-t)/1e9 for k,(_,t) in self.groups.items()},
                                           "imu":(now-imu_time)/1e9})


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--profile",type=Path,required=True)
    p.add_argument("--x2-json",type=Path,required=True)
    p.add_argument("--endpoint",default="tcp://127.0.0.1:5765")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seconds",type=float,default=60.)
    a=p.parse_args()
    cfg=json.loads(a.profile.read_text())
    if not cfg.get("measured_sensor_calibration_id") or cfg["measured_sensor_calibration_id"].startswith("REPLACE"):
        p.error("measured sensor calibration required; example is not a calibrated robot profile")
    if not 0<a.seconds<=600:p.error("read-only capture seconds must be 0..600")
    names=json.loads(a.x2_json.read_text())["robot"]["native_joint_names"]
    feedback=Feedback(names,cfg["imu_body"],cfg["body_from_sensor_wxyz"])
    # Imported only here: offline Mac commands do not require the ROS2 stack.
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from aimdk_msgs.msg import JointStateArray
    from sensor_msgs.msg import Imu
    rclpy.init();node=Node("x2_morph_readonly_shadow")
    client=Client(a.endpoint);seq=0;start=time.monotonic();subscriptions=[]
    try:
        for group in ("leg","waist","arm","head"):
            def ingest(msg,g=group):feedback.joints(g,msg,time.monotonic_ns())
            subscriptions.append(node.create_subscription(JointStateArray,cfg["joint_state_topics"][group],ingest,qos_profile_sensor_data))
        subscriptions.append(node.create_subscription(Imu,cfg["imu_topic"],lambda msg:feedback.inertial(msg,time.monotonic_ns()),qos_profile_sensor_data))
        with open(a.output,"x") as f:
            next_tick=time.monotonic()+.02
            while rclpy.ok() and time.monotonic()-start<a.seconds:
                rclpy.spin_once(node,timeout_sec=max(0.,next_tick-time.monotonic()))
                if time.monotonic()<next_tick:continue
                # At most 50Hz, never one inference per high-rate feedback callback.
                next_tick=max(next_tick+.02,time.monotonic()+.001)
                now=time.monotonic_ns();state=feedback.snapshot(seq,now)
                if state is None:continue
                t=time.perf_counter_ns();reply=client.request(state)
                f.write(json.dumps(dict(state=state,advisory=reply,local_rpc_ms=(time.perf_counter_ns()-t)/1e6))+"\n")
                seq+=1
    finally:
        client.close();node.destroy_node();rclpy.shutdown()


if __name__=="__main__":main()
