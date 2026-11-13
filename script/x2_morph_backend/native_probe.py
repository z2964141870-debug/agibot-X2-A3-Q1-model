"""Bounded SDK writer tests on private ROS topics, or real-feedback observation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

from script.x2_morph_human.artifacts import MODEL_SHA, home
from script.x2_morph_human.client import Client
from script.x2_morph_human.__main__ import adapter, demo_stream
from .contract import Contract, GROUPS
from .sensor_profile import validate


def endpoint():
    with socket.socket() as s:
        s.bind(("127.0.0.1",0))
        return f"tcp://127.0.0.1:{s.getsockname()[1]}"


def finish(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try: process.wait(timeout=5.)
        except subprocess.TimeoutExpired:
            process.terminate(); process.wait(timeout=5.)


def status(output):
    try: return json.loads((output/"status.json").read_text())
    except FileNotFoundError: return {}


class PolicyFixture:
    def __init__(self, address, contract):
        self.address, self.contract = address, contract
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.run)
        self.thread.start()
    def run(self):
        import zmq
        context=zmq.Context(); server=context.socket(zmq.REP)
        server.setsockopt(zmq.LINGER,0); server.setsockopt(zmq.RCVTIMEO,50)
        server.bind(self.address)
        try:
            while not self.done.is_set():
                try: row=server.recv_json()
                except zmq.Again: continue
                server.send_json(dict(kind="x2_morph_advisory_target",version=1,ready=True,
                    publish_allowed=False,policy_sha256=MODEL_SHA,robot_profile="morphmimic_v01_x2_ultra",
                    state_sequence=row["sequence"],joint_names=list(self.contract.names),
                    q_target_rad=list(self.contract.default),canonical_action=[0.]*29))
        finally:
            server.close(); context.term()
    def close(self): self.done.set(); self.thread.join(timeout=2.)


def isolated(args, root, out, cfg):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from aimdk_msgs.msg import JointStateArray, JointCommandArray
    from sensor_msgs.msg import Imu
    c=Contract.load(root/"assets/morphmimic_policy/x2.json")
    rclpy.init()
    summaries=[]
    try:
        for case in ("normal_return", "policy_loss", "feedback_loss"):
            run_id="t"+uuid.uuid4().hex[:10]
            prefix="/x2_morph_test/"+run_id
            node=Node("morph_fixture_"+run_id,enable_rosout=False,start_parameter_services=False)
            states={g:node.create_publisher(JointStateArray,prefix+"/"+g+"/state",qos_profile_sensor_data) for g in GROUPS}
            fake_mc={g:node.create_publisher(JointCommandArray,prefix+"/"+g+"/command",qos_profile_sensor_data) for g in GROUPS}
            imu=node.create_publisher(Imu,prefix+"/imu",qos_profile_sensor_data)
            received=[]; subscriptions=[]
            for group in GROUPS:
                subscriptions.append(node.create_subscription(JointCommandArray,prefix+"/"+group+"/command",
                    lambda msg,g=group:received.append((time.monotonic(),g,[(j.name,j.position,j.velocity,j.stiffness,j.damping) for j in msg.joints])),qos_profile_sensor_data))
            address=endpoint(); policy=PolicyFixture(address,c)
            case_out=out/case
            errors=(out/(case+".stderr.log")).open("x")
            process=subprocess.Popen([str(args.binary),"--profile",str(args.profile),"--output",str(case_out),
                "--endpoint",address,"--mode","isolated","--prefix",prefix,"--imu-topic",prefix+"/imu",
                "--run-id",run_id,"--seconds","25"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=errors,text=True)
            send_mc=True; send_feedback=True
            sent_after=0.
            def pump(seconds):
                nonlocal sent_after
                end=time.monotonic()+seconds
                while time.monotonic()<end:
                    now=time.monotonic()
                    if now-sent_after>=.01:
                        sent_after=now
                        for g,(a,b) in GROUPS.items():
                            if send_feedback:
                                m=JointStateArray()
                                # Instantiate the generated element type without
                                # hardcoding a second SDK schema.
                                from aimdk_msgs.msg import JointState, JointCommand
                                for i in range(a,b):
                                    j=JointState(); j.name=c.names[i]; j.position=c.default[i]; j.velocity=0.
                                    m.joints.append(j)
                                states[g].publish(m)
                            if send_mc:
                                from aimdk_msgs.msg import JointCommand
                                m=JointCommandArray()
                                for i in range(a,b):
                                    j=JointCommand(); j.name=c.names[i]; j.position=c.default[i]
                                    j.stiffness=c.kp[i]; j.damping=c.kd[i]; m.joints.append(j)
                                fake_mc[g].publish(m)
                        if send_feedback:
                            m=Imu(); m.header.frame_id="base_link"; m.orientation.w=1.; imu.publish(m)
                    rclpy.spin_once(node,timeout_sec=.001)
                    if process.poll() is not None: break
            def wait(predicate,seconds=8.):
                deadline=time.monotonic()+seconds
                while time.monotonic()<deadline:
                    row=status(case_out)
                    if predicate(row): return row
                    pump(.03)
                raise AssertionError(case+": "+json.dumps(status(case_out)))
            def command(op):
                process.stdin.write(json.dumps(dict(run_id=run_id,op=op,mc_stopped=True))+"\n")
                process.stdin.flush()
            try:
                milestones={}
                wait(lambda s:s.get("ready_target") and s.get("mc_capture_complete"))
                command("capture_return"); wait(lambda s:s.get("mc_capture_frozen"))
                send_mc=False; pump(.45); received.clear()
                command("arm"); wait(lambda s:s.get("stage")=="CONTROL")
                pump(.3)
                if status(case_out).get("stage")!="CONTROL": raise AssertionError("writer mistook own output for external MC")
                milestones["active"]=status(case_out)
                if set(g for _,g,_ in received)!=set(GROUPS): raise AssertionError("four SDK command groups not observed")
                # There is no second scale, sign flip, default-pose addition, or
                # head bypass. Fixture target equals start pose, so ramp is exact.
                for _,g,joints in received:
                    a,b=GROUPS[g]
                    if len(joints)!=b-a: raise AssertionError("SDK group width changed")
                    for i,j in zip(range(a,b),joints):
                        if j!=(c.names[i],c.default[i],0.,c.kp[i],c.kd[i]):
                            raise AssertionError("SDK writer changed native target or gains")
                before_fault=len(received)
                if case=="normal_return":
                    command("return"); milestones["returned"]=wait(lambda s:s.get("stage")=="HOLD_FOR_MC",4.)
                else:
                    if case=="policy_loss": policy.close()
                    else: send_feedback=False
                    milestones["fault"]=wait(lambda s:s.get("stage")=="FAULT_DAMPING",2.)
                    pump(.2)
                    damping=received[before_fault:]
                    damping_groups={g for _,g,joints in damping
                                    if all(j[3]==0. and j[4]==cfg["fault_kd"][i]
                                           for i,j in zip(range(*GROUPS[g]),joints))}
                    if damping_groups!=set(GROUPS):
                        raise AssertionError("fault did not produce zero-stiffness damping")
                    command("arm"); pump(.2)
                    if status(case_out).get("stage")!="FAULT_DAMPING": raise AssertionError("fault rearmed")
                    milestones["rearm_rejected"]=status(case_out)
                send_mc=True; pump(.3)
                process.wait(timeout=3.)
                final=status(case_out)
                if process.returncode or final["stage"]!="RELEASED" or final["hardware_commands"]!=0:
                    raise AssertionError("private bus handback failed")
                summaries.append(dict(case=case,status="PASSED",sdk_publish_calls=final["sdk_publish_calls"],
                    writer_ticks=final["writer_ticks"],command_messages_observed=len(received),
                    milestones=milestones,exact_native_target_and_gains_verified=True,
                    fault_vendor_damping_groups=list(GROUPS) if case!="normal_return" else [],
                    final=final,fixture="SYNTHETIC_POLICY_AND_PRIVATE_ROS_FEEDBACK",real_mc_calls=0,
                    private_topic_prefix=prefix,hardware_commands=0))
            finally:
                policy.close(); finish(process); errors.close()
                if process.stdin: process.stdin.close()
                if process.stdout: process.stdout.close()
                node.destroy_node()
    finally: rclpy.shutdown()
    return dict(status="ISOLATED_SDK_WRITER_AND_HANDOFF_PASSED",cases=summaries,hardware_commands=0,mc_mutations=0)


def real_feedback(args,root,out,cfg):
    address=endpoint(); env=dict(os.environ,X2_MORPH_HOME=str(root),PYTHONDONTWRITEBYTECODE="1")
    policy_err=(out/"policy.stderr.log").open("x")
    policy=subprocess.Popen([sys.executable,"-B","-m","script.x2_morph_human","serve",
                             "--endpoint",address,"--max-freshness-ms","120","--max-interp-gap-ms","100"],
                            stdout=subprocess.PIPE,stderr=policy_err,text=True,env=env)
    native=None; producer=None
    try:
        import select
        if not select.select([policy.stdout],[],[],30.)[0]: raise TimeoutError("owned policy startup timed out")
        startup=json.loads(policy.stdout.readline())
        if startup.get("status")!="ADVISORY_ONLY": raise ValueError("unexpected policy startup")
        producer=Client(address)
        packets=list(demo_stream(adapter(root),450))
        errors=(out/"native.stderr.log").open("x")
        run_id="o"+uuid.uuid4().hex[:10]
        native_out=out/"native"
        native=subprocess.Popen([str(args.binary),"--profile",str(args.profile),"--output",str(native_out),
            "--endpoint",address,"--mode","observe","--run-id",run_id,"--seconds","6"],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors,text=True,env=env)
        sequence=0
        while native.poll() is None:
            started=time.monotonic()
            packet=dict(packets[sequence],capture_ns=time.monotonic_ns(),capture_clock="synthetic_smpl_live_feedback_probe")
            response=producer.request(packet)
            if response.get("accepted_sequence")!=sequence: raise ValueError("SMPL acknowledgement mismatch")
            sequence+=1
            if sequence>=len(packets): raise TimeoutError("bounded SMPL source exhausted")
            time.sleep(max(0.,.02-(time.monotonic()-started)))
        if native.returncode: raise RuntimeError("read-only native controller failed")
        final=status(native_out)
        rows=[json.loads(line) for line in (native_out/"native.jsonl").read_text().splitlines()]
        samples=[r for r in rows if r.get("kind")=="policy_sample"]
        accepted=[r for r in samples if r["accepted"]]
        if final["sdk_publish_calls"] or final["hardware_commands"] or not accepted:
            raise ValueError("no valid real-feedback inference or observe writer was not muted")
        rpc=sorted(r["rpc_ms"] for r in accepted)
        return dict(status="REAL_FEEDBACK_TO_REAL_MORPH_NO_PUBLISH_PASSED",native=final,
            human_fixture="SYNTHETIC_SMPL_ASSUMED_STABLE",robot_feedback="LIVE_X2_ROS",
            human_frames=sequence,policy_requests=len(samples),accepted=len(accepted),
            accepted_rpc_ms=dict(median=rpc[len(rpc)//2],p95=rpc[min(len(rpc)-1,int(len(rpc)*.95))],maximum=max(rpc)),
            sensor_basis=validate(cfg),measured_physical_calibration=False,
            hardware_commands=0,mc_mutations=0,hardware_end_to_end_ms=None)
    finally:
        if producer is not None: producer.close()
        if native is not None:
            finish(native)
            if native.stdin: native.stdin.close()
            if native.stdout: native.stdout.close()
            errors.close()
        finish(policy)
        if policy.stdout: policy.stdout.close()
        policy_err.close()


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("mode",choices=("isolated","real-feedback"))
    p.add_argument("--binary",type=Path,required=True)
    p.add_argument("--profile",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); a.binary=a.binary.resolve(); a.profile=a.profile.resolve()
    cfg=json.loads(a.profile.read_text()); validate(cfg)
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    result=(isolated if a.mode=="isolated" else real_feedback)(a,home(),out,cfg)
    (out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
