"""Terminal 3 launcher. Default is real feedback with a permanently muted writer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import subprocess
import time
import uuid

from .mc_handoff import Handoff, HandoffError, NativeController, RosMc
from .sensor_profile import validate


class StableReadiness:
    """Do not admit a transient target or hide rejections between status polls."""
    def __init__(self, duration=1.0):
        self.duration = duration
        self.since = None
        self.last_rejected = None
        self.first_requests = None

    def update(self, status, now):
        rejected = status.get("rejected", 0)
        healthy = (status.get("stage") == "STANDBY" and status.get("ready_target")
                   and status.get("mc_capture_complete") and not status.get("transport_failed"))
        changed = self.last_rejected is not None and rejected != self.last_rejected
        self.last_rejected = rejected
        if not healthy or changed:
            self.since = self.first_requests = None
            return False
        if self.since is None:
            self.since, self.first_requests = now, status.get("requests", 0)
        return now - self.since >= self.duration and status.get("requests", 0) > self.first_requests


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--binary",type=Path,required=True)
    p.add_argument("--profile",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--endpoint",default="tcp://127.0.0.1:5765")
    p.add_argument("--seconds",type=float,default=10.)
    p.add_argument("--control",action="store_true")
    p.add_argument("--confirm-live",choices=("X2_ULTRA_MORPH_CONTROL",))
    a=p.parse_args()
    if not 0<a.seconds<=300 or (a.control and not a.confirm_live):
        p.error("0..300 seconds and explicit confirmation for --control required")
    cfg=json.loads(a.profile.read_text()); basis=validate(cfg)
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    run_id="r"+uuid.uuid4().hex[:12]
    native_out=out/"native"
    argv=[str(a.binary.resolve()),"--profile",str(a.profile.resolve()),"--output",str(native_out),
          "--endpoint",a.endpoint,"--run-id",run_id,"--seconds",str(a.seconds+90. if a.control else a.seconds),
          "--mode","hardware" if a.control else "observe"]
    if a.control: argv += ["--confirm-live",a.confirm_live]
    mc=handoff=None
    keep_writer=False
    with (out/"stdout.log").open("x") as stdout,(out/"stderr.log").open("x") as stderr:
        process=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=stdout,stderr=stderr,text=True)
        controller=NativeController(process,native_out,run_id)
        try:
            if a.control:
                # ROS setup can take longer than a target's lifetime. Finish
                # initialization before observing the stable readiness window.
                mc=RosMc(); handoff=Handoff(mc,controller)
            readiness = StableReadiness()
            deadline=time.monotonic()+20.
            while time.monotonic()<deadline:
                if process.poll() is not None: raise HandoffError("native backend exited during startup")
                try: status=controller.status()
                except FileNotFoundError: time.sleep(.05); continue
                if not a.control or readiness.update(status, time.monotonic()): break
                time.sleep(.05)
            else:
                status = controller.status()
                (out/"startup_not_ready.json").write_text(json.dumps(status, indent=2)+"\n")
                raise HandoffError("No continuous 1s ready window within 20s: ready_target="+str(status.get("ready_target"))+
                                   ", mc_capture_complete="+str(status.get("mc_capture_complete"))+
                                   ", transport_failed="+str(status.get("transport_failed"))+
                                   "; inspect "+str(native_out/"native.jsonl"))
            if a.control:
                handoff.acquire()
                print("CONTROL_ACTIVE: Morph owns command bus; timed session running", flush=True)
                deadline=time.monotonic()+a.seconds
                while time.monotonic()<deadline:
                    status=controller.status()
                    if status["stage"]!="CONTROL": raise HandoffError("control left active state: "+status["stage"])
                    time.sleep(.05)
                print("RETURNING_TO_MC: completing normal handoff", flush=True)
                handoff.release()
                print("MC_STAND_CONFIRMED: command bus returned", flush=True)
            process.wait(timeout=a.seconds+10 if not a.control else 5.)
            status=controller.status()
            if process.returncode: raise HandoffError("native backend exited unsuccessfully")
            (out/"summary.json").write_text(json.dumps(dict(status="SESSION_COMPLETED",sensor_basis=basis,
                native=status,mc_events=handoff.events if handoff else [],
                hardware_commands=status["hardware_commands"],mc_mutations=bool(a.control)),indent=2)+"\n")
            print(json.dumps(status))
        except (Exception,KeyboardInterrupt):
            if a.control and handoff is not None and process.poll() is None:
                try:
                    current=controller.status()
                    keep_writer=current["stage"] not in ("STANDBY","RELEASED")
                except Exception:
                    keep_writer=True
                if keep_writer:
                    try: controller.command("fault")
                    except (BrokenPipeError,OSError): pass
            if handoff is not None and not keep_writer:
                handoff.rollback_unarmed()
            if keep_writer:
                # Do not kill the only damping writer or pretend MC has resumed.
                # The terminal remains attached until a non-self MC publisher
                # actually releases the native writer. A field operator owns
                # fault recovery; no automatic stand-up follows a fault.
                print("CONTROL RECOVERY REQUIRED: writer remains alive; restore MC explicitly and inspect bus ownership.",flush=True)
                while process.poll() is None:
                    time.sleep(.2)
            raise
        finally:
            if process.poll() is None and not keep_writer:
                process.send_signal(signal.SIGINT)
                process.wait(timeout=5.)
            if process.stdin: process.stdin.close()
            if mc is not None: mc.close()


if __name__=="__main__": main()
