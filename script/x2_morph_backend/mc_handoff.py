"""Explicit Morph/MC ownership protocol, with typed SDK calls and fake transports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import urllib.request


class HandoffError(RuntimeError):
    pass


def stopped(runtime):
    # EM retains the old PID after a successful stop; PID presence is not liveness.
    return isinstance(runtime, dict) and runtime.get("status", "").lower().startswith("stopped")


class Handoff:
    def __init__(self, mc, controller, *, clock=time.monotonic, sleep=time.sleep):
        self.mc, self.controller, self.clock, self.sleep = mc, controller, clock, sleep
        self.stop_attempted = False
        self.damping_requested = False
        self.acquired = False
        self.events = []

    def wait(self, predicate, seconds, reason):
        deadline = self.clock()+seconds
        while self.clock() < deadline:
            value = predicate()
            if value:
                return value
            self.sleep(.05)
        raise HandoffError(reason)

    def acquire(self):
        status = self.controller.status()
        if (status.get("stage") != "STANDBY" or not status.get("ready_target")
                or not status.get("mc_capture_complete") or status.get("transport_failed")):
            raise HandoffError("controller must have fresh targets and a complete captured MC command")
        runtime = self.mc.runtime()
        if not runtime.get("status", "").startswith("Running(") or self.mc.mode() != "STAND_DEFAULT":
            raise HandoffError("MC must be running in STAND_DEFAULT before ownership transfer")
        self.controller.command("capture_return")
        self.wait(lambda: self.controller.status().get("mc_capture_frozen"), 2.,
                  "standing MC return command was not frozen")
        self.damping_requested = True
        self.mc.set_mode("DAMPING_DEFAULT")
        self.wait(lambda: self.mc.mode() == "DAMPING_DEFAULT", 3., "MC damping not confirmed")
        self.events.append("DAMPING_CONFIRMED")
        self.stop_attempted = True
        self.mc.app("stop_app")
        self.wait(lambda: stopped(self.mc.runtime()), 8., "MC process stop not confirmed")
        self.wait(lambda: self.controller.status().get("external_age_s", -1.) >= .3,
                  3., "MC command bus not silent")
        self.controller.command("arm", mc_stopped=True)
        self.wait(lambda: self.controller.status().get("stage") == "CONTROL", 3., "Morph writer did not arm")
        self.acquired = True
        self.events.append("MORPH_OWNS_BUS")

    def release(self):
        status = self.controller.status()
        if status.get("stage") == "FAULT_DAMPING":
            raise HandoffError("fault is latched: keep the damping writer alive; operator recovery required")
        if status.get("stage") == "CONTROL":
            self.controller.command("return")
        self.wait(lambda: self.controller.status().get("stage") == "HOLD_FOR_MC", 6.,
                  "return pose was not reached; MC restart must not race the return")
        self.mc.app("start_app")
        def available_mode():
            try: return self.mc.mode()
            except (HandoffError, OSError, TimeoutError): return None
        mode = self.wait(available_mode, 30., "MC services did not return")
        # As in Sonic, MC can boot into PASSIVE and needs JOINT before STAND.
        # The native writer relinquishes on the first non-self HAL command,
        # before further mode changes; receipt of a service response is not
        # evidence that the robot has reached the requested standing state.
        if mode == "PASSIVE_DEFAULT":
            self.mc.set_mode("JOINT_DEFAULT")
            self.wait(lambda: self.mc.mode() == "JOINT_DEFAULT", 5., "MC JOINT mode not confirmed")
            mode = "JOINT_DEFAULT"
        self.wait(lambda: self.controller.status().get("stage") == "RELEASED", 8.,
                  "MC has not demonstrably retaken the command bus")
        if mode == "JOINT_DEFAULT":
            self.mc.set_mode("STAND_DEFAULT")
        self.wait(lambda: self.mc.mode() == "STAND_DEFAULT", 8., "MC STAND mode not confirmed")
        self.stop_attempted = self.acquired = self.damping_requested = False
        self.events.append("MC_STAND_AND_BUS_CONFIRMED")

    def rollback_unarmed(self):
        if not self.stop_attempted and not self.damping_requested:
            return
        if self.controller.status().get("stage") != "STANDBY":
            raise HandoffError("armed or faulted writer must remain alive until observed MC takeover")
        if self.stop_attempted:
            self.mc.app("start_app")
            self.wait(lambda: self.mc.runtime().get("status", "").startswith("Running("), 15.,
                      "MC restart after failed admission not confirmed")
        mode = self.mc.mode()
        if mode == "PASSIVE_DEFAULT":
            self.mc.set_mode("JOINT_DEFAULT")
            self.wait(lambda: self.mc.mode() == "JOINT_DEFAULT", 5., "rollback JOINT not confirmed")
        self.mc.set_mode("STAND_DEFAULT")
        self.wait(lambda: self.mc.mode() == "STAND_DEFAULT", 8., "rollback STAND not confirmed")
        self.stop_attempted = self.damping_requested = False
        self.events.append("UNARMED_MC_ROLLBACK")


class RosMc:
    """Only this transport performs MC mutations, and only via explicit methods."""
    def __init__(self, base_url="http://10.0.1.40:50080"):
        import rclpy
        from rclpy.node import Node
        from aimdk_msgs.srv import GetMcAction, SetMcAction
        self.rclpy = rclpy
        self.get_type, self.set_type = GetMcAction, SetMcAction
        self.base_url = base_url
        self.owns_context = not rclpy.ok()
        if self.owns_context:
            rclpy.init()
        self.node = Node("x2_morph_mc_handoff", enable_rosout=False, start_parameter_services=False)
        self.get = self.node.create_client(GetMcAction, "/aimdk_5Fmsgs/srv/GetMcAction")
        self.set = self.node.create_client(SetMcAction, "/aimdk_5Fmsgs/srv/SetMcAction")

    def call(self, client, request):
        if not client.wait_for_service(timeout_sec=3.):
            raise HandoffError("MC service unavailable")
        future = client.call_async(request)
        self.rclpy.spin_until_future_complete(self.node, future, timeout_sec=3.)
        if not future.done() or future.result() is None:
            client.remove_pending_request(future)
            raise HandoffError("MC service response timed out")
        return future.result()

    def runtime(self):
        with urllib.request.urlopen(self.base_url+"/json/get_all_apps_info", timeout=3.) as response:
            return json.load(response)["data"]["mc"]

    def mode(self):
        response = self.call(self.get, self.get_type.Request())
        if response.header.code != 0:
            raise HandoffError("GetMcAction rejected")
        return response.info.action_desc

    def set_mode(self, mode):
        if mode not in ("DAMPING_DEFAULT", "JOINT_DEFAULT", "STAND_DEFAULT"):
            raise HandoffError("unsupported MC transition")
        request = self.set_type.Request()
        request.source = "x2_morph_backend"
        request.command.action_desc = mode
        if self.call(self.set, request).response.header.code != 0:
            raise HandoffError("SetMcAction rejected")

    def app(self, action):
        if action not in ("stop_app", "start_app"):
            raise HandoffError("unsupported EM action")
        request = urllib.request.Request(self.base_url+"/json/"+action,
            data=json.dumps({"app_name":"mc"}).encode(),
            headers={"Content-Type":"application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=30.) as response:
            body = response.read()
        # HTTP success is only transport acknowledgement. Handoff separately
        # checks process liveness, wire silence and actual MC mode.
        return body

    def close(self):
        self.node.destroy_node()
        if self.owns_context:
            self.rclpy.shutdown()


class NativeController:
    def __init__(self, process, output, run_id):
        self.process, self.output, self.run_id = process, Path(output), run_id

    def status(self):
        row = json.loads((self.output/"status.json").read_text())
        if row.get("run_id") != self.run_id:
            raise HandoffError("controller identity changed")
        if (self.process.poll() is not None and (self.process.returncode != 0
                or (row.get("mode") == "hardware" and row.get("stage") != "RELEASED"))):
            raise HandoffError("controller exited before bus release")
        return row

    def command(self, op, **fields):
        self.process.stdin.write(json.dumps(dict(run_id=self.run_id, op=op, **fields))+"\n")
        self.process.stdin.flush()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=("plan", "inspect"))
    a = p.parse_args()
    if a.mode == "plan":
        print(json.dumps(dict(sequence=["fresh SMPL/feedback and captured MC command", "STAND -> DAMPING",
            "EM stop -> runtime STOPPED -> bus silence", "explicit native arm -> measured-pose ramp",
            "normal return -> captured MC pose", "EM start -> non-self bus takeover -> writer silent",
            "JOINT -> STAND and GetMcAction confirmation"], faults="latched damping, explicit operator recovery",
            hardware_commands=0, mc_mutations=0), indent=2))
        return
    mc = RosMc()
    try:
        print(json.dumps(dict(runtime=mc.runtime(), mode=mc.mode(),
                              hardware_commands=0, mc_mutations=0), allow_nan=False))
    finally:
        mc.close()


if __name__ == "__main__": main()
