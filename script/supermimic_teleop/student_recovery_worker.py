"""Prime identical saved proprioception, then fork f0 student and privileged H10."""

import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/"q1_five_stage"), str(HERE.parent/"primebot_q1")]
from common import require
from delivery import read
from paused_runtime import assert_owner
from policy_process import MAX_MESSAGE, NativeSession
from student_data import OfflineStudentStream
from student_proprio_review_worker import ProprioReviewSession
from stream_events import EventSpec


class RecoverySession(ProprioReviewSession):
    def __init__(self, args):
        super().__init__(args)
        self.recovery = read(args.recovery_plan)
        self.forwards = 0

    def observation(self, raw):
        torch = self.torch
        from morphmimic.model.rewards_torch import analytic_obs
        from q1_native_review import native_state
        state = torch.tensor(raw, dtype=torch.float32)
        require(state.shape == (79,) and torch.isfinite(state).all().item(), "Invalid fork state")
        converted = native_state(self.robot, state[None], self.spec)
        value = analytic_obs(converted, torch.zeros(29), self.default, converted[:, 240:], self.spec.imu_slot)
        present = self.actor.present
        return converted, torch.cat((torch.where(present, value[:, :29], 0.),
            torch.where(present, value[:, 29:58], 0.), value[:, 58:64], torch.where(present, value[:, 64:], 0.)), -1)

    def prepare(self, request):
        index, arm = request.get("job_index"), request.get("arm")
        require(type(index) is int and 0 <= index < len(self.recovery["jobs"]), "Unregistered physical fork")
        self.job = self.recovery["jobs"][index]
        require(arm in self.job["arm_order"], "Unregistered fork arm")
        self.arm = arm
        source = read(HERE/"student_onpolicy_plan_20261108.json")["jobs"][self.job["clip_id"]]
        self.args.event_kind = source["packet_event"]["kind"]
        result = NativeSession.start(self, dict(clip=self.job["clip"], seed=0, interface="human"))
        self.stream = OfflineStudentStream(self.library.human_pos_w.numpy(), self.library.human_quat_w.numpy(),
            spec=EventSpec(**source["packet_event"]), sha256=source["human"]["sha256"],
            session="teacher-recovery:"+self.job["clip"])
        path = Path(self.recovery["collection_root"])/f'collection_{self.job["clip_id"]:02d}.npz'
        with np.load(path, allow_pickle=False) as saved:
            self.saved = {k:saved[k].copy() for k in ("actual", "canonical_action", "teacher_action", "input_ready")}
        anchor = self.job["anchor"]
        require(anchor < len(self.saved["canonical_action"]) and bool(self.saved["input_ready"][anchor]),
            "Fixed anchor is absent, terminal or unavailable; do not substitute another")
        for frame, raw in enumerate(self.saved["actual"][:anchor]):
            _, obs = self.observation(raw)
            self.history.reset(None, obs) if frame == 0 else self.history.push(obs)
        self.frame = anchor
        self.previous_action = self.torch.tensor(self.saved["canonical_action"][anchor-1:anchor])
        digest = hashlib.sha256(self.history.flat().numpy().tobytes()).hexdigest()
        return dict(result, arm=arm, anchor=anchor, history_sha256=digest,
            priming_model_forwards=0, same_saved_robot_history=True)

    def action(self, request):
        require(time.monotonic_ns() < self.args.deadline_ns-30_000_000_000
            and request["frame"] < self.job["anchor"]+80 and self.forwards < 1280, "Fork inference bound")
        frame, torch = request["frame"], self.torch
        if self.arm == "student_f0":
            reply = super().action(request)
            ready = reply["stream"]["input_ready"]
            self.forwards += int(ready)
        else:
            from morphmimic.constants import ANCHOR_IDX
            from morphmimic.model.human import pack_human_goal
            from morphmimic.model.state import split_state
            require(frame == self.frame, "Sequential teacher fork only")
            converted, obs = self.observation(request["state"])
            self.history.push(obs)
            labels = self.stream.teacher_labels(frame)
            native, started = self.preview.policy.native, time.perf_counter_ns()
            with torch.inference_mode():
                goal = pack_human_goal(torch.tensor(np.array(labels.positions), dtype=torch.float32, device="cuda:0")[None],
                    torch.tensor(np.array(labels.quaternions), dtype=torch.float32, device="cuda:0")[None],
                    split_state(converted).body_quat[:, ANCHOR_IDX].to("cuda:0"))
                action = native(self.history.flat().to("cuda:0"), {"goal_kp":goal}, "human").cpu()
            timing = self.compute_record(started)
            require(torch.isfinite(action).all().item() and action.shape == (1, 29), "Invalid teacher action")
            reply = dict(canonical_action=action[0].tolist(),
                native_action=self.spec.translator.to_native_delta(action)[0].tolist(),
                stream=dict(timing, frame=frame, privileged_teacher=True,
                    teacher_dependency_ns=labels.latest_source_dependency_ns, human_real_future_frames=10))
            self.previous_action, self.frame = action.clone(), frame+1
            self.forwards += 1
        if frame == self.job["anchor"]:
            expected = self.saved["canonical_action" if self.arm == "student_f0" else "teacher_action"][frame]
            require(np.array_equal(np.asarray(request["state"]), self.saved["actual"][frame]),
                "First fork observation is not exactly the saved anchor")
            difference = float(np.max(np.abs(np.asarray(reply["canonical_action"])-expected)))
            require(np.allclose(reply["canonical_action"], expected, atol=2e-5, rtol=2e-5),
                "First student action or independent teacher label did not reproduce")
            reply["anchor_action_max_abs_difference"] = difference
        return dict(reply, arm=self.arm, cumulative_model_forwards=self.forwards,
            training_only_privileged_reference=self.arm == "privileged_teacher_h10")


def serve(args):
    output = sys.stdout
    def reply(value):
        raw = json.dumps(value, allow_nan=False)
        require(len(raw.encode()) < MAX_MESSAGE, "Recovery RPC exceeds original bound")
        output.write(raw+"\n"); output.flush()
    with redirect_stdout(sys.stderr):
        try:
            session = RecoverySession(args)
            reply(dict(ok=True, value=session.ready()))
            while True:
                raw = sys.stdin.buffer.readline(MAX_MESSAGE+1)
                require(raw and raw.endswith(b"\n") and len(raw) <= MAX_MESSAGE, "Invalid fork RPC")
                request = json.loads(raw)
                if request.get("op") == "quit":
                    reply(dict(ok=True, value=dict(closed=True))); return
                require(request.get("op") in ("prepare", "action"), "No fitting or unregistered source operation")
                assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
                reply(dict(ok=True, value=getattr(session, request["op"])(request)))
        except BaseException as error:
            reply(dict(ok=False, error=f"{type(error).__name__}: {error}")); raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("plan", "checkpoint", "frozen", "primebot-checkpoint", "morph-source", "primebot-source",
            "student-checkpoint", "recovery-plan"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--expected-gpu", required=True)
    p.add_argument("--gpu-owner-pid", type=int, required=True)
    p.add_argument("--gpu-owner-ticks", required=True)
    p.add_argument("--deadline-ns", type=int, required=True)
    p.add_argument("--event-kind", default="nominal")
    p.add_argument("--residual-arm", choices=("causal_history",), default="causal_history")
    p.add_argument("--preview-frames", type=int, choices=(0,), default=0)
    p.add_argument("--tail-method", choices=("masked",), default="masked")
    serve(p.parse_args())
