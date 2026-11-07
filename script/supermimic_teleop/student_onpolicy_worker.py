"""The unchanged residual student drives physics; teacher calls only label it."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/"q1_five_stage"), str(HERE.parent/"primebot_q1")]
from common import identity, require, save
from delivery import read
from paused_runtime import assert_owner
from policy_process import MAX_MESSAGE, NativeSession
from student_data import OfflineStudentStream
from stream_events import EventSpec
from student_onpolicy_data import behavior
from student_proprio_review_worker import ProprioReviewSession


class OnPolicySession(ProprioReviewSession):
    def __init__(self, args):
        super().__init__(args)
        self.jobs = read(args.output/"collection_plan.json")["jobs"]
        self.buffers = {"train": [], "validation": []}
        self.finished = False

    def start(self, request):
        clip_id = request.get("clip_id")
        require(type(clip_id) is int and 0 <= clip_id < len(self.jobs) and not self.finished,
            "Unknown owned on-policy source")
        self.clip_id, self.job = clip_id, self.jobs[clip_id]
        require(request["clip"] == self.job["clip"], "On-policy source differs")
        self.args.event_kind = self.job["packet_event"]["kind"]
        result = NativeSession.start(self, request)
        self.stream = OfflineStudentStream(self.library.human_pos_w.numpy(), self.library.human_quat_w.numpy(),
            spec=EventSpec(**self.job["packet_event"]), sha256=self.job["human"]["sha256"],
            session="pure-student-labels:"+self.job["clip"])
        self.previous_action = self.torch.zeros(1, 29)
        return dict(result, behavior="causal_student_only", teacher_behavior_fraction=0.)

    def action(self, request):
        require(time.monotonic_ns() < self.args.deadline_ns-30_000_000_000
            and request["frame"] < self.job["maximum_steps"], "On-policy collection bound reached")
        reply = super().action(request)
        frame, torch = request["frame"], self.torch
        student = np.asarray(reply["canonical_action"], dtype=np.float32)[None]
        teacher = np.zeros((1, 29), dtype=np.float32)
        queried = frame % 2 == 0 and reply["stream"]["input_ready"]
        if queried:
            from morphmimic.constants import ANCHOR_IDX
            from morphmimic.model.human import pack_human_goal
            from morphmimic.model.state import split_state
            from morphmimic.train.normalizers import normalize_history
            from q1_native_review import native_state
            from student_inputs import current_features

            raw = np.asarray(request["state"], dtype=np.float64)
            converted = native_state(self.robot, torch.tensor(raw, dtype=torch.float32)[None], self.spec)
            torso = split_state(converted).body_quat[:, ANCHOR_IDX]
            labels = self.stream.teacher_labels(frame)
            native = self.preview.policy.native
            with torch.inference_mode():
                goal = pack_human_goal(torch.tensor(np.array(labels.positions), dtype=torch.float32, device="cuda:0")[None],
                    torch.tensor(np.array(labels.quaternions), dtype=torch.float32, device="cuda:0")[None], torso.to("cuda:0"))
                teacher = native(self.history.flat().to("cuda:0"), {"goal_kp": goal}, "human").cpu().numpy()
                history = normalize_history(native.norms["obs_norm"], self.history.flat().to("cuda:0")).cpu().numpy()
            require(np.array_equal(behavior(student, teacher), student), "Teacher changed actual behavior")
            current = self.stream.student(frame, mode="f0_history")
            self.buffers[self.job["split"]].append(dict(f0=current_features(current, torso[0].numpy()).astype(np.float32),
                history=history[0].copy(), teacher_action=teacher[0].copy(), student_action=student[0].copy(),
                actual_state=raw.copy(), frame=np.int64(frame), clip_id=np.int64(self.clip_id),
                teacher_dependency_ns=np.int64(labels.latest_source_dependency_ns)))
            require(sum(map(len, self.buffers.values())) <= 850, "On-policy sample cap reached")
        return dict(reply, teacher_action=teacher[0].tolist() if queried else None,
            teacher_queried=queried, teacher_behavior_fraction=0., causal_student_input=True)

    def finish_data(self, request):
        require(not self.finished, "Do not finish data twice")
        files = {}
        for split, rows in self.buffers.items():
            require(rows, "Empty owned split")
            path = self.args.output/(split+"_data.npz")
            with path.open("xb") as stream:
                np.savez(stream, **{k:np.stack([r[k] for r in rows]) for k in rows[0]})
            files[split] = identity(path)
        self.finished = True
        save(self.args.output/"dataset_receipt.json", dict(status="PURE_STUDENT_SHADOW_LABELS_SAVED",
            files=files, teacher_behavior_fraction=0., teacher_labels_privileged=True, test_sources_read=False))
        return dict(files=files, new_optimizer_updates=0)


def serve(args):
    output = sys.stdout
    def reply(value):
        raw = json.dumps(value, allow_nan=False)
        require(len(raw.encode()) < MAX_MESSAGE, "On-policy RPC too large")
        output.write(raw+"\n"); output.flush()
    with redirect_stdout(sys.stderr):
        try:
            session = OnPolicySession(args)
            reply(dict(ok=True, value=session.ready()))
            while True:
                raw = sys.stdin.buffer.readline(MAX_MESSAGE+1)
                require(raw and raw.endswith(b"\n") and len(raw) <= MAX_MESSAGE, "Invalid on-policy RPC")
                request = json.loads(raw)
                if request.get("op") == "quit":
                    reply(dict(ok=True, value=dict(closed=True))); return
                require(request.get("op") in ("start", "action", "finish_data"), "No training operation is exposed")
                assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
                reply(dict(ok=True, value=getattr(session, request["op"])(request)))
        except BaseException as error:
            reply(dict(ok=False, error=f"{type(error).__name__}: {error}")); raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("plan", "checkpoint", "frozen", "primebot-checkpoint", "morph-source", "primebot-source",
            "student-checkpoint", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    for name in ("expected-gpu", "gpu-owner-ticks"):
        p.add_argument("--"+name, required=True)
    for name in ("gpu-owner-pid", "deadline-ns"):
        p.add_argument("--"+name, type=int, required=True)
    p.add_argument("--event-kind", default="nominal")
    p.add_argument("--residual-arm", choices=("causal_history",), default="causal_history")
    p.add_argument("--preview-frames", type=int, choices=(0,), default=0)
    p.add_argument("--tail-method", choices=("masked",), default="masked")
    serve(p.parse_args())
