"""Query privileged labels on explicitly mixed student-state collection only."""

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
from stream_events import EventSpec
from student_causal_review_worker import CausalReviewSession
from student_data import OfflineStudentStream


def mixed_action(teacher, student, fraction):
    if (fraction != .5 or teacher.shape != student.shape or teacher.shape != (1, 29)
            or not np.isfinite(teacher).all() or not np.isfinite(student).all()):
        raise ValueError("Use the published finite equal teacher/student behavior mixture")
    return fraction*teacher+(1-fraction)*student


class AggregateSession(CausalReviewSession):
    def __init__(self, args):
        super().__init__(args)
        self.settings = read(args.aggregate_plan)
        self.jobs = read(args.output/"collection_plan.json")["jobs"]
        self.buffers = {"train": [], "validation": []}
        self.finished = False

    def start(self, request):
        clip_id = request.get("clip_id")
        require(type(clip_id) is int and 0 <= clip_id < len(self.jobs) and not self.finished,
                "Unknown aggregation source or collection already finished")
        self.clip_id, self.job = clip_id, self.jobs[clip_id]
        require(request.get("clip") == self.job["clip"], "Aggregation source identity changed")
        self.args.event_kind = self.job["packet_event"]["kind"]
        result = NativeSession.start(self, request)
        self.stream = OfflineStudentStream(self.library.human_pos_w.numpy(), self.library.human_quat_w.numpy(),
            spec=EventSpec(**self.job["packet_event"]), sha256=self.job["human"]["sha256"],
            session="r2-aggregate:"+self.job["clip"])
        self.previous_action = self.torch.zeros(1, 29)
        return dict(result, behavior="half_teacher_half_f0_student_training_only")

    def action(self, request):
        require(time.monotonic_ns() < self.args.deadline_ns-60_000_000_000,
                "Collection exhausted its reserved fit and cleanup time")
        frame = request["frame"]
        require(frame < self.job["maximum_steps"], "Collection exceeds the source step bound")
        student_reply = super().action(request)
        torch = self.torch
        from morphmimic.constants import ANCHOR_IDX
        from morphmimic.model.human import pack_human_goal
        from morphmimic.model.state import split_state
        from morphmimic.train.normalizers import normalize_history
        from q1_native_review import native_state
        from student_inputs import current_features

        raw_state = np.asarray(request["state"], dtype=np.float64)
        converted = native_state(self.robot, torch.tensor(raw_state, dtype=torch.float32)[None], self.spec)
        torso = split_state(converted).body_quat[:, ANCHOR_IDX]
        native = self.preview.policy.native
        labels = self.stream.teacher_labels(frame)
        with torch.no_grad():
            goal = pack_human_goal(torch.tensor(np.array(labels.positions), dtype=torch.float32, device="cuda:0")[None],
                torch.tensor(np.array(labels.quaternions), dtype=torch.float32, device="cuda:0")[None], torso.to("cuda:0"))
            teacher = native(self.history.flat().to("cuda:0"), {"goal_kp": goal}, "human").cpu()
            history = normalize_history(native.norms["obs_norm"], self.history.flat().to("cuda:0"))
        teacher_values = teacher.numpy()
        student_values = np.asarray(student_reply["canonical_action"], dtype=np.float32)[None]
        behavior = mixed_action(teacher_values, student_values, self.settings["teacher_behavior_fraction"])
        require(not np.count_nonzero(behavior[..., ~native.present.cpu().numpy()]),
                "Collection behavior activates an absent joint")
        self.previous_action = torch.tensor(behavior)
        current = self.stream.student(frame, mode="f0_history")
        if frame % self.settings["record_stride"] == 0 and current.can_step:
            self.buffers[self.job["split"]].append(dict(
                f0=current_features(current, torso[0].numpy()).astype(np.float32),
                history=history[0].cpu().numpy().copy(), teacher_action=teacher_values[0].copy(),
                actual_state=raw_state.copy(), behavior_action=behavior[0].copy(),
                student_action=student_values[0].copy(), frame=np.int64(frame), clip_id=np.int64(self.clip_id)))
            require(sum(map(len, self.buffers.values())) <= 4096, "Aggregation record bound reached")
        return dict(native_action=self.spec.translator.to_native_delta(torch.tensor(behavior))[0].tolist(),
            canonical_action=behavior[0].tolist(), teacher_action=teacher_values[0].tolist(),
            student_action=student_values[0].tolist(), stream=student_reply["stream"],
            behavior_teacher_fraction=.5, causal_student_input=True,
            collection_behavior_is_causal=False, teacher_labels_are_privileged=True)

    def finish_data(self, request):
        require(not self.finished, "Collection already finished")
        files = {}
        for split, rows in self.buffers.items():
            require(rows, "Empty aggregate split")
            path = self.args.output/(split+"_data.npz")
            with path.open("xb") as stream:
                np.savez(stream, **{key:np.stack([row[key] for row in rows]) for key in rows[0]})
            files[split] = identity(path)
        require(sum(row["bytes"] for row in files.values()) <= self.settings["maximum_dataset_bytes"],
                "Aggregation dataset exceeds the published storage bound")
        self.finished = True
        self.buffers.clear()
        value = dict(status="MIXED_STUDENT_STATE_LABELS_SAVED", files=files,
            teacher_behavior_fraction=.5, teacher_labels_equal_executed_action_claimed=False,
            state_source="actual common Primebot CPU physics", test_sources_read=False)
        save(self.args.output/"dataset_receipt.json", value)
        return value

    def train(self, request):
        from student_aggregate_fit import fit
        from student_causal_training import reused_dataset
        require(self.finished and time.monotonic_ns() < self.args.deadline_ns-60_000_000_000,
                "Fit requires completed and checked collection with remaining time")
        check = read(self.args.output/"collection_result.json")
        require(check.get("status") == "PASSED" and check.get("test_sources_read") is False
            and all(value["state_and_behavior_match"] and value["labels_match_saved_queries"]
                    for value in check["data_validation"].values()), "Aggregate states or labels were not checked")
        prior, original = reused_dataset(self.args.r1_training_root, self.jobs, load=True)
        receipt = read(self.args.output/"dataset_receipt.json")
        aggregate = {}
        for split, record in receipt["files"].items():
            require(identity(record["path"]) == record, "Aggregate data changed before fitting")
            with np.load(record["path"], allow_pickle=False) as data:
                aggregate[split] = {key:data[key] for key in ("f0", "history", "teacher_action", "clip_id")}
        payload = self.torch.load(self.args.student_checkpoint, map_location="cpu", weights_only=True)
        save(self.args.output/"current_statistics.json", payload["current_statistics"])
        result = fit(self.preview.policy.native, original, aggregate, payload, self.settings["fit"],
            self.args.output, self.args.deadline_ns, dict(teacher=identity(self.args.checkpoint),
                original_r1_fit=identity(self.args.r1_training_root/"fit_result.json"),
                original_datasets=read(self.args.r1_training_root/"dataset_receipt.json")["files"],
                aggregate_datasets=receipt["files"], warm_start=self.student_identity,
                aggregate_plan=identity(self.args.aggregate_plan)))
        require(all(identity(row["path"]) == row for row in prior["records"]+list(receipt["files"].values())),
                "Original or aggregate data changed during optimization")
        return result


def serve(args):
    output = sys.stdout
    def reply(value):
        raw = json.dumps(value, allow_nan=False)
        require(len(raw.encode()) < MAX_MESSAGE, "Aggregation RPC exceeds the fixed bound")
        output.write(raw+"\n")
        output.flush()
    with redirect_stdout(sys.stderr):
        try:
            session = AggregateSession(args)
            reply(dict(ok=True, value=session.ready()))
            while True:
                raw = sys.stdin.buffer.readline(MAX_MESSAGE+1)
                require(raw and raw.endswith(b"\n") and len(raw) <= MAX_MESSAGE, "Invalid aggregation RPC")
                request = json.loads(raw)
                if request.get("op") == "quit":
                    reply(dict(ok=True, value=dict(closed=True)))
                    return
                require(request.get("op") in ("start", "action", "finish_data", "train"), "Unknown aggregation operation")
                assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
                reply(dict(ok=True, value=getattr(session, request["op"])(request)))
        except BaseException as error:
            reply(dict(ok=False, error=f"{type(error).__name__}: {error}"))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("plan", "checkpoint", "frozen", "primebot-checkpoint", "morph-source", "primebot-source",
                 "student-checkpoint", "output", "aggregate-plan", "r1-training-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--event-kind", default="nominal")
    parser.add_argument("--expected-gpu", required=True)
    parser.add_argument("--gpu-owner-pid", type=int, required=True)
    parser.add_argument("--gpu-owner-ticks", required=True)
    parser.add_argument("--deadline-ns", type=int, required=True)
    parser.add_argument("--preview-frames", type=int, choices=(0,), default=0)
    parser.add_argument("--tail-method", choices=("masked",), default="masked")
    serve(parser.parse_args())
