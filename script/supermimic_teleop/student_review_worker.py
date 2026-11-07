"""Evaluate one frozen R1 encoder against native packing on the same receiver."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/"q1_five_stage"), str(HERE.parent/"primebot_q1")]
from common import identity, require
from gpu_preview_worker import GpuPreviewSession
from paused_runtime import assert_owner
from policy_process import MAX_MESSAGE, NativeSession
from stream_events import EventSpec
from student_data import OfflineStudentStream


class StudentReviewSession(GpuPreviewSession):
    def __init__(self, args):
        super().__init__(args)
        from student_fit import module_sha
        from student_models import short_preview_student
        from upper_mps_forecast import audit_cpu_tensors

        self.student = None
        self.student_identity = identity(args.student_checkpoint)
        payload = self.torch.load(args.student_checkpoint, map_location="cpu", weights_only=True)
        audit_cpu_tensors(self.torch, payload)
        native = self.preview.policy.native
        require(payload.get("schema") == "q1_low_latency_student_encoder_v1"
                and payload.get("interface") == "human_prefix_adapter"
                and payload.get("selection_split") == "validation"
                and payload.get("test_used_for_selection") is False
                and payload.get("actual_updates", 0) > 0
                and payload.get("provenance", {}).get("teacher") == identity(args.checkpoint)
                and payload.get("frozen_teacher_policy_sha256") == module_sha(native.policy)
                and payload.get("decoder_sha256") == module_sha(native.policy.decoder),
                "R1 encoder, frozen teacher or shared decoder provenance differs")
        if args.variant == "student":
            self.student = short_preview_student(native).eval().requires_grad_(False)
            self.student.encoder.load_state_dict(payload["encoder"], strict=True)
        self.selected_update = payload["selected_update"]

    def ready(self):
        return dict(super().ready(), variant=self.args.variant, student_checkpoint=self.student_identity,
                    selected_update=self.selected_update, input_clock="120Hz support-aware synthetic receiver")

    def start(self, request):
        value = NativeSession.start(self, request)
        self.stream = OfflineStudentStream(self.library.human_pos_w.numpy(), self.library.human_quat_w.numpy(),
            spec=EventSpec(self.args.event_kind, self.args.event_frame, self.args.event_duration, 0),
            sha256=self.rows[self.clip]["human"]["sha256"], session="r1-review:"+self.clip)
        self.previous_action = self.torch.zeros(1, 29)
        return value

    def action(self, request):
        torch = self.torch
        from morphmimic.constants import ANCHOR_IDX
        from morphmimic.model.rewards_torch import analytic_obs
        from morphmimic.model.state import split_state
        from morphmimic.train.normalizers import normalize_history
        from q1_native_review import native_state
        from student_models import native_preview_input

        frame = request.get("frame")
        require(type(frame) is int and frame == self.frame and self.clip is not None,
                "Student evaluation needs sequential actual states")
        overall = time.perf_counter_ns()
        state = torch.tensor(request["state"], dtype=torch.float32)
        require(state.shape == (79,) and torch.isfinite(state).all().item(), "Invalid common-plant state")
        converted = native_state(self.robot, state[None], self.spec)
        value = analytic_obs(converted, torch.zeros(29), self.default, converted[:, 240:], self.spec.imu_slot)
        present = self.actor.present
        obs = torch.cat((torch.where(present, value[:, :29], 0.), torch.where(present, value[:, 29:58], 0.),
            value[:, 58:64], torch.where(present, value[:, 64:], 0.)), -1)
        self.history.reset(None, obs) if frame == 0 else self.history.push(obs)
        started = time.perf_counter_ns()
        window = self.stream.student(frame, mode="short_preview", horizon=self.args.preview_frames)
        native = self.preview.policy.native
        parity = None
        with torch.inference_mode():
            if window.can_step:
                history = normalize_history(native.norms["obs_norm"], self.history.flat().to("cuda:0"))
                torso = split_state(converted).body_quat[:, ANCHOR_IDX].to("cuda:0")
                goal = native_preview_input(native, [window], torso)
                if self.student is None:
                    action = native.policy.forward_human(history, goal)
                    # Independently compose the encoder/decoder before new physical steps.
                    explicit = native.policy.decoder(native.policy.human_encoder(goal), history)
                    parity = float((action-explicit).abs().max())
                    require(torch.allclose(action, explicit, atol=2e-5, rtol=2e-5),
                            "Native evaluation encoder/decoder packing changed")
                else:
                    action = self.student(history, goal)
                action = torch.where(native.present, action, 0.).cpu()
            else:
                action = self.previous_action.clone()
        require(action.shape == (1, 29) and torch.isfinite(action).all().item()
                and not torch.count_nonzero(action[..., ~present]).item(), "Invalid student evaluation action")
        compute = self.compute_record(started)
        self.previous_action = action.clone()
        fallback = None if window.can_step else ("initial_zero_delta" if frame == 0 else "previous_action_hold")
        timing = dict(window.timing(), **compute, frame=frame, variant=self.args.variant,
            horizon_frames=self.args.preview_frames, event=self.args.event_kind,
            used_sequences=list(window.used_sequences), input_ready=window.can_step,
            input_status=window.status, fallback=fallback, fallback_safety_validated=False,
            packing_maximum_action_difference=parity,
            worker_action_wall_ns=time.perf_counter_ns()-overall)
        self.frame += 1
        return dict(native_action=self.spec.translator.to_native_delta(action)[0].tolist(),
                    canonical_action=action[0].tolist(), stream=timing)


def serve(args):
    output = sys.stdout

    def reply(value):
        raw = json.dumps(value, allow_nan=False)
        require(len(raw.encode()) < MAX_MESSAGE, "Student review reply exceeds RPC bound")
        output.write(raw+"\n")
        output.flush()

    with redirect_stdout(sys.stderr):
        try:
            session = StudentReviewSession(args)
            reply(dict(ok=True, value=session.ready()))
            while True:
                line = sys.stdin.buffer.readline(MAX_MESSAGE+1)
                require(line and line.endswith(b"\n") and len(line) <= MAX_MESSAGE, "Invalid review RPC")
                request = json.loads(line)
                if request.get("op") == "quit":
                    reply(dict(ok=True, value=dict(closed=True)))
                    return
                require(request.get("op") in ("start", "action"), "Unknown student review operation")
                assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
                reply(dict(ok=True, value=getattr(session, request["op"])(request)))
        except BaseException as error:
            reply(dict(ok=False, error=f"{type(error).__name__}: {error}"))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("plan", "checkpoint", "frozen", "primebot-checkpoint", "morph-source", "primebot-source",
                 "student-checkpoint"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--variant", choices=("teacher", "student"), required=True)
    parser.add_argument("--preview-frames", type=int, choices=(1, 4, 10), required=True)
    parser.add_argument("--tail-method", choices=("masked",), default="masked")
    parser.add_argument("--event-kind", choices=("nominal", "stop", "reverse", "dropout", "late_burst"), required=True)
    parser.add_argument("--event-frame", type=int, default=50)
    parser.add_argument("--event-duration", type=int, default=10)
    parser.add_argument("--expected-gpu", required=True)
    parser.add_argument("--gpu-owner-pid", type=int, required=True)
    parser.add_argument("--gpu-owner-ticks", required=True)
    serve(parser.parse_args())
