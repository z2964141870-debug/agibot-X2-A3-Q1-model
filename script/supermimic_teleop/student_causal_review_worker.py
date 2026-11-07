"""CUDA f0 student on the support-aware receiver, with explicit stale fallback."""

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


def checkpoint_contract(payload, teacher, teacher_sha, decoder_sha):
    require(payload.get("schema") == "q1_low_latency_f0_history_encoder_v1"
        and payload.get("interface") == "f0_plus_history"
        and payload.get("human_real_future_frames") == 0
        and payload.get("history_frames") == 10 and payload.get("history_frame_dim") == 226
        and payload.get("selection_split") == "validation"
        and payload.get("test_used_for_selection") is False
        and payload.get("actual_updates", 0) > 0
        and payload.get("provenance", {}).get("teacher") == teacher
        and payload.get("frozen_teacher_policy_sha256") == teacher_sha
        and payload.get("decoder_sha256") == decoder_sha,
        "f0 encoder, causal input or frozen teacher provenance differs")


class CausalReviewSession(GpuPreviewSession):
    def __init__(self, args):
        super().__init__(args)
        from student_fit import module_sha
        from student_models import current_student
        from upper_mps_forecast import audit_cpu_tensors

        self.student_identity = identity(args.student_checkpoint)
        payload = self.torch.load(args.student_checkpoint, map_location="cpu", weights_only=True)
        audit_cpu_tensors(self.torch, payload)
        native = self.preview.policy.native
        checkpoint_contract(payload, identity(args.checkpoint), module_sha(native.policy),
                            module_sha(native.policy.decoder))
        self.student = current_student(native, payload["current_statistics"]).eval().requires_grad_(False)
        self.student.encoder.load_state_dict(payload["encoder"], strict=True)
        self.selected_update = payload["selected_update"]

    def preview_config(self):
        return dict(interface="f0_plus_history", frames=0, real_human_future_frames=0,
            history_frames=10, maximum_current_capture_age_ns=200_000_000,
            preprocessing="offline_unverified", measured_end_to_end_latency_ns=None)

    def ready(self):
        return dict(super().ready(), variant="f0_student", student_checkpoint=self.student_identity,
                    selected_update=self.selected_update, human_real_future_frames=0)

    def start(self, request):
        result = NativeSession.start(self, request)
        self.stream = OfflineStudentStream(self.library.human_pos_w.numpy(), self.library.human_quat_w.numpy(),
            spec=EventSpec(self.args.event_kind, 50, 10, 0),
            sha256=self.rows[self.clip]["human"]["sha256"], session="r2-f0:"+self.clip)
        self.previous_action = self.torch.zeros(1, 29)
        return dict(result, preview=self.preview_config())

    def action(self, request):
        torch = self.torch
        from morphmimic.constants import ANCHOR_IDX
        from morphmimic.model.rewards_torch import analytic_obs
        from morphmimic.model.state import split_state
        from morphmimic.train.normalizers import normalize_history
        from q1_native_review import native_state
        from student_inputs import current_features

        frame = request.get("frame")
        require(type(frame) is int and frame == self.frame and self.clip is not None,
                "f0 review requires sequential actual robot states")
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
        window = self.stream.student(frame, mode="f0_history")
        native = self.preview.policy.native
        with torch.inference_mode():
            if window.can_step:
                torso = split_state(converted).body_quat[0, ANCHOR_IDX].numpy()
                features = current_features(window, torso)
                goal = torch.tensor(features, dtype=torch.float32, device="cuda:0")[None]
                history = normalize_history(native.norms["obs_norm"], self.history.flat().to("cuda:0"))
                action = torch.where(native.present, self.student(history, goal), 0.).cpu()
            else:
                action = self.previous_action.clone()
        require(action.shape == (1, 29) and torch.isfinite(action).all().item()
                and not torch.count_nonzero(action[..., ~present]).item(), "Invalid f0 student action")
        timing = dict(window.timing(), **self.compute_record(started), frame=frame,
            variant="f0_student", horizon_frames=0, human_real_future_frames=0, event=self.args.event_kind,
            input_ready=window.can_step, input_status=window.status,
            fallback=None if window.can_step else ("initial_zero_delta" if frame == 0 else "previous_action_hold"),
            fallback_safety_validated=False, worker_action_wall_ns=time.perf_counter_ns()-overall)
        timing["compute_scope"] = "f0 receiver/features, history normalization, GPU forward and output transfer"
        self.previous_action = action.clone()
        self.frame += 1
        return dict(native_action=self.spec.translator.to_native_delta(action)[0].tolist(),
                    canonical_action=action[0].tolist(), stream=timing)


def serve(args):
    output = sys.stdout

    def reply(value):
        raw = json.dumps(value, allow_nan=False)
        require(len(raw.encode()) < MAX_MESSAGE, "f0 reply exceeds the local RPC bound")
        output.write(raw+"\n")
        output.flush()

    with redirect_stdout(sys.stderr):
        try:
            session = CausalReviewSession(args)
            reply(dict(ok=True, value=session.ready()))
            while True:
                raw = sys.stdin.buffer.readline(MAX_MESSAGE+1)
                require(raw and raw.endswith(b"\n") and len(raw) <= MAX_MESSAGE, "Invalid f0 RPC")
                request = json.loads(raw)
                if request.get("op") == "quit":
                    reply(dict(ok=True, value=dict(closed=True)))
                    return
                require(request.get("op") in ("start", "action"), "Unknown f0 operation")
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
    parser.add_argument("--event-kind", choices=("nominal", "stop", "reverse", "dropout", "late_burst"), required=True)
    parser.add_argument("--expected-gpu", required=True)
    parser.add_argument("--gpu-owner-pid", type=int, required=True)
    parser.add_argument("--gpu-owner-ticks", required=True)
    parser.add_argument("--preview-frames", type=int, choices=(0,), default=0)
    parser.add_argument("--tail-method", choices=("masked",), default="masked")
    serve(parser.parse_args())
