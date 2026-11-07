"""Selected residual checkpoint on the unchanged causal receiver and action path."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/"q1_five_stage"), str(HERE.parent/"primebot_q1")]
from common import identity, require
from gpu_preview_worker import GpuPreviewSession
from paused_runtime import assert_owner
from policy_process import MAX_MESSAGE
from student_causal_review_worker import CausalReviewSession


def checkpoint_contract(payload, teacher, teacher_sha, decoder_sha, arm):
    require(payload.get("schema") == "q1_f0_proprio_residual_v1"
        and arm in ("zero_history", "causal_history") and payload.get("arm") == arm
        and payload.get("interface") == "f0_plus_history"
        and payload.get("human_real_future_frames") == 0
        and payload.get("history_frames") == 10 and payload.get("history_frame_dim") == 226
        and payload.get("robot_history_features") == 930 and payload.get("adapter_hidden_width") == 128
        and payload.get("residual_rule") == "z+(1-abs(z))*tanh(adapter(history))"
        and payload.get("selection_split") == "validation"
        and payload.get("test_used_for_selection") is False
        and payload.get("normalization_refitted") is False
        and payload.get("actual_updates") == 3000 and payload.get("actual_adam_steps") == [3000]*4
        and payload.get("selected_update") in (500, 1000, 1500, 2000, 2500, 3000)
        and payload.get("provenance", {}).get("teacher") == teacher
        and payload.get("frozen_teacher_policy_sha256") == teacher_sha
        and payload.get("decoder_sha256") == decoder_sha,
        "Residual arm, causal input, selected checkpoint or frozen teacher provenance differs")


def load_student(native, payload):
    from student_fit import module_sha
    from student_models import current_student
    from student_proprio_models import ProprioResidualStudent, zero_adapter
    base = current_student(native, payload["current_statistics"])
    base.encoder.load_state_dict(payload["base_encoder"], strict=True)
    require(module_sha(base.encoder) == payload["base_encoder_sha256"], "Frozen human encoder digest differs")
    require(not zero_adapter(payload["adapter"]), "Zero adapter must reuse the old f0 outcome without identical replay")
    student = ProprioResidualStudent(base, payload["arm"], native.policy.cfg.latent)
    student.adapter.load_state_dict(payload["adapter"], strict=True)
    return student.eval().requires_grad_(False)


class ProprioReviewSession(CausalReviewSession):
    def __init__(self, args):
        GpuPreviewSession.__init__(self, args)
        from student_fit import module_sha
        from upper_mps_forecast import audit_cpu_tensors
        self.student_identity = identity(args.student_checkpoint)
        payload = self.torch.load(args.student_checkpoint, map_location="cpu", weights_only=True)
        audit_cpu_tensors(self.torch, payload)
        native = self.preview.policy.native
        checkpoint_contract(payload, identity(args.checkpoint), module_sha(native.policy),
            module_sha(native.policy.decoder), args.residual_arm)
        self.student = load_student(native, payload)
        self.selected_update = payload["selected_update"]

    def ready(self):
        return dict(super().ready(), residual_arm=self.args.residual_arm,
            original_causal_action_path=True, cpu_state_converter="original")


def serve(args):
    output = sys.stdout

    def reply(value):
        raw = json.dumps(value, allow_nan=False)
        require(len(raw.encode()) < MAX_MESSAGE, "Residual reply exceeds the original RPC bound")
        output.write(raw+"\n")
        output.flush()

    with redirect_stdout(sys.stderr):
        try:
            session = ProprioReviewSession(args)
            reply(dict(ok=True, value=session.ready()))
            while True:
                raw = sys.stdin.buffer.readline(MAX_MESSAGE+1)
                require(raw and raw.endswith(b"\n") and len(raw) <= MAX_MESSAGE, "Invalid residual RPC")
                request = json.loads(raw)
                if request.get("op") == "quit":
                    reply(dict(ok=True, value=dict(closed=True)))
                    return
                require(request.get("op") in ("start", "action"), "Unknown residual operation")
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
    parser.add_argument("--residual-arm", choices=("zero_history", "causal_history"), required=True)
    parser.add_argument("--expected-gpu", required=True)
    parser.add_argument("--gpu-owner-pid", type=int, required=True)
    parser.add_argument("--gpu-owner-ticks", required=True)
    parser.add_argument("--preview-frames", type=int, choices=(0,), default=0)
    parser.add_argument("--tail-method", choices=("masked",), default="masked")
    serve(parser.parse_args())
