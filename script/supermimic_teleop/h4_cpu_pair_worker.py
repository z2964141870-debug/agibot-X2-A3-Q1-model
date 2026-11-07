"""Independent R1 H4 worker; keep the original action path and swap CPU FK only."""

import argparse
from contextlib import ExitStack, nullcontext, redirect_stdout
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/"q1_five_stage"), str(HERE.parent/"primebot_q1")]
from common import require
from native_state_action_math import isolated_converter
from paused_runtime import assert_owner
from policy_process import MAX_MESSAGE
from student_review_worker import StudentReviewSession


class PairedSession(StudentReviewSession):
    def __init__(self, args):
        super().__init__(args)
        from native_state_cpu import KinematicState
        self.converter = KinematicState(self.robot, self.spec) if args.cpu_converter == "kinematic" else None

    def ready(self):
        return dict(super().ready(), cpu_converter=self.args.cpu_converter,
            canonical_present=self.actor.present.tolist(), original_action_path=True)

    def action(self, request):
        import mujoco
        import q1_native_review
        started = time.perf_counter_ns()
        context = (nullcontext() if self.converter is None else
            isolated_converter(q1_native_review, self.robot, self.spec, self.converter))
        with ExitStack() as stack:
            stack.enter_context(context)
            for name in ("mj_step", "mj_step1", "mj_step2", "mj_forward", "mj_inverse"):
                stack.enter_context(patch.object(mujoco, name,
                    side_effect=AssertionError("Saved states only; no physical integration")))
            value = super().action(request)
        value["stream"]["isolated_request_worker_wall_ns"] = time.perf_counter_ns()-started
        value["stream"]["cpu_converter"] = self.args.cpu_converter
        return value


def serve(args):
    output = sys.stdout

    def reply(value):
        raw = json.dumps(value, allow_nan=False)
        require(len(raw.encode()) < MAX_MESSAGE, "H4 CPU comparison RPC bound")
        output.write(raw+"\n")
        output.flush()

    with redirect_stdout(sys.stderr):
        try:
            import mujoco
            with ExitStack() as stack:
                for name in ("mj_step", "mj_step1", "mj_step2"):
                    stack.enter_context(patch.object(mujoco, name,
                        side_effect=AssertionError("No new physical replay")))
                session = PairedSession(args)
                reply(dict(ok=True, value=session.ready()))
                while True:
                    raw = sys.stdin.buffer.readline(MAX_MESSAGE+1)
                    require(raw and raw.endswith(b"\n") and len(raw) <= MAX_MESSAGE, "Invalid bounded H4 request")
                    request = json.loads(raw)
                    if request.get("op") == "quit":
                        reply(dict(ok=True, value=dict(closed=True)))
                        return
                    require(request.get("op") in ("start", "action"), "Unknown H4 operation")
                    assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
                    reply(dict(ok=True, value=getattr(session, request["op"])(request)))
        except BaseException as error:
            reply(dict(ok=False, error=f"{type(error).__name__}: {error}"))
            raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("plan", "checkpoint", "frozen", "primebot-checkpoint", "morph-source",
                 "primebot-source", "student-checkpoint"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--variant", choices=("student",), required=True)
    p.add_argument("--preview-frames", type=int, choices=(4,), required=True)
    p.add_argument("--tail-method", choices=("masked",), required=True)
    p.add_argument("--cpu-converter", choices=("original", "kinematic"), required=True)
    p.add_argument("--event-kind", choices=("nominal", "dropout"), required=True)
    p.add_argument("--event-frame", type=int, choices=(50,), default=50)
    p.add_argument("--event-duration", type=int, choices=(10,), default=10)
    p.add_argument("--expected-gpu", required=True)
    p.add_argument("--gpu-owner-pid", type=int, required=True)
    p.add_argument("--gpu-owner-ticks", required=True)
    serve(p.parse_args())
