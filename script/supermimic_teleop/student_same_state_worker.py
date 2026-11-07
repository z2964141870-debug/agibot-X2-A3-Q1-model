"""Instrument the unchanged causal action math on recorded states; forbid physics."""

import argparse
from contextlib import ExitStack, redirect_stdout
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/"q1_five_stage"), str(HERE.parent/"primebot_q1")]
from common import require
from paused_runtime import assert_owner
from policy_process import MAX_MESSAGE
from student_causal_review_worker import CausalReviewSession


class SameStateSession(CausalReviewSession):
    def __init__(self, args):
        super().__init__(args)
        self.marks = [self.torch.cuda.Event(enable_timing=True) for _ in range(4)]

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
                "Require consecutive original f0 states from frame zero")
        overall = last = time.perf_counter_ns()
        stages = {}

        def mark(name):
            nonlocal last
            value = time.perf_counter_ns()
            stages[name], last = value-last, value

        state = torch.tensor(request["state"], dtype=torch.float32)
        require(state.shape == (79,) and torch.isfinite(state).all().item(), "Invalid recorded state")
        mark("cpu_state_validation")
        converted = native_state(self.robot, state[None], self.spec)
        mark("cpu_native_state")
        value = analytic_obs(converted, torch.zeros(29), self.default, converted[:, 240:], self.spec.imu_slot)
        mark("cpu_analytic_observation")
        present = self.actor.present
        obs = torch.cat((torch.where(present, value[:, :29], 0.), torch.where(present, value[:, 29:58], 0.),
            value[:, 58:64], torch.where(present, value[:, 64:], 0.)), -1)
        self.history.reset(None, obs) if frame == 0 else self.history.push(obs)
        mark("cpu_mask_and_history")
        started = time.perf_counter_ns()
        window = self.stream.student(frame, mode="f0_history")
        mark("cpu_causal_receiver")
        native = self.preview.policy.native
        with torch.inference_mode():
            if window.can_step:
                torso = split_state(converted).body_quat[0, ANCHOR_IDX].numpy()
                features = current_features(window, torso)
            mark("cpu_current_history_features")
            if window.can_step:
                self.marks[0].record()
                goal = torch.tensor(features, dtype=torch.float32, device="cuda:0")[None]
                history = normalize_history(native.norms["obs_norm"], self.history.flat().to("cuda:0"))
                self.marks[1].record()
            mark("input_transfer_and_history_normalization_host")
            if window.can_step:
                device_action = torch.where(native.present, self.student(history, goal), 0.)
                self.marks[2].record()
            mark("student_forward_host")
            if window.can_step:
                action = device_action.cpu()
                self.marks[3].record()
            else:
                action = self.previous_action.clone()
            mark("output_transfer_or_explicit_hold_host")
        require(action.shape == (1, 29) and torch.isfinite(action).all().item()
                and not torch.count_nonzero(action[..., ~present]).item(), "Invalid recorded f0 output")
        mark("output_validation_cpu")
        timing = dict(window.timing(), **self.compute_record(started), frame=frame,
            variant="f0_student", horizon_frames=0, human_real_future_frames=0, event=self.args.event_kind,
            input_ready=window.can_step, input_status=window.status,
            fallback=None if window.can_step else ("initial_zero_delta" if frame == 0 else "previous_action_hold"),
            fallback_safety_validated=False, worker_action_wall_ns=time.perf_counter_ns()-overall)
        timing["compute_scope"] = "f0 receiver/features, history normalization, GPU forward and output transfer"
        mark("original_final_sync_and_stream_record")
        self.previous_action = action.clone()
        self.frame += 1
        intervals = None if not window.can_step else {name:self.marks[i].elapsed_time(self.marks[i+1])
            for i,name in enumerate(("input_and_normalization", "student_forward", "output_transfer"))}
        result = dict(native_action=self.spec.translator.to_native_delta(action)[0].tolist(),
            canonical_action=action[0].tolist(), stream=timing)
        mark("timing_readback_and_action_translation")
        result["profile"] = dict(stage_wall_ns=stages, worker_action_wall_ns=last-overall,
            cuda_stream_interval_ms=intervals, intermediate_cuda_synchronizations=0,
            note="Host and CUDA stream intervals overlap and include host gaps. Worker excludes RPC serialization, physics and hardware latency.")
        return result


def serve(args):
    output = sys.stdout

    def reply(value):
        raw = json.dumps(value, allow_nan=False)
        require(len(raw.encode()) < MAX_MESSAGE, "Bounded same-state RPC exceeded")
        output.write(raw+"\n")
        output.flush()

    def forbidden(*args, **kwargs):
        raise AssertionError("No physical integration in the same-state diagnostic")

    with redirect_stdout(sys.stderr):
        try:
            import mujoco
            with ExitStack() as stack:
                for name in ("mj_step", "mj_step1", "mj_step2"):
                    stack.enter_context(patch.object(mujoco, name, side_effect=forbidden))
                session = SameStateSession(args)
                reply(dict(ok=True, value=dict(session.ready(), same_state_profile=True,
                    canonical_present=session.actor.present.tolist())))
                while True:
                    raw = sys.stdin.buffer.readline(MAX_MESSAGE+1)
                    require(raw and raw.endswith(b"\n") and len(raw) <= MAX_MESSAGE, "Invalid same-state request")
                    request = json.loads(raw)
                    if request.get("op") == "quit":
                        reply(dict(ok=True, value=dict(closed=True)))
                        return
                    require(request.get("op") in ("start", "action"), "Unknown same-state operation")
                    assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
                    reply(dict(ok=True, value=getattr(session, request["op"])(request)))
        except BaseException as error:
            reply(dict(ok=False, error=f"{type(error).__name__}: {error}"))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("plan", "checkpoint", "frozen", "primebot-checkpoint", "morph-source", "primebot-source", "student-checkpoint"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--event-kind", choices=("nominal",), required=True)
    parser.add_argument("--expected-gpu", required=True)
    parser.add_argument("--gpu-owner-pid", type=int, required=True)
    parser.add_argument("--gpu-owner-ticks", required=True)
    parser.add_argument("--preview-frames", type=int, choices=(0,), required=True)
    parser.add_argument("--tail-method", choices=("masked",), required=True)
    serve(parser.parse_args())
