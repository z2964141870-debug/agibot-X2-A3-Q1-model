"""Independent instrumented CUDA action path; never edit the saved replay worker."""

import argparse
from contextlib import ExitStack, redirect_stdout
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "q1_five_stage"), str(HERE.parent / "primebot_q1")]
from common import require
from event_gpu_worker import EventSession
from paused_runtime import assert_owner
from policy_process import MAX_MESSAGE


class ProfileSession(EventSession):
    def __init__(self, args):
        super().__init__(args)
        self.cuda_marks = [self.torch.cuda.Event(enable_timing=True) for _ in range(5)]

    def action(self, request):
        wall_start = last = time.perf_counter_ns()
        stages = {}

        def mark(name):
            nonlocal last
            current = time.perf_counter_ns()
            stages[name], last = current - last, current

        torch = self.torch
        from morphmimic.constants import ANCHOR_IDX
        from morphmimic.model.human import fixed_human_masks, pack_human_goal
        from morphmimic.model.rewards_torch import analytic_obs
        from morphmimic.model.state import split_state
        from morphmimic.train.imagination import act_human
        from q1_native_review import native_state

        frame = request.get("frame")
        require(type(frame) is int and frame == self.frame and self.clip is not None,
                "Profile policy requires the same sequential saved states")
        state = torch.tensor(request["state"], dtype=torch.float32)
        require(state.shape == (79,) and torch.isfinite(state).all().item(), "Invalid saved Q1 state")
        mark("imports_and_state_validation")
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
        goal = self.events.step(frame, self.args.preview_frames, self.tail_method)
        mark("cpu_event_goal")
        native = self.preview.policy.native
        with torch.inference_mode():
            self.cuda_marks[0].record()
            history = self.history.flat().to("cuda:0")
            torso = split_state(converted).body_quat[:, ANCHOR_IDX].to("cuda:0")
            positions = torch.tensor([p.positions for p in goal.poses], device="cuda:0", dtype=history.dtype)[None]
            orientations = torch.tensor([p.orientations for p in goal.poses], device="cuda:0", dtype=history.dtype)[None]
            self.cuda_marks[1].record()
            mark("input_transfer_host")
            raw = pack_human_goal(positions, orientations, torso)
            masks = fixed_human_masks(1, "full", goal.policy_frames, history.device, heading=True)
            self.cuda_marks[2].record()
            mark("goal_pack_and_masks_host")
            device_action = act_human(native.policy, native.norms["obs_norm"], native.norms["human_norm"],
                native.norms["heading_norm"], history, raw, *masks)
            self.cuda_marks[3].record()
            mark("policy_forward_host")
            action = device_action.cpu()
            self.cuda_marks[4].record()
            mark("output_transfer_host")
        require(action.shape == (1, 29) and torch.isfinite(action).all().item()
                and not torch.count_nonzero(action[..., ~present]).item(), "Invalid profiled canonical action")
        mark("output_validation_cpu")
        compute = self.compute_record(started)
        mark("original_final_sync")
        gpu = {name: self.cuda_marks[i].elapsed_time(self.cuda_marks[i+1])
               for i, name in enumerate(("input_transfer", "goal_pack_and_masks", "policy_forward", "output_transfer"))}
        self.frame += 1
        result = dict(native_action=self.spec.translator.to_native_delta(action)[0].tolist(),
            canonical_action=action[0].tolist(), stream={**goal.timing, **compute,
                "compute_scope": "event visibility and goal packing, transfers, GPU forward and output transfer",
                "timing_clock_note": "Synthetic schedule and synchronized software timing are separate clocks"})
        mark("cuda_timing_readback_and_action_translation")
        result["profile"] = dict(stage_wall_ns=stages, worker_action_wall_ns=last-wall_start,
            cuda_stream_interval_ms=gpu, intermediate_cuda_synchronizations=0,
            cuda_event_instrumentation_enabled=True,
            note="Host launch intervals and CUDA stream elapsed intervals overlap; do not sum them. Stream intervals include host gaps, not kernel-only time. Worker wall excludes RPC serialization and physical execution.")
        return result


def serve(args):
    output = sys.stdout

    def reply(value):
        encoded = json.dumps(value, allow_nan=False)
        require(len(encoded.encode()) < MAX_MESSAGE, "Profile reply exceeds the bounded RPC message")
        output.write(encoded + "\n")
        output.flush()

    def forbidden(*unused, **kwargs):
        raise AssertionError("No physical integration is allowed in the saved-state profiler")

    with redirect_stdout(sys.stderr):
        try:
            session = ProfileSession(args)
            import mujoco
            with ExitStack() as stack:
                for name in ("mj_step", "mj_step1", "mj_step2"):
                    stack.enter_context(patch.object(mujoco, name, side_effect=forbidden))
                reply(dict(ok=True, value=dict(session.ready(), profiling=True)))
                while True:
                    line = sys.stdin.buffer.readline(MAX_MESSAGE+1)
                    require(line and line.endswith(b"\n") and len(line) <= MAX_MESSAGE, "Invalid profile request")
                    request = json.loads(line)
                    if request.get("op") == "quit":
                        reply(dict(ok=True, value=dict(closed=True)))
                        return
                    require(request.get("op") in ("start", "action"), "Unknown profile operation")
                    assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
                    reply(dict(ok=True, value=getattr(session, request["op"])(request)))
        except BaseException as error:
            reply(dict(ok=False, error=f"{type(error).__name__}: {error}"))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("plan", "checkpoint", "frozen", "primebot-checkpoint", "morph-source", "primebot-source"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--preview-frames", type=int, choices=(4, 10), required=True)
    parser.add_argument("--tail-method", choices=("masked", "damped_cv"), required=True)
    parser.add_argument("--expected-gpu", required=True)
    parser.add_argument("--gpu-owner-pid", type=int, required=True)
    parser.add_argument("--gpu-owner-ticks", required=True)
    parser.add_argument("--event-kind", choices=("stop", "late_burst"), required=True)
    parser.add_argument("--event-frame", type=int, default=50)
    parser.add_argument("--event-duration", type=int, default=10)
    parser.add_argument("--transport-frames", type=int, default=0)
    serve(parser.parse_args())
