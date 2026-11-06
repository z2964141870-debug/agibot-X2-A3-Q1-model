"""CUDA human policy for independent synthetic event stress replays."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "q1_five_stage"), str(HERE.parent / "primebot_q1")]
from common import require
from event_inputs import EventReceiver
from gpu_preview_worker import GpuPreviewSession
from paused_runtime import assert_owner
from policy_process import MAX_MESSAGE
from stream_events import EventSpec


class EventSession(GpuPreviewSession):
    def start(self, request):
        result = super().start(request)
        self.events = EventReceiver(self.library.human_pos_w.numpy(), self.library.human_quat_w.numpy(),
            EventSpec(self.args.event_kind, self.args.event_frame, self.args.event_duration,
                      self.args.transport_frames), self.rows[self.clip]["human"]["sha256"],
            f"{self.clip}:seed{request['seed']}:h{self.args.preview_frames}:{self.tail_method}")
        return dict(result, event=dict(kind=self.args.event_kind, first_frame=self.args.event_frame,
            duration_frames=self.args.event_duration, transport_frames=self.args.transport_frames))

    def action(self, request):
        torch = self.torch
        from morphmimic.constants import ANCHOR_IDX
        from morphmimic.model.human import fixed_human_masks, pack_human_goal
        from morphmimic.model.rewards_torch import analytic_obs
        from morphmimic.model.state import split_state
        from morphmimic.train.imagination import act_human
        from q1_native_review import native_state

        frame = request.get("frame")
        require(type(frame) is int and frame == self.frame and self.clip is not None,
                "Event policy requires a started sequential clip")
        state = torch.tensor(request["state"], dtype=torch.float32)
        require(state.shape == (79,) and torch.isfinite(state).all().item(), "Invalid event common-plant state")
        converted = native_state(self.robot, state[None], self.spec)
        value = analytic_obs(converted, torch.zeros(29), self.default, converted[:, 240:], self.spec.imu_slot)
        present = self.actor.present
        obs = torch.cat((torch.where(present, value[:, :29], 0.), torch.where(present, value[:, 29:58], 0.),
                        value[:, 58:64], torch.where(present, value[:, 64:], 0.)), -1)
        self.history.reset(None, obs) if frame == 0 else self.history.push(obs)
        started = time.perf_counter_ns()
        goal = self.events.step(frame, self.args.preview_frames, self.tail_method)
        native = self.preview.policy.native
        with torch.inference_mode():
            history = self.history.flat().to("cuda:0")
            torso = split_state(converted).body_quat[:, ANCHOR_IDX].to("cuda:0")
            positions = torch.tensor([p.positions for p in goal.poses], device="cuda:0", dtype=history.dtype)[None]
            orientations = torch.tensor([p.orientations for p in goal.poses], device="cuda:0", dtype=history.dtype)[None]
            raw = pack_human_goal(positions, orientations, torso)
            masks = fixed_human_masks(1, "full", goal.policy_frames, history.device, heading=True)
            action = act_human(native.policy, native.norms["obs_norm"], native.norms["human_norm"],
                              native.norms["heading_norm"], history, raw, *masks).cpu()
        require(action.shape == (1, 29) and torch.isfinite(action).all().item()
                and not torch.count_nonzero(action[..., ~present]).item(), "Invalid canonical event action")
        compute = self.compute_record(started)
        self.frame += 1
        return dict(native_action=self.spec.translator.to_native_delta(action)[0].tolist(),
            canonical_action=action[0].tolist(), stream=dict(goal.timing, **compute,
                compute_scope="event visibility and goal packing, transfers, GPU forward and output transfer",
                timing_clock_note="Synthetic schedule and synchronized software timing are separate clocks"))


def serve(args):
    output = sys.stdout

    def reply(value):
        encoded = json.dumps(value, allow_nan=False)
        require(len(encoded.encode()) < MAX_MESSAGE, "Event reply exceeds its bound")
        output.write(encoded+"\n")
        output.flush()

    with redirect_stdout(sys.stderr):
        try:
            session = EventSession(args)
            reply(dict(ok=True, value=session.ready()))
            while True:
                line = sys.stdin.buffer.readline(MAX_MESSAGE+1)
                require(line and line.endswith(b"\n") and len(line) <= MAX_MESSAGE, "Invalid event RPC request")
                request = json.loads(line)
                if request.get("op") == "quit":
                    reply(dict(ok=True, value=dict(closed=True)))
                    return
                require(request.get("op") in ("start", "action"), "Unknown event operation")
                assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
                reply(dict(ok=True, value=getattr(session, request["op"])(request)))
        except BaseException as error:
            reply(dict(ok=False, error=f"{type(error).__name__}: {error}"))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("plan", "checkpoint", "frozen", "primebot-checkpoint", "morph-source", "primebot-source"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--preview-frames", type=int, choices=(1, 2, 4, 5, 10), required=True)
    parser.add_argument("--tail-method", choices=("masked", "hold", "damped_cv"), required=True)
    parser.add_argument("--expected-gpu", required=True)
    parser.add_argument("--gpu-owner-pid", type=int, required=True)
    parser.add_argument("--gpu-owner-ticks", required=True)
    parser.add_argument("--event-kind", choices=("nominal", "stop", "reverse", "dropout", "late_burst"), required=True)
    parser.add_argument("--event-frame", type=int, default=50)
    parser.add_argument("--event-duration", type=int, default=10)
    parser.add_argument("--transport-frames", type=int, default=0)
    serve(parser.parse_args())
