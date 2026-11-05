"""Isolated native CPU policy with timestamped human-only short previews.

This is offline sidecar playback with zero modeled transport, not live sensor
latency. Source preprocessing causality remains explicitly unverified. The ten
20 ms slots, full point mask, checkpoint and frozen normalizers stay unchanged.
"""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "q1_five_stage")]
from policy_process import MAX_MESSAGE, NativeSession
from common import require
from preview_policy import HumanPreviewPolicy
from stream_contract import DT_NS, HumanPose, PoseFrame, PoseTimeline, SourceIdentity


class PreviewSession(NativeSession):
    def __init__(self, args):
        super().__init__(args)
        self.preview = HumanPreviewPolicy(self.actor)

    def ready(self):
        return dict(super().ready(), preview=self.preview_config())

    def preview_config(self):
        return dict(interface="human", points="full",
            frames=self.args.preview_frames, target_offsets_ns=[(i+1)*DT_NS for i in range(10)],
            modeled_transport_ns=0, preprocessing="offline_unverified",
            measured_end_to_end_latency_ns=None)

    def start(self, request):
        require(request.get("interface") == "human", "Short preview study is human interface only")
        result = super().start(request)
        require(len(self.library.human_pos_w) == len(self.reference), "Human and physical clip lengths differ")
        row = self.rows[self.clip]
        source = SourceIdentity(self.clip, f"seed{request['seed']}_h{self.args.preview_frames}",
            "offline_replay_50hz_ns", "smpl24_world_m_wxyz", "offline_unverified", row["human"]["sha256"])
        self.timeline, self.last_admitted = PoseTimeline(source), -1
        return dict(result, preview=self.preview_config())

    def action(self, request):
        torch = self.torch
        from morphmimic.constants import ANCHOR_IDX
        from morphmimic.model.rewards_torch import analytic_obs
        from morphmimic.model.state import split_state
        from q1_native_review import native_state

        frame = request.get("frame")
        require(self.clip is not None and type(frame) is int and frame == self.frame,
                "Preview policy frames must be sequential within a started clip")
        state = torch.tensor(request["state"], dtype=torch.float32)
        require(state.shape == (79,) and torch.isfinite(state).all().item(), "Invalid common-plant Q1 state")
        converted = native_state(self.robot, state[None], self.spec)
        value = analytic_obs(converted, torch.zeros(29), self.default, converted[:, 240:], self.spec.imu_slot)
        present = self.actor.present
        observation = torch.cat((torch.where(present, value[:, :29], 0.),
            torch.where(present, value[:, 29:58], 0.), value[:, 58:64],
            torch.where(present, value[:, 64:], 0.)), -1)
        if frame == 0:
            self.history.reset(None, observation)
        else:
            self.history.push(observation)
        last = len(self.library.human_pos_w) - 1
        horizon = self.args.preview_frames
        for index in range(self.last_admitted+1, min(frame+horizon, last)+1):
            pose = HumanPose(self.library.human_pos_w[index].tolist(),
                             self.library.human_quat_w[index].tolist())
            capture = index*DT_NS
            self.timeline.ingest(PoseFrame(self.timeline.source, index, capture, None,
                capture, capture, pose, terminal=index == last))
            self.last_admitted = index
        window = self.timeline.window(origin_ns=frame*DT_NS, control_ns=(frame+horizon)*DT_NS,
                                      preview_frames=horizon, max_capture_age_ns=10*DT_NS)
        started = time.perf_counter_ns()
        action = self.preview(self.history.flat(), window, split_state(converted).body_quat[:, ANCHOR_IDX])
        compute_ns = time.perf_counter_ns() - started
        self.frame += 1
        require(not torch.cuda.is_initialized(), "Preview CPU policy worker initialized CUDA")
        return dict(native_action=self.spec.translator.to_native_delta(action)[0].tolist(),
            canonical_action=action[0].tolist(), stream=dict(window.timing(),
                policy_prefix_frames=window.prefix_frames, measured_frames=sum(window.measured_mask),
                provenance=window.provenance, used_sequences=window.used_sequences,
                cpu_policy_wall_compute_ns=compute_ns,
                timing_clock_note="Replay schedule and wall compute use separate clocks; do not sum as measured latency"))


def serve(args):
    output = sys.stdout

    def reply(value):
        encoded = json.dumps(value, allow_nan=False)
        require(len(encoded.encode()) < MAX_MESSAGE, "Preview worker response exceeds its bound")
        output.write(encoded + "\n")
        output.flush()

    with redirect_stdout(sys.stderr):
        try:
            session = PreviewSession(args)
            reply(dict(ok=True, value=session.ready()))
            while True:
                line = sys.stdin.buffer.readline(MAX_MESSAGE+1)
                require(line and line.endswith(b"\n") and len(line) <= MAX_MESSAGE,
                        "Invalid bounded preview policy request")
                request = json.loads(line)
                op = request.get("op")
                if op == "quit":
                    reply(dict(ok=True, value=dict(closed=True)))
                    return
                require(op in ("start", "action"), "Unknown preview worker operation")
                reply(dict(ok=True, value=getattr(session, op)(request)))
        except BaseException as error:
            reply(dict(ok=False, error=f"{type(error).__name__}: {error}"))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "checkpoint", "frozen", "primebot-checkpoint", "morph-source", "primebot-source"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--preview-frames", type=int, choices=(1, 2, 4, 5, 10), required=True)
    serve(parser.parse_args())
