"""Owned CUDA teacher collection and R1 fit; physics stays in the CPU parent."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/"q1_five_stage"), str(HERE.parent/"primebot_q1")]
from common import identity, require, save
from gpu_preview_worker import GpuPreviewSession
from paused_runtime import assert_owner
from policy_process import MAX_MESSAGE, NativeSession
from stream_events import EventSpec
from student_data import OfflineStudentStream
from student_models import native_preview_input


class TrainingSession(GpuPreviewSession):
    def __init__(self, args):
        super().__init__(args)
        self.jobs = json.loads((args.output/"collection_plan.json").read_text())["jobs"]
        self.student_plan = json.loads(args.student_plan.read_text())
        self.buffers = {"train": [], "validation": []}
        self.parity_maximum, self.parity_records = 0., 0
        self.finished = False

    def start(self, request):
        clip_id = request.get("clip_id")
        require(type(clip_id) is int and 0 <= clip_id < len(self.jobs), "Unknown collection job")
        job = self.jobs[clip_id]
        require(request.get("clip") == job["clip"] and not self.finished, "Collection clip or phase changed")
        result = NativeSession.start(self, request)
        self.clip_id, self.job = clip_id, job
        self.offline = OfflineStudentStream(self.library.human_pos_w.numpy(), self.library.human_quat_w.numpy(),
            spec=EventSpec(**job["packet_event"]), sha256=job["human"]["sha256"], session="collection:"+job["clip"])
        return result

    def action(self, request):
        torch = self.torch
        from morphmimic.constants import ANCHOR_IDX
        from morphmimic.model.human import (fixed_human_masks, heading_from_goal,
            human_features_from_goal, mask_human_goal, pack_human_goal)
        from morphmimic.model.rewards_torch import analytic_obs
        from morphmimic.model.state import split_state
        from morphmimic.train.normalizers import normalize_history
        from q1_native_review import native_state

        frame = request.get("frame")
        require(type(frame) is int and frame == self.frame and frame < self.job["maximum_steps"],
                "Collect actual states sequentially within the bounded episode")
        import numpy as np
        raw_state = np.asarray(request["state"], dtype=np.float64)
        state = torch.tensor(raw_state, dtype=torch.float32)
        require(state.shape == (79,) and torch.isfinite(state).all().item(), "Invalid collected physical state")
        converted = native_state(self.robot, state[None], self.spec)
        value = analytic_obs(converted, torch.zeros(29), self.default, converted[:, 240:], self.spec.imu_slot)
        present = self.actor.present
        obs = torch.cat((torch.where(present, value[:, :29], 0.), torch.where(present, value[:, 29:58], 0.),
            value[:, 58:64], torch.where(present, value[:, 64:], 0.)), -1)
        self.history.reset(None, obs) if frame == 0 else self.history.push(obs)
        native = self.preview.policy.native
        history = self.history.flat().to("cuda:0")
        torso = split_state(converted).body_quat[:, ANCHOR_IDX].to("cuda:0")
        indices = torch.arange(frame+1, frame+11)
        with torch.no_grad():
            raw = pack_human_goal(self.library.human_pos_w[indices][None].to("cuda:0"),
                self.library.human_quat_w[indices][None].to("cuda:0"), torso)
            action = native(history, {"goal_kp": raw}, "human").clone()
        require(action.shape == (1, 29) and torch.isfinite(action).all().item()
                and not torch.count_nonzero(action[..., ~native.present]).item(), "Invalid collected teacher action")
        if frame >= 9 and frame % self.job["record_stride"] == 0:
            from student_inputs import current_features

            with torch.no_grad():
                points, frames, heading_mask = fixed_human_masks(1, "full", 10, history.device, heading=True)
                features = native.norms["human_norm"](human_features_from_goal(raw, points))
                heading = native.norms["heading_norm"](heading_from_goal(raw, points))
                full = mask_human_goal(features, points, frames, heading, heading_mask)
                normalized_history = normalize_history(native.norms["obs_norm"], history)
                reproduced = native.policy.forward_human(normalized_history, full)
                reproduced = torch.where(native.present, reproduced, 0.)
                require(torch.allclose(action, reproduced, atol=2e-5, rtol=2e-5),
                        "Collected teacher action disagrees with explicit frozen packing")
                self.parity_maximum = max(self.parity_maximum, float((action-reproduced).abs().max()))
                prefix = []
                for horizon in (10, 5, 4, 2, 1):
                    window = self.offline.student(frame, mode="short_preview", horizon=horizon)
                    count = window.prefix_frames if window.can_step else 0
                    prefix.append(count)
                    if count:
                        goal = native_preview_input(native, [window], torso)
                        from student_models import masked_preview_goal
                        packed = masked_preview_goal(features, heading, torch.tensor([count], device=history.device))
                        require(torch.allclose(goal, packed, atol=2e-5, rtol=2e-5),
                                "Future packing changed a visible student coordinate or heading")
                self.parity_records += 1
            current = self.offline.student(frame, mode="f0_history")
            require(current.can_step, "Collection current/history support unexpectedly stale")
            row = dict(full_goal=full[0].detach().cpu().numpy().copy(),
                history=normalized_history[0].detach().cpu().numpy().copy(),
                f0=current_features(current, torso[0].cpu().numpy()).astype(np.float32),
                teacher_action=action[0].detach().cpu().numpy().copy(), actual_state=raw_state.copy(),
                prefix=np.asarray(prefix, dtype=np.int64), frame=np.int64(frame), clip_id=np.int64(self.clip_id))
            self.buffers[self.job["split"]].append(row)
            require(sum(len(rows) for rows in self.buffers.values()) <= 4096, "Bounded collection row limit reached")
        self.frame += 1
        return dict(native_action=self.spec.translator.to_native_delta(action)[0].detach().cpu().tolist(),
                    canonical_action=action[0].detach().cpu().tolist())

    def finish_data(self, request):
        import numpy as np

        require(not self.finished and self.parity_records > 0, "Collection already finished or parity absent")
        files = {}
        for split, rows in self.buffers.items():
            require(rows, "Collection split is empty: "+split)
            arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
            path = self.args.output/(split+"_data.npz")
            with path.open("xb") as stream:
                np.savez(stream, **arrays)
            files[split] = identity(path)
        require(sum(value["bytes"] for value in files.values()) <= self.student_plan["data"]["maximum_dataset_bytes"],
                "Collected dataset storage exceeds published bound")
        self.finished = True
        self.buffers.clear()
        result = dict(status="PACKING_ACTION_PARITY_PASSED", files=files,
            parity_records=self.parity_records, maximum_absolute_action_difference=self.parity_maximum,
            actual_state_source="common Primebot CPU physics", test_sources_read=False)
        save(self.args.output/"dataset_receipt.json", result)
        return result

    def train(self, request):
        import numpy as np
        from student_fit import fit_r1

        require(self.finished and time.monotonic_ns() < self.args.deadline_ns-60_000_000_000,
                "Training needs completed collection and unspent reserved time")
        collection = json.loads((self.args.output/"collection_result.json").read_text())
        require(collection.get("status") == "PASSED" and all(row["physical_state_trace_match"]
            and row["teacher_action_trace_match"] for row in collection["data_validation"].values()),
            "Dataset does not match actual physical traces")
        receipt = json.loads((self.args.output/"dataset_receipt.json").read_text())
        data = {}
        for split in ("train", "validation"):
            path = self.args.output/(split+"_data.npz")
            require(identity(path) == receipt["files"][split], "Collected dataset changed before fit")
            with np.load(path, allow_pickle=False) as source:
                data[split] = dict(source)
        result = fit_r1(self.preview.policy.native, data, self.student_plan["r1"], self.args.output,
            self.args.deadline_ns, dict(teacher=identity(self.args.checkpoint), datasets=receipt["files"],
                student_plan=identity(self.args.student_plan), collection_plan=identity(self.args.output/"collection_plan.json")))
        require(all(identity(row["path"]) == row for row in receipt["files"].values()),
                "Collected data changed during optimization")
        return result


def serve(args):
    output = sys.stdout

    def reply(value):
        raw = json.dumps(value, allow_nan=False)
        require(len(raw.encode()) < MAX_MESSAGE, "Training RPC response exceeds bound")
        output.write(raw+"\n")
        output.flush()

    with redirect_stdout(sys.stderr):
        try:
            session = TrainingSession(args)
            reply(dict(ok=True, value=session.ready()))
            while True:
                line = sys.stdin.buffer.readline(MAX_MESSAGE+1)
                require(line and line.endswith(b"\n") and len(line) <= MAX_MESSAGE, "Invalid training RPC")
                request = json.loads(line)
                if request.get("op") == "quit":
                    reply(dict(ok=True, value=dict(closed=True)))
                    return
                require(request.get("op") in ("start", "action", "finish_data", "train"), "Unknown training operation")
                assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
                require(time.monotonic_ns() < args.deadline_ns, "Reserved training wall budget exhausted")
                reply(dict(ok=True, value=getattr(session, request["op"])(request)))
        except BaseException as error:
            reply(dict(ok=False, error=f"{type(error).__name__}: {error}"))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("plan", "checkpoint", "frozen", "primebot-checkpoint", "morph-source", "primebot-source", "output", "student-plan"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--expected-gpu", required=True)
    parser.add_argument("--gpu-owner-pid", type=int, required=True)
    parser.add_argument("--gpu-owner-ticks", required=True)
    parser.add_argument("--deadline-ns", type=int, required=True)
    parser.add_argument("--preview-frames", type=int, choices=(10,), default=10)
    parser.add_argument("--tail-method", choices=("masked",), default="masked")
    serve(parser.parse_args())
