"""Query a frozen CUDA teacher on new CPU-physics prefixes, then fit f0/history."""

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
from gpu_preview_worker import GpuPreviewSession
from paused_runtime import assert_owner
from policy_process import MAX_MESSAGE, NativeSession
from stream_events import EventSpec
from student_data import OfflineStudentStream


class CoverageSession(GpuPreviewSession):
    def __init__(self, args):
        super().__init__(args)
        self.settings = read(args.coverage_plan)
        self.candidate = read(args.candidate_plan)
        self.jobs = self.candidate["new_collection_jobs"]
        self.buffers = {"train":[], "validation":[]}
        self.finished, self.parity_records, self.parity_max = False, 0, 0.

    def start(self, request):
        clip_id = request.get("clip_id")
        require(type(clip_id) is int and 0 <= clip_id < len(self.jobs) and not self.finished,
            "Unknown new coverage source or already finished collection")
        self.clip_id, self.job = clip_id, self.jobs[clip_id]
        require(request.get("clip") == self.job["clip"], "Coverage source changed")
        result = NativeSession.start(self, request)
        self.stream = OfflineStudentStream(self.library.human_pos_w.numpy(), self.library.human_quat_w.numpy(),
            spec=EventSpec(**self.job["packet_event"]), sha256=self.job["human"]["sha256"],
            session="coverage:"+self.job["clip"])
        self.checked_clip = False
        return dict(result, collection_behavior="privileged_full_teacher_only", collection_behavior_is_causal=False)

    def action(self, request):
        torch = self.torch
        from morphmimic.constants import ANCHOR_IDX
        from morphmimic.model.human import (fixed_human_masks, heading_from_goal,
            human_features_from_goal, mask_human_goal, pack_human_goal)
        from morphmimic.model.rewards_torch import analytic_obs
        from morphmimic.model.state import split_state
        from morphmimic.train.normalizers import normalize_history
        from q1_native_review import native_state
        from student_inputs import current_features

        frame = request.get("frame")
        require(type(frame) is int and frame == self.frame and frame < self.job["maximum_steps"]
            and time.monotonic_ns() < self.args.deadline_ns-120_000_000_000,
            "Coverage source, sequence or reserved fit deadline changed")
        started = time.perf_counter_ns()
        raw_state = np.asarray(request["state"], dtype=np.float64)
        require(raw_state.shape == (79,) and np.isfinite(raw_state).all(), "Invalid coverage physical state")
        converted = native_state(self.robot, torch.tensor(raw_state, dtype=torch.float32)[None], self.spec)
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
            action = native(history, {"goal_kp":raw}, "human").cpu()
        require(action.shape == (1, 29) and torch.isfinite(action).all().item()
            and not torch.count_nonzero(action[..., ~present]).item(), "Invalid coverage teacher action")
        window = self.stream.student(frame, mode="f0_history")
        if frame % self.job["record_stride"] == 0 and window.can_step:
            with torch.no_grad():
                normalized_history = normalize_history(native.norms["obs_norm"], history)
                if not self.checked_clip:
                    points, frames, heading_mask = fixed_human_masks(1, "full", 10, history.device, heading=True)
                    features = native.norms["human_norm"](human_features_from_goal(raw, points))
                    heading = native.norms["heading_norm"](heading_from_goal(raw, points))
                    full = mask_human_goal(features, points, frames, heading, heading_mask)
                    reproduced = torch.where(native.present, native.policy.forward_human(normalized_history, full), 0.).cpu()
                    require(torch.allclose(action, reproduced, atol=2e-5, rtol=2e-5),
                        "Coverage explicit frozen teacher packing differs")
                    self.parity_max = max(self.parity_max, float((action-reproduced).abs().max()))
                    self.parity_records += 1
                    self.checked_clip = True
            self.buffers[self.job["split"]].append(dict(
                f0=current_features(window, torso[0].cpu().numpy()).astype(np.float32),
                history=normalized_history[0].cpu().numpy().copy(), teacher_action=action[0].numpy().copy(),
                actual_state=raw_state.copy(), frame=np.int64(frame), clip_id=np.int64(self.clip_id)))
            require(sum(map(len, self.buffers.values())) <= self.settings["maximum_new_records"],
                "Coverage exceeded its new record bound")
        self.frame += 1
        timing = dict(window.timing(), frame=frame, input_ready=window.can_step, input_status=window.status,
            human_real_future_frames=0, variant="coverage_teacher_collection",
            policy_device="cuda", collection_behavior_is_causal=False,
            worker_action_wall_ns=time.perf_counter_ns()-started)
        return dict(native_action=self.spec.translator.to_native_delta(action)[0].tolist(),
            canonical_action=action[0].tolist(), stream=timing)

    def finish_data(self, request):
        require(not self.finished and self.parity_records == len(self.jobs), "Incomplete coverage packing checks")
        files = {}
        for split, rows in self.buffers.items():
            require(rows, "Empty new coverage split")
            path = self.args.output/(split+"_data.npz")
            with path.open("xb") as stream:
                np.savez(stream, **{k:np.stack([r[k] for r in rows]) for k in rows[0]})
            files[split] = identity(path)
        require(sum(r["bytes"] for r in files.values()) <= self.settings["maximum_dataset_bytes"],
            "Coverage dataset storage bound exceeded")
        self.finished = True
        self.buffers.clear()
        result = dict(status="COVERAGE_TEACHER_LABELS_SAVED", files=files, parity_records=self.parity_records,
            maximum_absolute_action_difference=self.parity_max, collection_behavior_is_causal=False,
            teacher_labels_equal_executed_action=True, test_sources_read=False)
        save(self.args.output/"dataset_receipt.json", result)
        return result

    def train(self, request):
        from student_coverage_fit import fit
        from student_causal_review_worker import checkpoint_contract
        from student_fit import module_sha
        from upper_mps_forecast import audit_cpu_tensors

        require(self.finished and read(self.args.output/"collection_result.json").get("status") == "PASSED"
            and time.monotonic_ns() < self.args.deadline_ns-60_000_000_000,
            "Coverage fitting needs checked actual collection and remaining time")
        origins, records = [], []
        for root in (self.args.training_root, self.args.aggregate_root, self.args.output):
            receipt = read(root/"dataset_receipt.json")
            values = {}
            for split in ("train", "validation"):
                record = receipt["files"][split]
                require(identity(record["path"]) == record, "A reused or new coverage dataset changed")
                with np.load(record["path"], allow_pickle=False) as arrays:
                    values[split] = {k:arrays[k] for k in ("f0", "history", "teacher_action", "clip_id")}
                records.append(record)
            origins.append(values)
        payload = self.torch.load(self.args.student_checkpoint, map_location="cpu", weights_only=True)
        audit_cpu_tensors(self.torch, payload)
        native = self.preview.policy.native
        checkpoint_contract(payload, identity(self.args.checkpoint), module_sha(native.policy), module_sha(native.policy.decoder))
        save(self.args.output/"current_statistics.json", payload["current_statistics"])
        result = fit(native, origins, payload, self.settings["fit"], self.candidate, self.args.output,
            self.args.deadline_ns, dict(teacher=identity(self.args.checkpoint),
                warm_start=identity(self.args.student_checkpoint), datasets=records,
                coverage_plan=identity(self.args.coverage_plan), candidate_plan=identity(self.args.candidate_plan)))
        require(all(identity(r["path"]) == r for r in records), "Coverage data changed during fitting")
        return result


def serve(args):
    output = sys.stdout
    def reply(value):
        raw = json.dumps(value, allow_nan=False)
        require(len(raw.encode()) < MAX_MESSAGE, "Coverage RPC exceeds the fixed bound")
        output.write(raw+"\n")
        output.flush()
    with redirect_stdout(sys.stderr):
        try:
            session = CoverageSession(args)
            reply(dict(ok=True, value=session.ready()))
            while True:
                raw = sys.stdin.buffer.readline(MAX_MESSAGE+1)
                require(raw and raw.endswith(b"\n") and len(raw) <= MAX_MESSAGE, "Invalid coverage RPC")
                request = json.loads(raw)
                if request.get("op") == "quit":
                    reply(dict(ok=True, value=dict(closed=True)))
                    return
                require(request.get("op") in ("start", "action", "finish_data", "train"), "Unknown coverage operation")
                assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
                reply(dict(ok=True, value=getattr(session, request["op"])(request)))
        except BaseException as error:
            reply(dict(ok=False, error=f"{type(error).__name__}: {error}"))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("plan", "checkpoint", "frozen", "primebot-checkpoint", "morph-source", "primebot-source",
            "output", "coverage-plan", "candidate-plan", "training-root", "aggregate-root", "student-checkpoint"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--expected-gpu", required=True)
    parser.add_argument("--gpu-owner-pid", type=int, required=True)
    parser.add_argument("--gpu-owner-ticks", required=True)
    parser.add_argument("--deadline-ns", type=int, required=True)
    parser.add_argument("--preview-frames", type=int, choices=(10,), default=10)
    parser.add_argument("--tail-method", choices=("masked",), default="masked")
    serve(parser.parse_args())
