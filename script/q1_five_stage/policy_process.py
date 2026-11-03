"""CPU native-policy RPC, keeping the physical simulator in its locked runtime."""

import argparse
from contextlib import redirect_stdout
import importlib.metadata
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "primebot_q1"))
from common import identity, require

MAX_MESSAGE = 512 * 1024


class PolicyProcess:
    def __init__(self, command, log, timeout=90.):
        self.timeout, self.buffer, self.child, self.log = timeout, b"", None, None
        self.closed = False
        require(timeout > 0, "Policy process timeout must be positive")
        try:
            self.log = Path(log).open("xb")
            self.child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.log, bufsize=0, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
            self.ready = self.receive()
            require(self.ready.get("cuda_initialized") is False, "CPU policy worker initialized CUDA")
        except BaseException:
            self.close(graceful=False)
            raise

    def receive(self):
        deadline = time.monotonic() + self.timeout
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            require(remaining > 0 and select.select([self.child.stdout], [], [], max(0., remaining))[0],
                    "CPU policy response timed out; do not restart it")
            block = os.read(self.child.stdout.fileno(), 65536)
            require(block, "CPU policy worker exited before its response")
            self.buffer += block
            require(len(self.buffer) <= MAX_MESSAGE, "CPU policy response exceeds its bound")
        line, self.buffer = self.buffer.split(b"\n", 1)
        value = json.loads(line)
        require(isinstance(value, dict) and value.get("ok") is True,
                "CPU policy worker failed: " + str(value.get("error") if isinstance(value, dict) else value))
        return value["value"]

    def request(self, value):
        require(not self.closed and self.child.poll() is None, "CPU policy worker is not live")
        encoded = json.dumps(value, allow_nan=False).encode() + b"\n"
        require(len(encoded) <= MAX_MESSAGE, "CPU policy request exceeds its bound")
        self.child.stdin.write(encoded)
        self.child.stdin.flush()
        return self.receive()

    def start(self, item, seed, interface):
        return self.request(dict(op="start", clip=Path(item["q1_primebot"]["path"]).stem,
                                 seed=seed, interface=interface))

    def action(self, frame, state):
        return self.request(dict(op="action", frame=frame, state=state.tolist()))

    def close(self, graceful=True):
        if self.closed:
            return
        try:
            if self.child is not None:
                if graceful and self.child.poll() is None:
                    self.request(dict(op="quit"))
                    require(self.child.wait(timeout=5) == 0, "CPU policy worker did not exit successfully")
        finally:
            if self.child is not None:
                if self.child.poll() is None:
                    self.child.terminate()
                    try:
                        self.child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.child.kill()
                        self.child.wait(timeout=5)
                for stream in (self.child.stdin, self.child.stdout):
                    stream.close()
            if self.log is not None:
                self.log.close()
            self.closed = True


class NativeSession:
    def __init__(self, args):
        import gc
        import torch
        torch.set_num_threads(1)
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
                "Native policy subprocess must stay on CPU")
        self.torch, self.args = torch, args
        self.plan = json.loads(args.plan.read_text())
        require(self.plan.get("source") == str(args.morph_source), "Policy source differs from its frozen plan")
        self.rows = {Path(row["q1_primebot"]["path"]).stem: row
                     for row in json.loads(args.frozen.read_text())["rows"]}
        sys.path[:0] = [str(args.morph_source / "src"), str(args.primebot_source / "src"),
                       str(Path(__file__).resolve().parents[1] / "morphmimic_q1"),
                       str(Path(__file__).resolve().parents[1] / "primebot_q1_review")]
        from bootstrap import install
        from inference import load_policy
        from evaluate import load_actor
        self.spec = install("q1")["robot"]
        self.actor, self.metadata = load_policy(args.checkpoint, self.spec,
                                               self.plan["code_commit"], expected_plan=self.plan)
        self.robot, unused, self.config, _, _, _ = load_actor(args.primebot_checkpoint)
        del unused
        gc.collect()
        self.default = self.spec.translator.to_canonical(torch.zeros(22))
        self.clip, self.frame = None, 0

    def ready(self):
        return dict(metadata=self.metadata, cuda_initialized=self.torch.cuda.is_initialized(),
            checkpoint=identity(self.args.checkpoint), primebot_checkpoint=identity(self.args.primebot_checkpoint),
            python=sys.version, packages={name: importlib.metadata.version(name) for name in ("torch", "mujoco")},
            physical_transitions=0, optimizer_updates=0)

    def start(self, request):
        import random
        import numpy as np
        from morphmimic.data.motion_library import MotionLibrary
        from morphmimic.train.obs_history import ObsHistory
        from q1_native_review import assert_paired_reference
        from supermimic.data import Q1Motions
        seed, clip = request.get("seed"), request.get("clip")
        require(type(seed) is int and seed in (0, 1, 2) and clip in self.rows
                and request.get("interface") in ("motion", "human"), "Invalid fixed native policy case")
        self.torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        row = self.rows[clip]
        for key in ("q1_primebot", "q1_morph", "human"):
            require(identity(row[key]["path"]) == row[key], "Frozen policy reference changed")
        self.reference = Q1Motions(self.robot, [Path(row["q1_primebot"]["path"])], device="cpu").load(0)
        self.library = MotionLibrary([row["q1_morph"]["path"]], device="cpu", robot=self.spec,
                                     human_root=self.plan["human_root"])
        paired = assert_paired_reference(self.robot, self.reference, self.library, self.spec)
        with np.load(row["human"]["path"], allow_pickle=False) as human:
            require(np.array_equal(self.library.human_pos_w.numpy(), human["human_pos_w"])
                    and np.array_equal(self.library.human_quat_w.numpy(), human["human_quat_w"]),
                    "Policy resolved a different human sidecar")
        self.history = ObsHistory(1, 10, 93, "cpu")
        self.clip, self.frame, self.interface = clip, 0, request["interface"]
        return dict(paired_reference=paired)

    def action(self, request):
        torch = self.torch
        from morphmimic.constants import ANCHOR_IDX
        from morphmimic.model.human import pack_human_goal
        from morphmimic.model.rewards_torch import analytic_obs, goal_features
        from morphmimic.model.state import split_state
        from q1_native_review import native_state, reference_window_indices
        frame = request.get("frame")
        require(self.clip is not None and type(frame) is int and frame == self.frame,
                "Native policy frames must be sequential within a started clip")
        state = torch.tensor(request["state"], dtype=torch.float32)
        require(state.shape == (79,) and torch.isfinite(state).all().item(), "Invalid physical Q1 state")
        converted = native_state(self.robot, state[None], self.spec)
        value = analytic_obs(converted, torch.zeros(29), self.default, converted[:, 240:], self.spec.imu_slot)
        mask = self.actor.present
        observation = torch.cat((torch.where(mask, value[:, :29], 0.), torch.where(mask, value[:, 29:58], 0.),
                                  value[:, 58:64], torch.where(mask, value[:, 64:], 0.)), -1)
        if frame == 0:
            self.history.reset(None, observation)
        else:
            self.history.push(observation)
        offsets = torch.tensor(reference_window_indices(frame, len(self.reference)))
        goals = dict(goal=goal_features(converted, self.library.state(offsets)[None], anchor="reference"),
            goal_kp=pack_human_goal(self.library.human_pos_w[offsets][None],
                self.library.human_quat_w[offsets][None], split_state(converted).body_quat[:, ANCHOR_IDX]))
        action = self.actor(self.history.flat(), goals, self.interface)
        self.frame += 1
        require(not torch.cuda.is_initialized(), "CPU policy worker initialized CUDA")
        return dict(native_action=self.spec.translator.to_native_delta(action)[0].tolist(),
                    canonical_action=action[0].tolist())


def serve(args):
    output = sys.stdout

    def reply(value):
        encoded = json.dumps(value, allow_nan=False)
        require(len(encoded.encode()) < MAX_MESSAGE, "Worker response exceeds its bound")
        output.write(encoded + "\n")
        output.flush()

    with redirect_stdout(sys.stderr):
        try:
            session = NativeSession(args)
            reply(dict(ok=True, value=session.ready()))
            while True:
                line = sys.stdin.buffer.readline(MAX_MESSAGE + 1)
                require(line and line.endswith(b"\n") and len(line) <= MAX_MESSAGE, "Invalid bounded policy request")
                request = json.loads(line)
                op = request.get("op")
                if op == "quit":
                    reply(dict(ok=True, value=dict(closed=True)))
                    return
                require(op in ("start", "action"), "Unknown CPU policy operation")
                reply(dict(ok=True, value=getattr(session, op)(request)))
        except BaseException as error:
            reply(dict(ok=False, error=f"{type(error).__name__}: {error}"))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "checkpoint", "frozen", "primebot-checkpoint", "morph-source", "primebot-source"):
        parser.add_argument("--" + key, type=Path, required=True)
    serve(parser.parse_args())
