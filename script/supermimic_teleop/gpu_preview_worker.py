"""CUDA policy only; preserve the separate pinned CPU physical simulator."""

import argparse
from contextlib import redirect_stdout
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "q1_five_stage"), str(HERE.parent / "primebot_q1")]
from common import identity, require
from policy_process import MAX_MESSAGE, PolicyProcess
from preview_worker import PreviewSession
from preview_policy import HumanPreviewPolicy
from forecast_policy import HumanForecastPolicy
from paused_runtime import assert_owner


class GpuPolicyProcess(PolicyProcess):
    def __init__(self, command, log, gpu, timeout=90.):
        import subprocess
        self.timeout, self.buffer, self.child, self.log = timeout, b"", None, None
        self.closed = False
        require(timeout > 0, "Policy process timeout must be positive")
        try:
            self.log = Path(log).open("xb")
            self.child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.log, bufsize=0, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu))
            self.ready = self.receive()
            require(self.ready.get("cuda_initialized") is True and self.ready.get("policy_device") == "cuda",
                    "Requested GPU policy did not initialize CUDA")
        except BaseException:
            self.close(graceful=False)
            raise


class DevicePreview:
    def __init__(self, native, tail):
        self.policy = HumanPreviewPolicy(native) if tail == "masked" else HumanForecastPolicy(native)

    def __call__(self, history, window, torso):
        import torch
        with torch.inference_mode():
            return self.policy(history.to("cuda:0"), window, torso.to("cuda:0")).cpu()


class GpuPreviewSession(PreviewSession):
    def __init__(self, args):
        assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == args.expected_gpu,
                "GPU worker must expose only the admitted original GPU")
        import torch
        torch.set_num_threads(1)
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Admitted CUDA device is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch, self.args = torch, args
        self.plan = json.loads(args.plan.read_text())
        require(self.plan["source"] == str(args.morph_source), "GPU policy source differs from its plan")
        self.rows = {Path(row["q1_primebot"]["path"]).stem: row
                     for row in json.loads(args.frozen.read_text())["rows"]}
        sys.path[:0] = [str(args.morph_source / "src"), str(args.primebot_source / "src"),
            str(HERE.parent / "morphmimic_q1"), str(HERE.parent / "primebot_q1_review")]
        from bootstrap import install
        from inference import load_policy
        from interim_review import load_physical_robot
        self.spec = install("q1")["robot"]
        native, self.metadata = load_policy(args.checkpoint, self.spec, self.plan["code_commit"],
                                            "cuda:0", expected_plan=self.plan)
        self.robot, self.config = load_physical_robot(args.primebot_checkpoint)
        self.actor = SimpleNamespace(present=native.present.cpu())
        self.default = self.spec.translator.to_canonical(torch.zeros(22))
        self.tail_method = args.tail_method
        self.preview = DevicePreview(native, args.tail_method)
        self.clip, self.frame = None, 0

    def ready(self):
        self.torch.cuda.synchronize()
        return dict(metadata=self.metadata, cuda_initialized=True, policy_device="cuda",
            gpu=self.args.expected_gpu, checkpoint=identity(self.args.checkpoint),
            primebot_checkpoint=identity(self.args.primebot_checkpoint), preview=self.preview_config(),
            python=sys.version, packages={k: importlib.metadata.version(k) for k in ("torch", "mujoco")},
            physical_transitions=0, optimizer_updates=0)

    def compute_record(self, started):
        self.torch.cuda.synchronize()
        return dict(gpu_policy_wall_compute_ns=time.perf_counter_ns() - started,
            policy_device="cuda", compute_scope="goal packing, input transfer, GPU forward and output transfer")


def serve(args):
    output = sys.stdout

    def reply(value):
        encoded = json.dumps(value, allow_nan=False)
        require(len(encoded.encode()) < MAX_MESSAGE, "GPU worker reply exceeds its bound")
        output.write(encoded + "\n")
        output.flush()

    with redirect_stdout(sys.stderr):
        try:
            session = GpuPreviewSession(args)
            reply(dict(ok=True, value=session.ready()))
            while True:
                line = sys.stdin.buffer.readline(MAX_MESSAGE+1)
                require(line and line.endswith(b"\n") and len(line) <= MAX_MESSAGE, "Invalid GPU policy request")
                request = json.loads(line)
                op = request.get("op")
                if op == "quit":
                    reply(dict(ok=True, value=dict(closed=True)))
                    return
                require(op in ("start", "action"), "Unknown GPU policy operation")
                assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
                reply(dict(ok=True, value=getattr(session, op)(request)))
        except BaseException as error:
            reply(dict(ok=False, error=f"{type(error).__name__}: {error}"))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("plan", "checkpoint", "frozen", "primebot-checkpoint", "morph-source", "primebot-source"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--preview-frames", type=int, choices=(1, 2, 4, 5, 10), required=True)
    parser.add_argument("--tail-method", choices=("masked", "hold", "damped_cv"), default="masked")
    parser.add_argument("--expected-gpu", required=True)
    parser.add_argument("--gpu-owner-pid", type=int, required=True)
    parser.add_argument("--gpu-owner-ticks", required=True)
    serve(parser.parse_args())
