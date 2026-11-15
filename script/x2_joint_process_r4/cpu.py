"""Ordered CPU-only author path in a clean spawn interpreter."""
import contextlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

from .config import DEADLINE_UTC, ROLE_CPUS


def bind(role, output):
    cpus = ROLE_CPUS[role]
    os.sched_setaffinity(0, cpus)
    actual = sorted(os.sched_getaffinity(0))
    if actual != cpus:
        raise RuntimeError("Own CPU affinity rejected")
    with (Path(output)/"kernel_binding_fgp.jsonl").open("a") as f:
        f.write(json.dumps(dict(role=role, pid=os.getpid(), tid=threading.get_native_id(),
                               cpus=actual, monotonic_ns=time.monotonic_ns(), hardware_commands=0))+"\n")


def decode(value, previous, limit):
    if not isinstance(value, dict) or set(value) != {"sequence", "frame", "calibrated", "acc", "rot"}:
        raise ValueError("Malformed own CPU packet")
    sequence, frame = value["sequence"], value["frame"]
    if (type(sequence) is not int or type(frame) is not int or sequence != previous[0]+1
            or not 1 <= sequence <= limit or frame <= previous[1]):
        raise ValueError("Out-of-order or over-budget CPU packet")
    for key, shape in (("acc", (10, 3)), ("rot", (10, 3, 3)), ("calibrated", (120,))):
        array = np.asarray(value[key])
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError("Invalid CPU packet "+key)
    return sequence, frame


def child(connection, source, output, limit, owner):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    output, source = Path(output), Path(source)
    output.mkdir(exist_ok=False)
    bind("cpu", output.parent)
    pipeline = None
    count = 0
    status, error, hits = "CPU_PROCESS_FAILED", None, 0
    try:
        from script.x2_joint_pipeline.worker import tune_sessions
        from script.x2_morph_human.clotho_fgp_runner import prepare_clotho_environment
        from script.x2_joint_pipeline.pipeline import PreparedDenoiser
        from script.x2_fgp_accel.feature_cache import ExactFeatureCache
        from script.x2_joint_opt.profile import instrument
        tune_sessions()
        prepare_clotho_environment(source)
        import torch
        from onnx_inference import OnnxBundle
        from stage_tran_gp.tracker import GlobalPoseTracker
        from stage_tran_gp.features import valid_rotations
        torch.set_num_threads(2)
        caches = []
        original = GlobalPoseTracker.__init__
        def tracker(instance, *args, **kwargs):
            original(instance, *args, **kwargs)
            if instance.network.fps != 30:
                raise ValueError("Original physics/network rate changed")
            instance.features = ExactFeatureCache(instance.features)
            caches.append(instance.features)
        prepared = PreparedDenoiser()
        # The CPU child consumes denoised inputs, so it constructs no CANO session.
        with instrument(source, output), contextlib.redirect_stdout(sys.stderr):
            spec = importlib.util.spec_from_file_location("isolated_author", source/"online_fgp+LIP_realtime_onnx.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            old_flow = OnnxBundle.flow_model
            OnnxBundle.flow_model = lambda bundle: None
            GlobalPoseTracker.__init__ = tracker
            module.RealtimeIMUDenoiser = lambda *args, **kwargs: prepared
            try:
                args = module.parse_args(["--onnx-dir", str(source/"models/onnx_s1"), "--onnx-threads", "1"])
                pipeline = module.RealtimePipeline(args)
            finally:
                OnnxBundle.flow_model = old_flow
                GlobalPoseTracker.__init__ = original
            from stage_tran_gp.backend import NativePhysics
            if not isinstance(pipeline.gp.tracker.physics, NativePhysics):
                raise RuntimeError("Require original native physics, no substituted worker")
            for task in Path('/proc/self/task').iterdir():
                try:
                    os.sched_setaffinity(int(task.name), ROLE_CPUS["cpu"])
                except ProcessLookupError:
                    pass
            bindings = {task.name: sorted(os.sched_getaffinity(int(task.name)))
                        for task in Path('/proc/self/task').iterdir()}
            if any(mask != ROLE_CPUS["cpu"] for mask in bindings.values()):
                raise RuntimeError("Own CPU child thread affinity mismatch")
            (output/"thread_bindings.json").write_text(json.dumps(bindings, indent=2)+"\n")
            identity = SimpleNamespace(calibrate=lambda raw, acc_scale: raw)
            connection.send(dict(kind="ready", pid=os.getpid(), tid=threading.get_native_id(),
                                 native=True, cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"]))
            previous = (0, -1)
            while True:
                if time.time() >= DEADLINE_UTC:
                    raise TimeoutError("Fixed three-hour deadline reached")
                if os.getppid() != owner:
                    raise RuntimeError("Own parent exited")
                if not connection.poll(.5):
                    continue
                packet = connection.recv()
                if isinstance(packet, str) and packet == "close":
                    break
                previous = decode(packet, previous, limit)
                began = time.monotonic_ns()
                with torch.inference_mode():
                    prepared.bind(torch.from_numpy(packet["acc"]), torch.from_numpy(packet["rot"]))
                    result = pipeline.step(packet["calibrated"], identity)
                    if result is None or prepared.value is not None:
                        raise ValueError("CPU author path did not consume one frame")
                    native = pipeline.gp.last_result
                    pose = native["pose"].numpy().copy()
                    root = native["translation"].numpy().copy()
                    if not valid_rotations(pose, 24) or not np.isfinite(root).all():
                        raise ValueError("Invalid native result")
                ended = time.monotonic_ns()
                count += 1
                if len(caches) != 1 or caches[0].hits != count:
                    raise RuntimeError("CPU feature/model accounting failed")
                hits = caches[0].hits
                connection.send(dict(kind="result", sequence=count, frame=packet["frame"],
                    pose=pose, root=root, cpu_started_ns=began, inferred_ns=ended,
                    stages=dict(pipeline.perf.frame), feature_hits=hits))
            status = "CPU_PROCESS_CLEAN_EXIT"
    except BaseException as exc:
        error = repr(exc)
        try:
            connection.send(dict(kind="error", error=error))
        except (EOFError, BrokenPipeError, OSError):
            pass
        raise
    finally:
        if pipeline is not None:
            pipeline.close()
        connection.close()
        (output/"status.json").write_text(json.dumps(dict(status=status, error=error,
            pid=os.getpid(), completed=count, feature_hits=hits, gpu_calls=0,
            hardware_commands=0, cpus=sorted(os.sched_getaffinity(0))), indent=2)+"\n")


class CpuClient:
    def __init__(self, source, output, limit, init_timeout=35):
        if limit not in (64, 900):
            raise ValueError("Unregistered frame budget")
        ctx = mp.get_context("spawn")
        self.connection, remote = ctx.Pipe()
        self.process = ctx.Process(target=child, args=(remote, str(source), str(output), limit, os.getpid()))
        self.count = self.hits = 0
        self.closed = False
        self.output = Path(output)
        self.rpc_bound = False
        self.process.start()
        remote.close()
        try:
            ready = self.receive(init_timeout)
            if ready.get("kind") != "ready" or ready.get("cuda_visible_devices") != "":
                raise RuntimeError("CPU initialization failed")
        except BaseException:
            self.close()
            raise

    def receive(self, timeout):
        if not self.connection.poll(timeout):
            raise TimeoutError("Bounded own CPU RPC timeout")
        value = self.connection.recv()
        if value.get("kind") == "error":
            raise RuntimeError(value["error"])
        return value

    def __call__(self, item):
        if not self.rpc_bound:
            os.sched_setaffinity(0, ROLE_CPUS["cpu"])
            (self.output.parent/"cpu_rpc_binding.json").write_text(json.dumps(dict(
                pid=os.getpid(), tid=threading.get_native_id(),
                cpus=sorted(os.sched_getaffinity(0))), indent=2)+"\n")
            self.rpc_bound = True
        row, calibrated, acc, rot = item
        rpc_began = time.monotonic_ns()
        self.connection.send(dict(sequence=self.count+1, frame=int(row["frame"]), calibrated=calibrated,
                                  acc=acc.detach().numpy().copy(), rot=rot.detach().numpy().copy()))
        reply = self.receive(5)
        received = time.monotonic_ns()
        if reply.get("kind") != "result" or reply["sequence"] != self.count+1 or reply["frame"] != row["frame"]:
            raise ValueError("Mismatched CPU response")
        self.count += 1
        self.hits = reply["feature_hits"]
        begin, end = reply["cpu_started_ns"], reply["inferred_ns"]
        row.update(cpu_started_ns=begin, inferred_ns=end, cpu_stage_ms=(end-begin)/1e6,
                   cpu_queue_ms=(begin-row["denoised_ns"])/1e6, fgp_ms=(end-row["started_ns"])/1e6,
                   cpu_rpc_ms=(received-rpc_began)/1e6, cpu_ipc_return_ms=(received-end)/1e6,
                   stages=reply["stages"])
        return row, reply["pose"], reply["root"]

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.is_alive():
                try:
                    self.connection.send("close")
                except (EOFError, BrokenPipeError, OSError):
                    pass
                self.process.join(2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(5)
        finally:
            self.connection.close()
