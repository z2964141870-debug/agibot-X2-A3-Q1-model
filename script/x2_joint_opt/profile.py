"""Bounded wall/thread timers and CUDA stream events, without extra forwards."""
from contextlib import contextmanager, ExitStack
import ctypes as ct
import json
import threading
import time
from unittest.mock import patch


class Recorder:
    def __init__(self, limit=20000):
        self.limit, self.rows = limit, []
        self.lock = threading.Lock()

    def add(self, stage, began, cpu, **extra):
        row = dict(stage=stage, began_ns=began, ended_ns=time.monotonic_ns(),
                   thread_cpu_ms=(time.thread_time_ns()-cpu)/1e6,
                   tid=threading.get_native_id(), **extra)
        row["wall_ms"] = (row["ended_ns"]-began)/1e6
        with self.lock:
            if len(self.rows) >= self.limit:
                raise RuntimeError("Bounded profile capacity exhausted")
            self.rows.append(row)

    def wrap(self, stage, function):
        def measured(*args, **kwargs):
            wall, cpu = time.monotonic_ns(), time.thread_time_ns()
            try:
                return function(*args, **kwargs)
            finally:
                self.add(stage, wall, cpu)
        return measured

    def save(self, path):
        with path.open("x") as stream:
            for row in self.rows:
                stream.write(json.dumps(row, allow_nan=False)+"\n")


def timed_graph_type(original, recorder):
    class TimedGraph(original):
        def __init__(self, *args, **kwargs):
            self.events, self.in_forward = [], False
            super().__init__(*args, **kwargs)
            cuda = self.cuda
            declarations = {
                "cudaEventCreate": [ct.POINTER(ct.c_void_p)],
                "cudaEventRecord": [ct.c_void_p, ct.c_void_p],
                "cudaEventElapsedTime": [ct.POINTER(ct.c_float), ct.c_void_p, ct.c_void_p],
                "cudaEventDestroy": [ct.c_void_p],
            }
            try:
                for name, signature in declarations.items():
                    fn = getattr(cuda, name)
                    fn.argtypes, fn.restype = signature, ct.c_int
                for _ in range(4):
                    event = ct.c_void_p()
                    self.check(cuda.cudaEventCreate(ct.byref(event)))
                    self.events.append(event)
            except BaseException:
                self.close()
                raise
            session = self

            class Calls:
                def __getattr__(self, name):
                    return getattr(cuda, name)

                def cudaMemcpyAsync(self, *args):
                    wall, cpu = time.monotonic_ns(), time.thread_time_ns()
                    result = cuda.cudaMemcpyAsync(*args)
                    if session.in_forward:
                        recorder.add("cuda_memcpy_host", wall, cpu, bytes=args[2], direction=args[3])
                    return result

                def cudaStreamSynchronize(self, *args):
                    if session.in_forward:
                        session.check(cuda.cudaEventRecord(session.events[3], session.stream))
                    wall, cpu = time.monotonic_ns(), time.thread_time_ns()
                    result = cuda.cudaStreamSynchronize(*args)
                    if session.in_forward:
                        recorder.add("cuda_sync_host", wall, cpu)
                    return result
            self.cuda = Calls()

        def enqueue(self):
            if not self.in_forward:
                return super().enqueue()
            self.check(self.cuda.cudaEventRecord(self.events[1], self.stream))
            wall, cpu = time.monotonic_ns(), time.thread_time_ns()
            result = super().enqueue()
            recorder.add("cuda_graph_enqueue_host", wall, cpu)
            self.check(self.cuda.cudaEventRecord(self.events[2], self.stream))
            return result

        def run(self, *args, **kwargs):
            wall, cpu = time.monotonic_ns(), time.thread_time_ns()
            self.in_forward = True
            try:
                self.check(self.cuda.cudaEventRecord(self.events[0], self.stream))
                result = super().run(*args, **kwargs)
                spans = {}
                for name, a, b in (("h2d_stream_span_ms", 0, 1), ("graph_stream_span_ms", 1, 2),
                                   ("d2h_stream_span_ms", 2, 3)):
                    value = ct.c_float()
                    self.check(self.cuda.cudaEventElapsedTime(ct.byref(value), self.events[a], self.events[b]))
                    spans[name] = value.value
                recorder.add("cano_session", wall, cpu, **spans)
                return result
            finally:
                self.in_forward = False

        def close(self):
            if getattr(self, "stream", None) is not None and self.stream.value:
                self.cuda.cudaStreamSynchronize(self.stream)
            for event in self.events:
                self.cuda.cudaEventDestroy(event)
            self.events = []
            super().close()
    return TimedGraph


@contextmanager
def instrument(source, output):
    from script.x2_morph_human.clotho_fgp_runner import prepare_clotho_environment, ClothoCalibrator
    prepare_clotho_environment(source)
    from script.x2_fgp_accel import graph_session
    import onnx_inference
    from stage_tran_gp import features, onnx_network, backend, tracker
    rec = Recorder()
    targets = ((ClothoCalibrator, "calibrate_ordered_10x7", "imu_calibration"),
               (onnx_inference.OnnxFlowModel, "__call__", "cano_adapter"),
               (onnx_inference.OnnxLipRunner, "forward", "lip_forward"),
               (features.PoseFeatures, "__call__", "gp_features_miss"),
               (onnx_network.OnnxGPNetwork, "step", "gp_forward"),
               (backend.NativePhysics, "update", "native_solver"),
               (backend.WorkerPhysics, "update", "worker_solver_rpc"),
               (tracker.GlobalPoseTracker, "update", "gp_tracker_inclusive"))
    with ExitStack() as stack:
        stack.enter_context(patch.object(graph_session, "CudaGraphSession",
                                        timed_graph_type(graph_session.CudaGraphSession, rec)))
        for owner, name, stage in targets:
            stack.enter_context(patch.object(owner, name, rec.wrap(stage, getattr(owner, name))))
        try:
            yield
        finally:
            rec.save(output/"profile.jsonl")
