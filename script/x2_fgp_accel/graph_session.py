"""Capture the unchanged, static TensorRT execution on its own CUDA stream."""
import ctypes as ct

from .trt_session import TrtSession


class CudaGraphSession(TrtSession):
    def __init__(self, engine_path):
        self.graph, self.graph_exec = ct.c_void_p(), ct.c_void_p()
        super().__init__(engine_path)
        signatures = {
            "cudaStreamBeginCapture": [ct.c_void_p, ct.c_int],
            "cudaStreamEndCapture": [ct.c_void_p, ct.POINTER(ct.c_void_p)],
            "cudaGraphInstantiateWithFlags": [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_ulonglong],
            "cudaGraphLaunch": [ct.c_void_p, ct.c_void_p],
            "cudaGraphExecDestroy": [ct.c_void_p],
            "cudaGraphDestroy": [ct.c_void_p],
        }
        try:
            for name, signature in signatures.items():
                fn = getattr(self.cuda, name)
                fn.argtypes, fn.restype = signature, ct.c_int
            # Flush TensorRT lazy state before capture; no downstream human state advances.
            for name, array in self.inputs.items():
                self.check(self.cuda.cudaMemcpyAsync(self.pointers[name], array.ctypes.data,
                                                    array.nbytes, 1, self.stream))
            super().enqueue()
            self.check(self.cuda.cudaStreamSynchronize(self.stream))
            self.check(self.cuda.cudaStreamBeginCapture(self.stream, 1))
            try:
                super().enqueue()
            finally:
                self.check(self.cuda.cudaStreamEndCapture(self.stream, ct.byref(self.graph)))
            self.check(self.cuda.cudaGraphInstantiateWithFlags(ct.byref(self.graph_exec), self.graph, 0))
        except BaseException:
            self.close()
            raise

    def enqueue(self):
        self.check(self.cuda.cudaGraphLaunch(self.graph_exec, self.stream))

    def close(self):
        if getattr(self, "closed", True):
            return
        if self.stream.value:
            self.cuda.cudaStreamSynchronize(self.stream)
        if self.graph_exec.value:
            self.cuda.cudaGraphExecDestroy(self.graph_exec)
            self.graph_exec = ct.c_void_p()
        if self.graph.value:
            self.cuda.cudaGraphDestroy(self.graph)
            self.graph = ct.c_void_p()
        super().close()
