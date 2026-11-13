"""Static float32 I/O TensorRT session using the installed CUDA runtime."""
import ctypes as ct
from pathlib import Path
import sys

import numpy as np

GP_INPUTS = {"features": (1, 1, 267), "h": (3, 1, 512), "c": (3, 1, 512)}
GP_OUTPUTS = {"prediction": (1, 1, 9), "h_out": (3, 1, 512), "c_out": (3, 1, 512)}


def validate_gp_feed(feed):
    if set(feed) != set(GP_INPUTS):
        raise ValueError("Expected explicit GP features/h/c")
    result = {}
    for name, shape in GP_INPUTS.items():
        value = feed[name]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Invalid GP input " + name)
        result[name] = np.ascontiguousarray(value)
    return result


def validate_feed(feed):
    if set(feed) != {"x_t", "x_cond"}:
        raise ValueError("Expected exactly x_t and x_cond")
    result = {}
    for name, value in feed.items():
        if value.shape != (1, 30, 10, 9) or value.dtype != np.float32:
            raise ValueError("Expected float32 [1,30,10,9]")
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite model input")
        result[name] = np.ascontiguousarray(value)
    return result


class TrtSession:
    def __init__(self, engine_path, *, gp=False):
        self.gp = gp
        # Only TensorRT is taken from the OS installation; keep venv numpy/torch.
        sys.path.append("/usr/lib/python3.10/dist-packages")
        import tensorrt as trt
        self.version = trt.__version__
        self.cuda = ct.CDLL("/usr/local/cuda/lib64/libcudart.so")
        signatures = {
            "cudaMalloc": [ct.POINTER(ct.c_void_p), ct.c_size_t],
            "cudaFree": [ct.c_void_p],
            "cudaStreamCreateWithFlags": [ct.POINTER(ct.c_void_p), ct.c_uint],
            "cudaStreamDestroy": [ct.c_void_p],
            "cudaStreamSynchronize": [ct.c_void_p],
            "cudaMemcpyAsync": [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int, ct.c_void_p],
        }
        for name, signature in signatures.items():
            fn = getattr(self.cuda, name)
            fn.argtypes, fn.restype = signature, ct.c_int
        self.stream = ct.c_void_p()
        self.pointers, self.inputs, self.outputs = {}, {}, {}
        self.recorded = []
        self.closed = False
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT engine deserialization failed")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT context creation failed")
        try:
            self.check(self.cuda.cudaStreamCreateWithFlags(ct.byref(self.stream), 1))
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                shape = tuple(self.engine.get_tensor_shape(name))
                dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                if any(d <= 0 for d in shape) or dtype != np.float32:
                    raise ValueError("Expected static float32 TensorRT I/O")
                array = np.zeros(shape, dtype=dtype)
                pointer = ct.c_void_p()
                self.check(self.cuda.cudaMalloc(ct.byref(pointer), array.nbytes))
                self.pointers[name] = pointer
                if not self.context.set_tensor_address(name, pointer.value):
                    raise RuntimeError("Tensor address rejected")
                target = self.inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else self.outputs
                target[name] = array
            if gp:
                validate_gp_feed(self.inputs)
                if set(self.outputs) != set(GP_OUTPUTS) or any(self.outputs[n].shape != s for n, s in GP_OUTPUTS.items()):
                    raise ValueError("Unexpected GP output contract")
            else:
                if set(self.inputs) != {"x_t", "x_cond"} or set(self.outputs) != {"residual"}:
                    raise ValueError("Unexpected engine I/O names")
                validate_feed(self.inputs)
                if self.outputs["residual"].shape != (1, 1, 10, 9):
                    raise ValueError("Unexpected residual output shape")
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(code):
        if code != 0:
            raise RuntimeError("CUDA runtime returned " + str(code))

    def run(self, names, feed):
        ordered = list(GP_OUTPUTS) if self.gp else ["residual"]
        if self.closed or (names != ordered and not (self.gp and names is None)) or len(self.recorded) >= 32:
            raise RuntimeError("Invalid request or bounded 32-call limit reached")
        feed = validate_gp_feed(feed) if self.gp else validate_feed(feed)
        for name, value in feed.items():
            self.check(self.cuda.cudaMemcpyAsync(self.pointers[name], value.ctypes.data,
                                                value.nbytes, 1, self.stream))
        self.enqueue()
        for name in ordered:
            output = self.outputs[name]
            self.check(self.cuda.cudaMemcpyAsync(output.ctypes.data, self.pointers[name],
                                                output.nbytes, 2, self.stream))
        self.check(self.cuda.cudaStreamSynchronize(self.stream))
        if not all(np.isfinite(self.outputs[n]).all() for n in ordered):
            raise ValueError("Nonfinite TensorRT output")
        results = [self.outputs[n].copy() for n in ordered]
        self.recorded.append(({k: v.copy() for k, v in feed.items()}, results if self.gp else results[0]))
        return results

    def enqueue(self):
        if not self.context.execute_async_v3(stream_handle=self.stream.value):
            raise RuntimeError("TensorRT enqueue failed")

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.stream.value:
            self.cuda.cudaStreamSynchronize(self.stream)
        self.context = None
        for pointer in self.pointers.values():
            self.cuda.cudaFree(pointer)
        if self.stream.value:
            self.cuda.cudaStreamDestroy(self.stream)
        self.engine = self.runtime = None
