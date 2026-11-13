"""Ordered CPU and sender stages with a shared admission bound."""
from concurrent.futures import ThreadPoolExecutor
import threading


class BoundedPipeline:
    def __init__(self, process, send, capacity=2, close_sender=None):
        if capacity != 2:
            raise ValueError("This experiment admits at most two frames")
        self.process, self.send, self.close_sender = process, send, close_sender
        self.cpu = ThreadPoolExecutor(1, thread_name_prefix="fgp_cpu")
        self.sender = ThreadPoolExecutor(1, thread_name_prefix="human_sender")
        self.credit = threading.BoundedSemaphore(capacity)
        self.lock = threading.Lock()
        self.error, self.futures = None, []
        self.active = self.maximum = self.submitted = self.completed = 0
        self.closed = False

    def check(self):
        with self.lock:
            error = self.error
        if error is not None:
            raise RuntimeError("Pipeline stage failed") from error

    def reserve(self):
        if self.closed:
            raise RuntimeError("Pipeline closed")
        while not self.credit.acquire(timeout=.05):
            self.check()
        try:
            self.check()
        except BaseException:
            self.credit.release()
            raise
        with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)

    def release(self):
        with self.lock:
            self.active -= 1
        self.credit.release()

    def submit(self, item):
        self.check()
        future = self.cpu.submit(self.process, item)

        def deliver():
            try:
                result = self.send(future.result())
                with self.lock:
                    self.completed += 1
                return result
            except BaseException as exc:
                with self.lock:
                    if self.error is None:
                        self.error = exc
                raise
            finally:
                self.release()

        self.futures.append(self.sender.submit(deliver))
        self.submitted += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            for future in self.futures:
                future.result()
        finally:
            try:
                if self.close_sender:
                    self.sender.submit(self.close_sender).result()
            finally:
                self.cpu.shutdown(wait=True)
                self.sender.shutdown(wait=True)
        self.check()


class PreparedDenoiser:
    """Consume exactly one already denoised frame through the author CPU path."""
    window_size = 30
    acc_buffer = tuple(range(30))

    def __init__(self):
        self.value = None

    def bind(self, acc, rot):
        if self.value is not None:
            raise RuntimeError("Unconsumed prepared frame")
        self.value = acc, rot

    def step(self, acc, rot):
        if self.value is None:
            raise RuntimeError("Missing prepared denoiser output")
        value, self.value = self.value, None
        return value

    def reset(self):
        raise RuntimeError("This measured-input probe cannot reset a live producer")
