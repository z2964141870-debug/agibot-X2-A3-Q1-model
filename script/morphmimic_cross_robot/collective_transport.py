"""Synchronous host-staged Gloo transport for pinned upstream collective calls."""

from contextlib import contextmanager


MODULES = frozenset({
    "morphmimic.train.distributed",
    "morphmimic.train.multi_robot",
    "morphmimic.train.normalizers",
    "morphmimic.train.trainer",
})
FORWARDED = frozenset({
    "ReduceOp", "get_rank", "get_world_size", "is_available", "is_initialized",
})


class HostCollectives:
    """Keep all reduction arithmetic in torch.distributed on CPU tensors.

    This changes transport only. Upstream retains gradient averaging, clipping,
    optimizer order, detached latent centers and robot-specific buffers.
    """

    def __init__(self, api):
        import torch

        if not api.is_initialized() or str(api.get_backend()) != "gloo":
            raise RuntimeError("Host transport requires an initialized Gloo group")
        self.torch = torch
        self.api = api
        self.calls = {"broadcast": 0, "all_reduce": 0, "all_gather": 0,
                      "all_gather_object": 0}
        self.staged_input_bytes = 0

    def __getattr__(self, name):
        if name in FORWARDED:
            return getattr(self.api, name)
        raise AttributeError("Unaudited distributed operation: " + name)

    def _host(self, tensor):
        self._check_tensor(tensor)
        self.staged_input_bytes += tensor.numel() * tensor.element_size()
        return tensor.detach().to(device="cpu", copy=True).contiguous()

    def _check_tensor(self, tensor):
        if tensor.layout != self.torch.strided or tensor.device.type not in {"cpu", "cuda"}:
            raise ValueError("Only dense CPU/CUDA tensors have been audited")

    def all_reduce(self, tensor, op=None):
        op = self.api.ReduceOp.SUM if op is None else op
        if op not in (self.api.ReduceOp.SUM, self.api.ReduceOp.MIN, self.api.ReduceOp.MAX):
            raise ValueError("Only upstream SUM/MIN/MAX reductions have been audited")
        with self.torch.no_grad():
            host = self._host(tensor)
            self.api.all_reduce(host, op=op)
            tensor.copy_(host)
        self.calls["all_reduce"] += 1

    def broadcast(self, tensor, src):
        with self.torch.no_grad():
            host = self._host(tensor)
            self.api.broadcast(host, src=src)
            tensor.copy_(host)
        self.calls["broadcast"] += 1

    def all_gather(self, tensor_list, tensor):
        if len(tensor_list) != self.api.get_world_size():
            raise ValueError("One destination tensor per rank is required")
        if any(t.shape != tensor.shape or t.dtype != tensor.dtype for t in tensor_list):
            raise ValueError("Upstream gathered tensors must have matching shape/dtype")
        for target in tensor_list:
            self._check_tensor(target)
        with self.torch.no_grad():
            host = self._host(tensor)
            outputs = [self.torch.empty_like(host) for _ in tensor_list]
            self.api.all_gather(outputs, host)
            for target, source in zip(tensor_list, outputs):
                target.copy_(source)
        self.calls["all_gather"] += 1

    def all_gather_object(self, object_list, obj):
        # Gloo's object transport uses CPU; objects are trusted, local rank state.
        self.api.all_gather_object(object_list, obj)
        self.calls["all_gather_object"] += 1


@contextmanager
def installed_transport(modules, api=None):
    """Replace only these modules' dist aliases, then restore them on every exit."""
    import torch.distributed as distributed

    api = distributed if api is None else api
    modules = list(modules)
    if not modules or len({id(module) for module in modules}) != len(modules):
        raise ValueError("Pass a nonempty, distinct list of upstream modules")
    if any(module.__name__ not in MODULES or module.dist is not api for module in modules):
        raise ValueError("Only unmodified aliases of the four pinned modules are allowed")
    bridge = HostCollectives(api)
    for module in modules:
        module.dist = bridge
    try:
        yield bridge
    finally:
        for module in modules:
            module.dist = api
