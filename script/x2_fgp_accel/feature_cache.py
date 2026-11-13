"""Reuse only exactly repeated inputs to the author's stateless PoseFeatures."""
import numpy as np

FEATURE_SOURCES = {
    "features.py": "8b6a4a67d4a4d054b139ceebc36ba8501e84dcebb0ca0768a7c7fb621e89a207",
    "tracker.py": "324e93beeb21ab86c9ea8f9dfb32031cb127161b3bdb35ea3e5734c94d1ccc58",
}


class ExactFeatureCache:
    def __init__(self, original):
        self.original = original
        self.previous = self.values = None
        self.calls = self.hits = self.misses = 0

    def __call__(self, pose, acc, gyro):
        self.calls += 1
        arrays = tuple(np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)
                       for value in (pose, acc, gyro))
        if self.previous is not None and all(np.isfinite(a).all() for a in arrays) and all(
                a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
                for a, b in zip(arrays, self.previous)):
            self.hits += 1
            return tuple(value.clone() for value in self.values)
        self.misses += 1
        result = self.original(pose, acc, gyro)
        self.previous = tuple(value.copy() for value in arrays)
        self.values = tuple(value.clone() for value in result)
        return result
