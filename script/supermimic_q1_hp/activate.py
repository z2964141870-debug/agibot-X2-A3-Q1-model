"""Launch the isolated runtime with existing CUDA libraries on its loader path."""

import os
from pathlib import Path
import sys

root = Path(os.environ.get("YUANQI_ROOT", "/opt/data/private/ZHY"))
libraries = []
for path in sys.path:
    parent = Path(path) / "nvidia"
    if parent.is_dir():
        libraries.extend(str(p) for p in sorted(parent.glob("*/lib")) if p.is_dir())
environment = os.environ.copy()
environment["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys([*libraries, environment.get("LD_LIBRARY_PATH", "")]))
cache = root / "data/cache/supermimic_q1_hp_20261026"
for key, directory in (("XDG_CACHE_HOME", cache), ("TORCHINDUCTOR_CACHE_DIR", cache / "inductor"), ("WARP_CACHE_PATH", cache / "warp")):
    directory.mkdir(parents=True, exist_ok=True)
    environment[key] = str(directory)
environment.update({"YUANQI_ROOT": str(root), "MUJOCO_GL": "egl", "PYTORCH_ALLOC_CONF": "expandable_segments:True", "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "1"})
os.execvpe(sys.executable, [sys.executable, *sys.argv[1:]], environment)
