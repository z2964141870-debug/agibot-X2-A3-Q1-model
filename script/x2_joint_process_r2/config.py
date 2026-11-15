"""Fixed wall deadline and source identity for this independent candidate."""
import hashlib
from pathlib import Path

DEADLINE_UTC = 1794730350
ROLE_CPUS = {"gpu": [2, 3], "sender": [0], "cpu": [4, 5]}
EXECUTION = ("__init__.py", "config.py", "cpu.py", "worker.py", "run.py")


def sources():
    root = Path(__file__).parent
    return {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in EXECUTION}
