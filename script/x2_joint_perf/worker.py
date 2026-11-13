"""Process-local tuning around the frozen joint measurement worker."""
import contextlib
import json
import os
from pathlib import Path
import sys


def settings():
    config = json.loads(os.environ["X2_JOINT_PERF_CONFIG"])
    if config not in (
        dict(variant="control", threads=4, spinning=True, batched_fk=False),
        dict(variant="sleep1", threads=1, spinning=False, batched_fk=False),
        dict(variant="sleep1_fk", threads=1, spinning=False, batched_fk=True),
        dict(variant="all_sleep1_fk", threads=1, spinning=False, batched_fk=True),
    ):
        raise ValueError("Unregistered performance variant")
    return config


def main():
    config = settings()
    mode = sys.argv[1]
    if mode == "policy" or config["variant"] == "all_sleep1_fk":
        import onnxruntime as ort
        original = ort.InferenceSession

        def configured_session(model, sess_options=None, *args, **kwargs):
            options = sess_options or ort.SessionOptions()
            options.intra_op_num_threads = config["threads"]
            options.inter_op_num_threads = 1
            if not config["spinning"]:
                options.add_session_config_entry("session.intra_op.allow_spinning", "0")
                options.add_session_config_entry("session.inter_op.allow_spinning", "0")
            return original(model, options, *args, **kwargs)

        ort.InferenceSession = configured_session
    if mode == "fgp" and config["batched_fk"]:
        from .adapter import BatchedSmplAdapter
        from script.x2_morph_human import core
        core.SmplAdapter = BatchedSmplAdapter
    if mode == "fgp" and config["variant"] == "all_sleep1_fk":
        import torch
        original_threads = torch.set_num_threads
        torch.set_num_threads = lambda ignored: original_threads(1)
    from script.x2_joint_latency.worker import main as original_main
    original_main()


if __name__ == "__main__":
    main()
