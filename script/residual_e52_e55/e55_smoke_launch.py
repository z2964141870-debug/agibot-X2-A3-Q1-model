"""E55 channel-flow smoke: evaluate_mujoco chain + LaunchProxy word swap to e55 entry."""
import sys
from pathlib import Path
from types import SimpleNamespace
from script.a3 import evaluate_mujoco as evaluation
from script.a3.frozen_evaluation_launch import LaunchProxy


class E55Proxy(LaunchProxy):
    def Popen(self, argv, **kwargs):
        argv = ["script.a3.e55_frozen_entry" if w == "script.a3.sim_entry" else w for w in argv]
        kwargs["start_new_session"] = False
        return self.spawn(argv, **kwargs)


evaluation.subprocess = E55Proxy(evaluation.subprocess)
args = SimpleNamespace(
    output=Path("/opt/data/private/hechengzhu/ZHY/data/experiments/a3_zhy_residual_p1_20261019_E55/smoke/out"), logs=Path("/opt/data/private/hechengzhu/ZHY/logs/e55_smoke"),
    checkpoint=Path("/opt/data/private/hechengzhu/ZHY/data/experiments/a3_zhy_residual_p1_20261019_E55/checkpoints/e55_smoke_ckpt.pt"),
    motion=Path("/opt/data/private/hechengzhu/ZHY/data/experiments/a3_zhy_cloth_resource_20261013_E38/official_references/029_rotate_place_slow.csv"),
    reference_fps=30., frame_stride=4, max_policy_steps=150, action_delay_ms=0.,
    reference_buffer_ms=0., capture_inputs=False, capture_count=None, summarize_only=False,
    joint_reference_mode=None, joint_predictor=None, joint_noise_std=0., joint_noise_seed=0,
    causal_orientation=False)
ok = evaluation.evaluate_explicit(args)
print("SMOKE_EVAL_OK" if ok else "SMOKE_EVAL_FAIL")
raise SystemExit(0 if ok else 1)
