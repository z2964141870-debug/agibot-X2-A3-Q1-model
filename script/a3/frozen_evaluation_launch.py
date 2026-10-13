"""Preserve subprocess behavior while selecting the passive frozen entry."""

from types import SimpleNamespace


class LaunchProxy:
    def __init__(self, base, spawn=None):
        self.base = base
        self.spawn = spawn or base.Popen

    def __getattr__(self, name):
        return getattr(self.base, name)

    def Popen(self, argv, **kwargs):
        argv = ['script.a3.smpl_frozen_entry' if word == 'script.a3.sim_entry' else word for word in argv]
        kwargs['start_new_session'] = False
        return self.spawn(argv, **kwargs)


def arguments(output, logs, checkpoint, references):
    return SimpleNamespace(output=output, logs=logs, checkpoint=checkpoint, motion=references,
        reference_fps=30., frame_stride=1, max_policy_steps=None, action_delay_ms=0.,
        reference_buffer_ms=0., capture_inputs=True, capture_count=100000, summarize_only=False,
        joint_reference_mode=None, joint_predictor=None, joint_noise_std=0., joint_noise_seed=0,
        causal_orientation=False)
