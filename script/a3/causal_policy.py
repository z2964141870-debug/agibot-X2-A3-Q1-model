"""Causal reference-window hook for the fixed MuJoCo policy simulator."""

import numpy as np

from script.a3.reference_contract import causal_window


class CausalPolicyReplay:
    def __init__(self, sim, source_fps, stride, nominal_delay_ms):
        if nominal_delay_ms < 180:
            raise ValueError("A3-fast requires at least 180ms reference coverage")
        self.sim = sim
        self.source_fps = source_fps
        self.stride = stride
        self.nominal_delay_s = nominal_delay_ms / 1000
        # The last velocity token is a forward difference and needs one more policy frame.
        self.prefill_s = max(self.nominal_delay_s, .20)
        self.original = sim.build_encoder_input
        self.original_record = sim.LoopSimRunner._record_metrics_step
        self.sources = {}
        self.rows = []
        self.last_target_frame = 0

    def build(self, reference, ref_frame, anchor, on_end="hold_last", frame_skip=1,
              history_frames=0, valid_future_frames=None, zero_pad_invalid_frames=False):
        if frame_skip != 1 or history_frames or valid_future_frames is not None or zero_pad_invalid_frames:
            raise ValueError("This replay tests the fixed standard A3-fast window")
        key = str(reference.path)
        if key not in self.sources:
            root, _, _, _ = self.sim.load_a3_flat_csv(
                reference.path, source_fps=self.source_fps, frame_stride=self.stride)
            self.sources[key] = np.arange(len(root)) / self.source_fps
        times = self.sources[key]
        arrival = ref_frame / 50 + self.prefill_s
        window = causal_window(times, arrival, nominal_delay=self.prefill_s, lookahead_s=.20)
        if window is None:
            raise ValueError("Buffer has not been filled")
        base = min(int(round(window["base"] * 50)), reference.num_frames - 1)
        end_seen = arrival >= times[-1] - 1e-9
        if end_seen:
            # EOF is an arrived event; known terminal samples can now be held like the offline runner.
            base = min(max(0, int(np.floor((arrival - self.prefill_s + 1e-10) * 50))),
                       reference.num_frames - 1)
        last_read_frame = min(base + 10, reference.num_frames - 1)
        last_read_time = last_read_frame / 50
        upper = min(int(np.searchsorted(times, last_read_time - 1e-10)), len(times) - 1)
        if times[upper] > arrival + 1e-9:
            raise ValueError("Velocity/interpolation attempted to read a future source frame")
        self.last_target_frame = base
        value = self.original(reference, base, anchor, on_end, frame_skip, history_frames,
                              valid_future_frames, zero_pad_invalid_frames)
        self.rows.append([ref_frame / 50, arrival, base / 50, arrival - base / 50,
                          times[upper], upper, int(end_seen)])
        return value

    def install(self):
        replay = self

        def record(runner, policy_step, ref_frame, step_result):
            replay.original_record(runner, policy_step, replay.last_target_frame, step_result)

        self.sim.build_encoder_input = self.build
        self.sim.LoopSimRunner._record_metrics_step = record

    def save(self, path):
        rows = np.asarray(self.rows)
        if not len(rows):
            raise ValueError("No actual policy windows recorded")
        np.savez_compressed(path, rows=rows,
            columns=np.asarray(["policy_time_s", "arrival_time_s", "target_time_s", "age_s",
                                "latest_source_read_s", "latest_source_index", "end_of_stream_seen"]),
            nominal_buffer_ms=self.nominal_delay_s * 1000,
            startup_prefill_ms=self.prefill_s * 1000,
            scope=np.asarray("SIMULATED_ARRIVALS_WITH_FORWARD_VELOCITY_GUARD"))
