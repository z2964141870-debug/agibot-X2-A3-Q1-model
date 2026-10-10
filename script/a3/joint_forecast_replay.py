"""Joint-only causal ablation; future anchor orientations remain privileged."""

import numpy as np


MODES = ("oracle_aligned", "hold", "cv", "smooth_cv", "E12", "E14")
HORIZONS = np.arange(1, 11, dtype=float) / 50


def load_predictor(path):
    with np.load(path, allow_pickle=False) as stored:
        model = {key: stored[key].copy() for key in ("mean", "std", "weights", "bias")}
        if (stored["history_frames"].item() != 6 or stored["source_fps"].item() != 30 or
                not np.allclose(stored["horizons_s"], HORIZONS, atol=1e-12, rtol=0)):
            raise ValueError("Unsupported predictor timing contract")
    expected = dict(mean=(145,), std=(145,), weights=(145, 290), bias=(290,))
    if any(model[key].shape != shape or not np.isfinite(model[key]).all()
           for key, shape in expected.items()) or np.any(model["std"] <= 0):
        raise ValueError("Invalid predictor parameters")
    return model


def forecast_window(arrived, times, query, mode, model=None, end_seen=False):
    """Only arrived observations are accepted, including startup/EOF handling."""
    q, times, query = np.asarray(arrived), np.asarray(times), np.asarray(query)
    if (q.ndim != 2 or q.shape[1] != 29 or len(q) != len(times) or not len(q) or
            not np.isfinite(q).all() or not np.isfinite(times).all() or
            np.any(np.diff(times) <= 0) or not np.isfinite(query).all()):
        raise ValueError("Invalid arrived joint history")
    if mode not in MODES[1:]:
        raise ValueError("Not a causal joint mode")
    if query.max() > times[-1] + .20 + 1e-9 and not end_seen:
        raise ValueError("Prediction would exceed its trained 200ms horizon")
    history = q[-6:]
    current = history[-1]
    if end_seen or mode == "hold" or len(q) < 6:
        future = np.repeat(current[None], 10, axis=0)
    elif mode == "cv":
        velocity = (q[-1] - q[-2]) / (times[-1] - times[-2])
        future = current + HORIZONS[:, None] * velocity
    elif mode == "smooth_cv":
        t = times[-6:] - times[-1]
        design = np.column_stack((np.ones(6), t))
        intercept, velocity = np.linalg.lstsq(design, history, rcond=None)[0]
        current = intercept
        future = current + HORIZONS[:, None] * velocity
    else:
        if model is None:
            raise ValueError("Missing registered predictor")
        x = (history[:-1] - history[-1]).reshape(-1)
        offset = ((x - model["mean"]) / model["std"]) @ model["weights"] + model["bias"]
        future = current + offset.reshape(10, 29)
    support_t = np.concatenate((times[:-1], times[-1] + np.arange(11) / 50))
    support_q = np.concatenate((q[:-1], current[None], future))
    result = np.column_stack([np.interp(query, support_t, support_q[:, joint]) for joint in range(29)])
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite joint forecast")
    return result


class JointForecastReplay:
    def __init__(self, sim, mode, model_path=None, noise_std=0., seed=0):
        if mode not in MODES or not np.isfinite(noise_std) or noise_std < 0:
            raise ValueError("Invalid joint ablation settings")
        if (mode in ("E12", "E14")) != (model_path is not None):
            raise ValueError("Only learned modes require a predictor")
        if mode == "oracle_aligned" and noise_std:
            raise ValueError("Oracle comparator uses clean privileged references")
        self.sim, self.mode, self.noise_std, self.seed = sim, mode, noise_std, seed
        self.model = load_predictor(model_path) if model_path else None
        self.sources, self.rows, self.errors = {}, [], []
        self.last_target_frame = 0
        self.original_load = sim.load_loop_motion_reference
        self.original_record = sim.LoopSimRunner._record_metrics_step

    def register(self, reference, permutation, source_fps, stride):
        if source_fps != 30:
            raise ValueError("Forecasts were trained for 30Hz arrived samples")
        permutation = np.asarray(permutation)
        if not np.array_equal(np.sort(permutation), np.arange(29)):
            raise ValueError("Joint permutation is not bijective")
        if not np.array_equal(reference.dof_il, reference.dof_mj29[:, permutation]):
            raise ValueError("Runtime joint order differs from the registered permutation")
        _, _, q, _ = self.sim.load_a3_flat_csv(reference.path, source_fps=source_fps, frame_stride=stride)
        if q.shape[1] != 29 or not np.isfinite(q).all():
            raise ValueError("Invalid raw joint reference")
        noise = np.random.default_rng(self.seed).normal(0, self.noise_std, q.shape)
        self.sources[str(reference.path)] = (np.arange(len(q)) / source_fps, q + noise, permutation)

    def build(self, reference, ref_frame, anchor, on_end="hold_last", frame_skip=1,
              history_frames=0, valid_future_frames=None, zero_pad_invalid_frames=False):
        if (on_end != "hold_last" or frame_skip != 1 or history_frames or
                valid_future_frames is not None or zero_pad_invalid_frames):
            raise ValueError("Ablation requires the standard ten-frame A3-fast window")
        times, observations, permutation = self.sources[str(reference.path)]
        arrival = ref_frame / 50
        latest = min(int(np.searchsorted(times, arrival + 1e-10, side="right") - 1), len(times) - 1)
        if latest < 0:
            raise ValueError("No source observation has arrived")
        end_seen = latest == len(times) - 1
        base = ref_frame if end_seen else int(np.floor((times[latest] + 1e-10) * 50))
        base = min(base, reference.num_frames - 1)
        self.last_target_frame = base
        indices = np.minimum(base + np.arange(11), reference.num_frames - 1)
        if self.mode == "oracle_aligned":
            pos = reference.dof_il[indices[:10]]
            vel = reference.dof_vel_il[indices[:10]]
        else:
            query = base / 50 + np.arange(11) / 50
            pos_all = forecast_window(observations[:latest+1], times[:latest+1], query,
                                      self.mode, self.model, end_seen)[:, permutation]
            pos, vel = pos_all[:10], np.diff(pos_all, axis=0) * 50
        # Upstream packs all positions followed by all velocities, then reshapes.
        command = np.concatenate((pos.reshape(-1), vel.reshape(-1))).reshape(10, 58)
        orientation = self.sim.root_ori_diff_6d(anchor, reference.anchor_quat_wxyz[indices[:10]])
        value = np.concatenate((command, orientation), axis=1).astype(np.float32)
        if value.shape != (10, 64) or not np.isfinite(value).all():
            raise ValueError("Invalid A3-fast ablation input")
        error = pos - reference.dof_il[indices[:10]]
        self.errors.append([float(np.sum(error**2)), int(error.size), float(np.max(np.abs(error)))])
        self.rows.append([arrival, base / 50, arrival-base/50, times[latest], latest,
                          int(end_seen), max(0., base/50+.2-times[latest])])
        return value

    def install(self):
        replay = self

        def load(model, runtime, solver, path, **kwargs):
            reference = replay.original_load(model, runtime, solver, path, **kwargs)
            replay.register(reference, runtime.mapping.mj29_to_il,
                            kwargs.get("csv_source_fps", 30), kwargs.get("csv_frame_stride", 4))
            return reference

        def record(runner, policy_step, ref_frame, result):
            replay.original_record(runner, policy_step, replay.last_target_frame, result)

        self.sim.load_loop_motion_reference = load
        self.sim.build_encoder_input = self.build
        self.sim.LoopSimRunner._record_metrics_step = record

    def save(self, path):
        if not self.rows:
            raise ValueError("No actual policy commands captured")
        np.savez_compressed(path, rows=np.asarray(self.rows), errors=np.asarray(self.errors),
            columns=np.asarray(["arrival_s", "target_s", "age_s", "latest_joint_read_s",
                                "latest_joint_index", "eof_seen", "prediction_horizon_s"]),
            mode=np.asarray(self.mode), noise_std_rad=self.noise_std, seed=self.seed,
            scope=np.asarray("JOINT_ONLY_FUTURE_ORIENTATION_ORACLE_NOT_FULL_CAUSAL_TELEOP"))
