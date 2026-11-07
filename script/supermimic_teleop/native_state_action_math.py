"""Contracts for a64-call saved-action check of the isolated CPU converter."""

from contextlib import contextmanager
from unittest.mock import patch

import numpy as np


CALLS_PER_CLIP = 32
ATOL = RTOL = 2e-5


def saved_prefix(actual, actions, cpu_actual):
    n = CALLS_PER_CLIP
    if (actual.ndim != 2 or actual.shape[1] != 79 or actions.shape != (len(actual)-1, 29)
            or cpu_actual.ndim != 2 or cpu_actual.shape[1] != 79 or min(len(actual), len(cpu_actual)) < n
            or len(actions) < n
            or not np.isfinite(actual[:n]).all() or not np.isfinite(actions[:n]).all()
            or not np.array_equal(actual[:n], cpu_actual[:n])):
        raise ValueError('Require the same saved frame-zero prefix and finite29-slot actions')
    return actual[:n].copy(), actions[:n].copy()


def action_parity(value, expected, present):
    value, expected, present = np.asarray(value), np.asarray(expected), np.asarray(present, dtype=bool)
    if (value.shape != (29,) or expected.shape != (29,) or present.shape != (29,)
            or not np.isfinite(value).all() or not np.isfinite(expected).all()
            or np.count_nonzero(value[~present]) or np.count_nonzero(expected[~present])
            or not np.allclose(value, expected, atol=ATOL, rtol=RTOL)):
        raise ValueError('Saved canonical action parity or missing-slot contract failed')
    return float(np.max(np.abs(value-expected)))


@contextmanager
def isolated_converter(module, robot, spec, candidate):
    def convert(request_robot, state, request_spec):
        if request_robot is not robot or request_spec is not spec:
            raise ValueError('Converter is bound to its isolated robot and specification')
        return candidate(state)
    with patch.object(module, 'native_state', side_effect=convert):
        yield
