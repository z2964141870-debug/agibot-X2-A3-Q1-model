"""Compare captured initialization arrays without importing a physics runtime."""

import hashlib

import numpy as np


def array_summary(value):
    value = np.asarray(value)
    if value.dtype.kind not in "biuf":
        raise ValueError("Initialization fields must be real numeric arrays")
    return dict(shape=list(value.shape), dtype=str(value.dtype), size=value.size,
                finite=bool(np.isfinite(value).all()),
                sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest())


def compare_arrays(reference, current, atol=2e-6, rtol=2e-5):
    reference, current = np.asarray(reference), np.asarray(current)
    before, after = array_summary(reference), array_summary(current)
    if before["shape"] != after["shape"] or before["dtype"] != after["dtype"]:
        return dict(matches=False, reason="shape_or_dtype_changed", reference=before, current=after)
    if not before["finite"] or not after["finite"]:
        return dict(matches=False, reason="nonfinite", reference=before, current=after)
    if not reference.size:
        return dict(matches=True, changed_values=0, outside_tolerance=0, maximum_absolute_difference=0.)
    difference = np.abs(current.astype(np.float64) - reference.astype(np.float64))
    exact = current == reference
    close = (difference <= atol + rtol * np.abs(reference)) if reference.dtype.kind == "f" else exact
    index = int(difference.argmax())
    return dict(matches=bool(close.all()), changed_values=int(np.count_nonzero(~exact)),
                outside_tolerance=int(np.count_nonzero(~close)),
                maximum_absolute_difference=float(difference.flat[index]),
                maximum_difference_index=[int(part) for part in np.unravel_index(index, reference.shape)],
                reference_at_maximum=reference.flat[index].item(),
                current_at_maximum=current.flat[index].item())


class InitializationAudit:
    """Retain one reference trace and report each branch's first divergence."""

    def __init__(self, receipt, atol=2e-6, rtol=2e-5):
        self.atol, self.rtol = atol, rtol
        self.reference = []
        self.receipt = receipt
        receipt.update(atol=atol, rtol=rtol, traces=[])

    def begin(self, label, metadata):
        self.trace = dict(label=label, metadata=metadata, snapshots=[], complete=False,
                          first_state_action_difference=None, first_rng_difference=None)
        self.receipt["traces"].append(self.trace)
        self.is_reference = len(self.receipt["traces"]) == 1

    def record(self, stage, values):
        arrays = {key: np.array(value, copy=True) for key, value in values.items()}
        snapshot = dict(stage=stage, fields={key: array_summary(value) for key, value in arrays.items()})
        index = len(self.trace["snapshots"])
        if self.is_reference:
            self.reference.append((stage, arrays))
        elif index >= len(self.reference) or stage != self.reference[index][0]:
            snapshot["comparison"] = dict(matches=False, reason="stage_order_changed")
            if self.trace["first_state_action_difference"] is None:
                self.trace["first_state_action_difference"] = dict(stage=stage, field=None)
        else:
            reference = self.reference[index][1]
            comparison = {}
            for key in sorted(reference.keys() | arrays.keys()):
                comparison[key] = (compare_arrays(reference[key], arrays[key], self.atol, self.rtol)
                                   if key in reference and key in arrays else
                                   dict(matches=False, reason="field_set_changed"))
                category = "first_rng_difference" if key.startswith("rng.") else "first_state_action_difference"
                if not comparison[key]["matches"] and self.trace[category] is None:
                    self.trace[category] = dict(stage=stage, field=key)
            snapshot["comparison"] = comparison
        self.trace["snapshots"].append(snapshot)

    def finish(self):
        self.trace["complete"] = len(self.trace["snapshots"]) == len(self.reference)
        if not self.trace["complete"] and self.trace["first_state_action_difference"] is None:
            self.trace["first_state_action_difference"] = dict(stage="trace_end", field=None)
