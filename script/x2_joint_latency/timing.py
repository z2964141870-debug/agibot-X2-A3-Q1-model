"""One-host monotonic timestamps; no subtraction of original recording clocks."""
import numpy as np


def statistics(values, budget=None):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return None
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite timing")
    result = dict(count=len(a), p50=float(np.median(a)), p95=float(np.percentile(a, 95)),
                  p99=float(np.percentile(a, 99)), maximum=float(a.max()))
    if budget is not None:
        result.update(budget_ms=budget, over_budget=int((a > budget).sum()))
    return result


def reference_ages(reply, now_ns, clock):
    if not reply.get("ready"):
        return None
    t = reply["timing"]
    if t["capture_clock"] != clock:
        raise ValueError("Cannot subtract unlike clocks")
    slots, support = t["slot_capture_ns"], t["raw_support_end_ns"]
    if len(slots) != 10 or slots[-1] - slots[0] != 180_000_000:
        raise ValueError("Original preview window changed")
    if max(support) > now_ns or any(b < a for a, b in zip(slots, support)):
        raise ValueError("Future or insufficient raw support")
    return dict(oldest_reference_age_ms=(now_ns-slots[0])/1e6,
                newest_slot_age_ms=(now_ns-slots[-1])/1e6,
                newest_support_age_ms=(now_ns-max(support))/1e6,
                preview_span_ms=180.)
