"""Keep the prior physical criteria and explicitly disallow RF16 extension."""

from decisions import gate as original_gate


def gate(repeats):
    value=original_gate(repeats)
    value.update(maximum_training_seconds=1300,extension_allowed=False)
    return value
