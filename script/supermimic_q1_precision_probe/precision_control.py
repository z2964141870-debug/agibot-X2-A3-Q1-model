"""Copy inference precision metadata without changing checkpoint tensors or inputs."""


def precision_checkpoint(checkpoint, precision):
    if precision not in ("bf16", "fp32"):
        raise ValueError("Require bf16 or fp32")
    if checkpoint["args"].get("amp") != "bf16":
        raise ValueError("This paired probe requires a BF16 source checkpoint")
    value = dict(checkpoint)
    value["args"] = dict(checkpoint["args"], amp="bf16" if precision == "bf16" else "none")
    return value
