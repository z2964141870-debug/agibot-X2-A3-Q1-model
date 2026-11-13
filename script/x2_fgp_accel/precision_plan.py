"""Describe actual floating compute layers without changing the author graph."""
import argparse
import json
from pathlib import Path
import sys

from script.x2_fgp_realtime.assets import file_digest


COMPUTE_TYPES = {"MATRIX_MULTIPLY", "CONVOLUTION", "DECONVOLUTION", "ELEMENTWISE",
                 "UNARY", "ACTIVATION", "POOLING", "REDUCE", "SOFTMAX", "NORMALIZATION", "SCALE"}


def make_plan(rows, onnx_sha256):
    selected = []
    names = set()
    for row in rows:
        if row["type"] not in COMPUTE_TYPES or not row["floating_data_outputs"]:
            continue
        name = row["name"]
        if not name or any(c in name for c in ",:*\n\r") or name in names:
            raise ValueError("Unsafe or duplicate precision layer name")
        names.add(name)
        selected.append(dict(name=name, type=row["type"],
                             precision="fp16" if row["type"] == "MATRIX_MULTIPLY" else "fp32",
                             output_type="fp32"))
    if not any(row["precision"] == "fp16" for row in selected):
        raise ValueError("No matrix multiplication layers found")
    return dict(format_version=1, onnx_sha256=onnx_sha256, layers=selected,
                policy="FP16 matrix compute, FP32 floating compute otherwise and FP32 compute outputs",
                hardware_commands=0, training_updates=0, model_calls=0)


def specifications(plan, expected_sha):
    if plan.get("format_version") != 1 or plan.get("onnx_sha256") != expected_sha:
        raise ValueError("Precision plan/source identity mismatch")
    rows = plan["layers"]
    rebuilt = make_plan([dict(name=r["name"], type=r["type"], floating_data_outputs=True)
                         for r in rows], expected_sha)
    if rebuilt["layers"] != rows:
        raise ValueError("Unexpected precision plan policy")
    return (",".join(r["name"]+":"+r["precision"] for r in rows),
            ",".join(r["name"]+":fp32" for r in rows))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing precision plans")
    from .build import resources, require_resources
    require_resources(resources())
    sys.path.append("/usr/lib/python3.10/dist-packages")
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    reader = trt.OnnxParser(network, logger)
    source_hash = file_digest(args.onnx)
    if not reader.parse_from_file(str(args.onnx.resolve())):
        raise ValueError([str(reader.get_error(i)) for i in range(reader.num_errors)])
    rows = []
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        outputs = [layer.get_output(j) for j in range(layer.num_outputs)]
        floating = bool(outputs) and all(t is not None and t.dtype == trt.float32
                                        and not t.is_shape_tensor for t in outputs)
        rows.append(dict(name=layer.name, type=str(layer.type).split(".")[-1],
                         floating_data_outputs=floating))
    plan = make_plan(rows, source_hash)
    specifications(plan, source_hash)
    if file_digest(args.onnx) != source_hash:
        raise ValueError("Author ONNX changed during planning")
    plan.update(parser_layers=network.num_layers, tensorrt_version=trt.__version__,
                resource_after=resources())
    require_resources(plan["resource_after"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(status="PRECISION_PLAN_PREPARED", parser_layers=network.num_layers,
                         compute_layers=len(plan["layers"]),
                         fp16_matrix_layers=sum(r["precision"] == "fp16" for r in plan["layers"]),
                         plan_sha256=file_digest(args.output), model_calls=0)))


if __name__ == "__main__":
    main()
