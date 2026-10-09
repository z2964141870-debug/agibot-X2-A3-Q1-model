"""Check an actual ONNX against the pinned RKNN input contract, without conversion."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from script.a3.fullchain_support import DATA, VENDOR, digest, mark, write_json


def run(path):
    module_path = VENDOR / "gear_sonic_deploy/scripts/convert_a3_onnx_to_rknn.py"
    spec = importlib.util.spec_from_file_location("yuanqi_pinned_rknn_converter", module_path)
    converter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = converter
    spec.loader.exec_module(converter)
    model_spec = converter.external_model_spec(path)
    schema = converter.probe_onnx_schema(path)
    kwargs = converter.rknn_load_kwargs(model_spec, schema, path)
    evidence = {"onnx": str(path), "sha256": digest(path), "schema": schema,
                "load_kwargs": kwargs, "scope": "INPUT_CONTRACT_ONLY_NO_RKNN_PACKAGE_OR_RUNTIME",
                "converter_sha256": digest(module_path), "backup_status": "LOCAL_ONLY"}
    write_json(DATA / "rknn_input_contract.json", evidence)
    mark("rknn_input_contract", "passed", evidence=str(DATA / "rknn_input_contract.json"),
         scope=evidence["scope"])
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    run(parser.parse_args().onnx.resolve())
