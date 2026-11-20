"""Verify the five frozen V2b graphs and deployment configuration."""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED = {
    "config/q1_deploy_config.json": "fd058f4caaa7334b5846c07ec8ef4e2dfedf0248bf39a2ee792a6454f8409db8",
    "onnx/q1_h002_token_head.onnx": "e0455c0b50db1c1fdc8aed1e0d283820d3fd64174b261093d99dc4430ff3e80f",
    "onnx/q1_tok_human_smpl.onnx": "7dab4d9d056e797c1ae8631d6c15ffd5fc3324234c86b821b253fcbb46adfb0e",
    "onnx/q1_tok_planner_cmd.onnx": "9c33375e028e634c908c5b2a6a6d5523b512319978076a8de12ed771a33ca1b4",
    "onnx/q1_tok_robot_pose.onnx": "9ee7a429347d0c9076657c6bb47e4635ee6b149cdc2504370fbfcecf09e6e6d9",
    "onnx/q1_tok_vr_3pt.onnx": "ff0586884a3f1b0aa33883145945be8f32e9cd91110891ffc1d7b41aa97e06ba",
}


def verify(bundle):
    actual = {name: hashlib.sha256((bundle / name).read_bytes()).hexdigest() for name in EXPECTED}
    mismatches = [name for name in EXPECTED if actual[name] != EXPECTED[name]]
    if mismatches:
        raise ValueError("Frozen V2b hash mismatch in " + ", ".join(mismatches))
    return actual


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    print(json.dumps(verify(parser.parse_args().bundle), indent=2))

