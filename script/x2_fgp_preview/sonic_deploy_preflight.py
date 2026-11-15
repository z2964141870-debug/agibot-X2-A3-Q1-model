"""Check the installed SONIC model pair before wiring it to the X2 backend."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--policies", type=Path, default=Path("/home/agi/gear-sonic/policies"))
    parser.add_argument("--pair", choices=("stage3", "v16"), required=True)
    args = parser.parse_args()

    import onnxruntime as ort

    pairs = {
        "stage3": ("x2_smpl_tokenizer_v11release.onnx", "x2_sonic_s1ft16000_g1_token.onnx"),
        "v16": ("x2_sonic_v16ft8_45000_smpl_tokenizer.onnx", "x2_sonic_v16ft8_45000_g1_token.onnx"),
    }
    encoder, actor = pairs[args.pair]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    def inspect(name):
        session = ort.InferenceSession(str(args.policies / name), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        return (session.get_modelmeta().custom_metadata_map,
                [(x.name, x.shape[-1]) for x in session.get_inputs()],
                [(x.name, x.shape[-1]) for x in session.get_outputs()])

    enc, enc_in, enc_out = inspect(encoder)
    act, act_in, act_out = inspect(actor)
    issues = []
    if (enc_in, enc_out, act_in, act_out) != (
            [("smpl_obs", 840)], [("motion_token", 64)], [("obs", 1670)], [("action", 31)]):
        issues.append("model tensor contract differs from the installed SONIC backend")
    if enc.get("graph_kind") != "smpl_tokenizer" or act.get("graph_kind") != "any2any_token_deploy":
        issues.append("model graph kinds differ from the SONIC token path")
    if enc.get("pairs_with") != actor:
        issues.append("tokenizer metadata names another decoder")
    fingerprint = enc.get("codec_fingerprint")
    if not fingerprint or fingerprint != act.get("codec_fingerprint"):
        issues.append("shared codec fingerprint cannot be verified")
    if enc.get("ori_mode") not in ("heading", "full"):
        issues.append("tokenizer does not declare a supported orientation mode")
    print(json.dumps(dict(pair=args.pair, tokenizer=encoder, decoder=actor,
                          model_contract_verified=not issues, issues=issues,
                          ori_mode=enc.get("ori_mode"), hardware_commands=0)))
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
