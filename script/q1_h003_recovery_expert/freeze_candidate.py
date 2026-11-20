"""Freeze a selected trajectory prior and trained PPO residual as a portable ONNX."""
from pathlib import Path
import argparse
import json
import hashlib
import shutil
import torch
from sequence_model import SequenceExpert,export

def freeze(prior,checkpoint,dest):
    dest.mkdir(parents=True,exist_ok=False)
    for name in ("feedback.npy","keys_1.npy"):
        shutil.copyfile(prior/name,dest/name)
    model=SequenceExpert(dest)
    weights=torch.load(checkpoint,map_location="cpu")
    model.residual.load_state_dict({k.removeprefix("residual."):v for k,v in weights.items() if k.startswith("residual.")})
    model.eval()
    path=dest/"q1_recovery_expert.onnx"
    export(model,path)
    (dest/"candidate.json").write_text(json.dumps(dict(prior=str(prior),checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),onnx_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for a in ("prior","checkpoint","dest"):p.add_argument(a,type=Path)
    args=p.parse_args()
    freeze(args.prior,args.checkpoint,args.dest)

