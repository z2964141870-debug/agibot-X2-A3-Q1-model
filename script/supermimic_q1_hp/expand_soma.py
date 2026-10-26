"""Extend HP's bounded SOMA selection while keeping the original actor split."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile
import time

import pandas as pd

from retarget_soma import digest, write_json

QUOTAS = {"train":{"walking":90,"jogging":50,"turning":40,"gesture":60,"dancing":45,"jumping":15},
          "validation":{"walking":6,"jogging":4,"turning":4,"gesture":8,"dancing":6,"jumping":2}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime","archive","metadata","previous","output"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    destination=args.output/"source_bvh"
    destination.mkdir()
    sys.path.insert(0,str(args.runtime))
    from tools.prepare_soma_pilot import EXCLUDE
    previous=json.loads((args.previous/"selection.json").read_text())
    frame=pd.read_parquet(args.metadata)
    eligible=frame[(~frame.is_mirror)&(frame.is_neutral==1)&(frame.content_props=="0")
                   &frame.move_duration_frames.between(240,1200)
                   &frame.content_type_of_movement.isin(QUOTAS["train"])
                   &frame.content_body_position.isin(["standing","crouching"])
                   &~frame.filename.str.contains(EXCLUDE,regex=True)]
    lookup={r["move_soma_uniform_path"]:r for r in eligible.to_dict("records")}
    counts={s:dict.fromkeys(QUOTAS[s],0) for s in QUOTAS}
    records=[]
    seen=set()
    actor_counts={}
    for record in previous["clips"]:
        split,group=record["split"],record["group"]
        if counts[split][group]>=QUOTAS[split][group]:
            raise ValueError("Expansion would discard a previous source record")
        source=args.previous/"source_bvh"/(record["name"]+".bvh")
        if digest(source)!=record["sha256"]:
            raise ValueError("Original SOMA source changed")
        shutil.copy2(source,destination/source.name)
        records.append(record)
        seen.add(record["name"])
        counts[split][group]+=1
        key=(split,record["actor_uid"])
        actor_counts[key]=actor_counts.get(key,0)+1
    old_count=len(records)
    archive_entries=0
    last_report=time.monotonic()
    with tarfile.open(args.archive,"r|gz") as archive:
        for entry in archive:
            archive_entries+=1
            if time.monotonic()-last_report>20:
                print(json.dumps({"archive_entries":archive_entries,"clips":len(records),"counts":counts}),flush=True)
                last_report=time.monotonic()
            if not entry.isfile() or entry.name not in lookup:
                continue
            row=lookup[entry.name]
            name,actor,group=row["filename"],str(row["actor_uid"]),row["content_type_of_movement"]
            split="validation" if int(hashlib.sha256(actor.encode()).hexdigest()[:8],16)%5==0 else "train"
            if name in seen or counts[split][group]>=QUOTAS[split][group] or actor_counts.get((split,actor),0)>=4:
                continue
            if not re.fullmatch(r"[A-Za-z0-9_.-]+",name):
                raise ValueError("Unexpected SOMA identifier")
            payload=archive.extractfile(entry).read()
            (destination/(name+".bvh")).write_bytes(payload)
            records.append({"name":name,"archive_path":entry.name,"sha256":hashlib.sha256(payload).hexdigest(),
                            "split":split,"actor_uid":actor,"group":group,"frames_120hz":int(row["move_duration_frames"]),
                            "description":row["content_natural_desc_4"]})
            counts[split][group]+=1
            actor_counts[split,actor]=actor_counts.get((split,actor),0)+1
            seen.add(name)
            if all(counts[s][g]>=n for s in QUOTAS for g,n in QUOTAS[s].items()):
                break
    actors={s:sorted({r["actor_uid"] for r in records if r["split"]==s}) for s in QUOTAS}
    if set(actors["train"])&set(actors["validation"]):
        raise ValueError("Actor split overlap")
    receipt={"source_repo":"bones-studio/seed","archive":str(args.archive),"archive_sha256_previously_verified":previous["archive_sha256"],
             "metadata_sha256":digest(args.metadata),"previous_selection_sha256":digest(args.previous/"selection.json"),
             "selection":"Retain every AP01 source, then archive-order additions using HP metadata filters and actor-hash split; up to four sources per actor. No physical quality admission gates.",
             "counts":counts,"actors":actors,"reused_sources":old_count,"added_sources":len(records)-old_count,
             "archive_entries_read":archive_entries,"quotas":QUOTAS,"clips":records,
             "deficits":{f"{s}/{g}":n-counts[s][g] for s in QUOTAS for g,n in QUOTAS[s].items() if counts[s][g]<n}}
    write_json(args.output/"selection.json",receipt)
    write_json(args.output/"quotas.json",QUOTAS)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ("clips","actors")}),flush=True)


if __name__=="__main__":
    main()
