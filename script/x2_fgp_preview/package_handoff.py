"""Build the internal preview runtime; existing model weights travel separately."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda:stream.read(1024*1024),b""):
            h.update(data)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--workspace",type=Path,required=True)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    runtime=Path("data/runtime/x2-fgp-viewer")
    clotho=Path("data/runtime/clotho_fgp_latest/CLOTHO-mocap-main")
    files={}
    for source in (args.source/"script/x2_fgp_preview").rglob("*"):
        if source.is_file() and "__pycache__" not in source.parts:
            files[source.relative_to(args.source)]=source
    helper=Path("script/x2_morph_human/clotho_fgp_runner.py")
    files[helper]=args.source/helper
    for source in (args.workspace/clotho).rglob("*.py"):
        if "__pycache__" not in source.parts:
            files[source.relative_to(args.workspace)]=source
    for rel in ("smpl_template.json","lucide.min.js","node_modules/three/build/three.module.js",
                "node_modules/three/build/three.core.js","node_modules/three/examples/jsm/controls/OrbitControls.js",
                "node_modules/three/LICENSE"):
        files[runtime/rel]=args.workspace/runtime/rel
    for source in (args.workspace/runtime/"wheels").glob("*.whl"):
        files[source.relative_to(args.workspace)]=source
    rows=[dict(path=str(rel),bytes=source.stat().st_size,sha256=sha(source)) for rel,source in sorted(files.items())]
    weights=[("checkpoints/canonicalization/U100hL7.07h_residual.pth","U100hL7.07h_residual.pth"),
             ("checkpoints/globalpose/weights.pt","weights.pt"),("models/SMPL_male.pkl","SMPL_male.pkl")]
    for rel,incoming in weights:
        source=args.workspace/clotho/rel
        rows.append(dict(path=str(clotho/rel),bytes=source.stat().st_size,sha256=sha(source),incoming_name=incoming))
    manifest=dict(status="INTERNAL_HANDOFF_MANIFEST",three_version="0.187.0",files=rows,
                  credentials_included=False,hardware_commands=0,
                  scope="Internal user-supplied CLOTHO/SMPL runtime, not a public model redistribution")
    encoded=(json.dumps(manifest,indent=2)+"\n").encode()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(args.output,"w:gz",compresslevel=2) as archive:
        for rel,source in sorted(files.items()):
            info=tarfile.TarInfo(str(rel))
            info.size=source.stat().st_size
            info.mtime=int(time.time())
            info.mode=0o755 if str(rel).endswith(".sh") else 0o644
            with source.open("rb") as stream:
                archive.addfile(info,stream)
        info=tarfile.TarInfo(str(runtime/"assets_manifest.json"))
        info.size=len(encoded);info.mtime=int(time.time());info.mode=0o644
        archive.addfile(info,io.BytesIO(encoded))
    (args.output.with_suffix(".manifest.json")).write_bytes(encoded)
    print(json.dumps(dict(path=str(args.output),bytes=args.output.stat().st_size,sha256=sha(args.output),
                          members=len(files)+1,separate_weight_bytes=sum(r["bytes"] for r in rows if r.get("incoming_name")))))


if __name__=="__main__":
    main()
