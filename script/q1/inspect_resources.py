#!/usr/bin/env python3
"""Inspect supplied Q1 archives without extracting or executing vendor code."""

import argparse
import hashlib
import json
import posixpath
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def asset_reference(member, reference, members, meshdir=None):
    if "://" in reference or reference.startswith("/"):
        return {"reference": reference, "status": "external_unverified"}
    resolved = posixpath.normpath(posixpath.join(
        posixpath.dirname(member), meshdir or "", reference
    ))
    return {
        "reference": reference,
        "archive_member": resolved,
        "status": "present" if resolved in members else "missing",
    }


def inspect_model(archive, member, members):
    root = ET.fromstring(archive.read(member))
    if root.tag == "robot":
        joints = []
        for joint in root.findall("joint"):
            limit = joint.find("limit")
            axis = joint.find("axis")
            joints.append({
                **joint.attrib,
                "limit": dict(limit.attrib) if limit is not None else None,
                "axis": dict(axis.attrib) if axis is not None else None,
            })
        return {
            "member": member,
            "format": "URDF",
            "movable_joint_count": sum(j["type"] != "fixed" for j in joints),
            "joints": joints,
            "link_count": len(root.findall("link")),
            "mass_kg": sum(float(m.get("value")) for m in root.findall(".//mass")),
            "collision_count": len(root.findall(".//collision")),
            "mesh_references": [asset_reference(member, m.get("filename"), members)
                                for m in root.findall(".//mesh")],
        }
    if root.tag != "mujoco":
        raise ValueError(f"Unsupported asset root: {root.tag} ({member})")
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    joints = [dict(j.attrib) for j in root.findall("./worldbody//joint")]
    actuators = [{"type": a.tag, **a.attrib} for a in root.findall("./actuator/*")]
    return {
        "member": member,
        "format": "MJCF",
        "joint_count": len(joints),
        "freejoint_count": len(root.findall("./worldbody//freejoint")),
        "joints": joints,
        "mass_kg": sum(float(m.get("mass", 0)) for m in root.findall("./worldbody//inertial")),
        "actuators": actuators,
        "sensors": [{"type": s.tag, **s.attrib} for s in root.findall("./sensor/*")],
        "mesh_references": [asset_reference(member, m.get("file"), members, meshdir)
                            for m in root.findall("./asset/mesh")],
        "runtime_validated": False,
    }


def inspect_archive(path):
    result = {"filename": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
    if path.suffix.lower() != ".zip":
        result["content_inspection"] = "disk_image_metadata_only"
        return result
    with zipfile.ZipFile(path) as archive:
        entries = [i for i in archive.infolist() if not i.is_dir()]
        names = {i.filename for i in entries}
        result.update({
            "file_count": len(entries),
            "uncompressed_bytes": sum(i.file_size for i in entries),
            "zip_comment": archive.comment.decode("utf-8", errors="replace"),
            "unsafe_paths": [n for n in sorted(names)
                             if n.startswith("/") or ".." in n.split("/")],
            "model_files": sorted(n for n in names if n.endswith((".onnx", ".pt", ".pth", ".ckpt"))),
            "top_level_documents": sorted(n for n in names if n.count("/") == 1 and n.endswith(".md")),
        })
        if path.name == "q1_v3.zip":
            result["assets"] = [inspect_model(archive, n, names) for n in sorted(names)
                                if n.endswith(".urdf") or ("/xml/" in n and n.endswith(".xml"))]
        if path.name.startswith("sdk_q1"):
            result["message_definitions"] = sorted(n for n in names if n.endswith((".msg", ".srv")))
            result["smpl_named_members"] = sorted(n for n in names if "smpl" in n.lower())
            result["imu_named_definitions"] = sorted(n for n in result["message_definitions"]
                                                     if "imu" in n.lower())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = sorted(p for p in args.source.iterdir() if p.suffix.lower() in (".zip", ".dmg"))
    if not paths:
        parser.error("No Q1 archives found")
    result = {"scope": "static_only", "archives": [inspect_archive(p) for p in paths]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for archive in result["archives"]:
        print(f"{archive['filename']}: {archive['bytes']} bytes, sha256={archive['sha256']}")
        for asset in archive.get("assets", []):
            missing = sum(m["status"] == "missing" for m in asset["mesh_references"])
            count = asset.get("movable_joint_count", asset.get("joint_count"))
            print(f"  {asset['member']}: joints={count}, mass={asset['mass_kg']:.6f}, missing_meshes={missing}")
    print(f"Static report: {args.output}")


if __name__ == "__main__":
    main()
