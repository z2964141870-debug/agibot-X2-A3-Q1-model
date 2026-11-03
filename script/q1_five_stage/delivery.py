"""Build and verify the small final Primebot delivery before shared training."""

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess

from contracts import required_delivery_paths, validate_frozen
from common import FIXED, ROOT, identity, now, require, save


def read(path):
    return json.loads(Path(path).read_text())


def artifact(root, relative):
    parts = PurePosixPath(relative)
    require(not parts.is_absolute() and ".." not in parts.parts, "Invalid delivery member")
    path = root / relative
    require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), "Delivery symlink or path escape")
    value = identity(path)
    return dict(relative=relative, bytes=value["bytes"], sha256=value["sha256"])


def load_manifest(path):
    value = read(path)
    source = Path(value["manifest_server_path"])
    require(source.is_absolute() and source.is_relative_to(ROOT), "Manifest server path must be in ZHY")
    value["identity"] = dict(identity(path), path=str(source))
    return value


def prepare(review, frozen_path, output):
    require(not output.exists(), "Preserve an existing delivery manifest")
    require(review.resolve().is_relative_to(ROOT), "Prepare the manifest on the sole server root")
    names = validate_frozen(read(frozen_path))
    paths = required_delivery_paths(names)
    paths.update(str(p.relative_to(review)) for p in review.rglob("frame_*.png"))
    media = []
    for stage in ("initial", "final"):
        fixed = read(review / (stage + "_fixed/result.json"))
        require(fixed.get("status") == "PASSED", "Fixed evaluation is unfinished")
        for row in fixed["cases"]:
            if row["seed"] != 0:
                continue
            require(row["clip"] in FIXED, "Unexpected fixed video")
            video = row.get("video", {})
            require(video.get("status") == "PASSED" and video.get("original_speed") is True,
                    "Original-speed video is missing")
            relative = stage + "_fixed/" + row["clip"] + "_s0/tracking.mp4"
            item = artifact(review, relative)
            require(item["bytes"] == video["bytes"] and item["sha256"] == video["sha256"],
                    "Video identity differs from the evaluator")
            media.append(dict(relative=relative, probe=video["probe"]))
    require(len(media) == 6 and len({x["relative"] for x in media}) == 6, "Need six distinct initial/final videos")
    record = dict(status="READY_FOR_MAC_VERIFICATION", started_utc=now(),
        review_root=str(review), manifest_server_path=str(output), frozen_data=identity(frozen_path),
        artifacts=[artifact(review, name) for name in sorted(paths)], media=media,
        checkpoint_and_npz_storage="SERVER_LOCAL_ONLY", user_accepted_checkpoint=None)
    save(output, record)
    return load_manifest(output)


def check_visuals(manifest, visual):
    expected = {row["relative"]: row for row in manifest["artifacts"] if row["relative"].endswith(".mp4")}
    rows = visual.get("media", [])
    require(visual.get("status") == "REVIEWED" and len(rows) == len(expected)
            and len({row.get("relative") for row in rows}) == len(expected)
            and {row.get("relative") for row in rows} == set(expected),
            "Need a separate visual receipt for every exact video")
    require(all(row.get("visual_reviewed") is True and row.get("sha256") == expected[row["relative"]]["sha256"]
                for row in rows), "Visual review is missing or belongs to another video")


def verify(manifest_path, local_root, visual_path, output, ffmpeg="ffmpeg", ffprobe="ffprobe"):
    require(not output.exists(), "Preserve an existing Mac verification receipt")
    manifest, visual = load_manifest(manifest_path), read(visual_path)
    require(manifest.get("status") == "READY_FOR_MAC_VERIFICATION", "Delivery is not ready")
    record = dict(status="VERIFYING", started_utc=now(), delivery_manifest=manifest["identity"],
                  artifacts=[], media_checks=[], visual_receipt=identity(visual_path),
                  new_physical_steps=0, new_optimizer_updates=0, user_accepted_checkpoint=None)
    try:
        check_visuals(manifest, visual)
        rows = manifest["artifacts"]
        require(rows and len({r["relative"] for r in rows}) == len(rows), "Duplicate delivery members")
        for row in rows:
            require(artifact(local_root, row["relative"]) == row, "Mac artifact mismatch: " + row["relative"])
        record["artifacts"] = rows
        require(shutil.which(ffmpeg) and shutil.which(ffprobe), "Use the existing Mac video tools")
        for media in manifest["media"]:
            path = local_root / media["relative"]
            probe = json.loads(subprocess.check_output([ffprobe, "-v", "error", "-show_entries",
                "stream=width,height,r_frame_rate", "-of", "json", str(path)], text=True))["streams"][0]
            expected = media["probe"]
            require(all(str(probe[k]) == str(expected[k]) for k in ("width", "height", "r_frame_rate")),
                    "Mac video dimensions or original frame rate changed")
            decoded = subprocess.check_output([ffmpeg, "-v", "error", "-i", str(path),
                "-an", "-pix_fmt", "rgb24", "-f", "framehash", "-"], text=True)
            frames = sum(bool(line.strip()) and not line.startswith("#") for line in decoded.splitlines())
            require(frames == int(expected["nb_frames"]) and frames > 0, "Incomplete full video decode")
            record["media_checks"].append(dict(relative=media["relative"], status="PASSED",
                decoded_frames=frames, visual_reviewed=True, probe=probe))
        require(len(record["media_checks"]) == 6, "Need all six Mac video checks")
        record["status"] = "VERIFIED"
    except BaseException as error:
        record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["ended_utc"] = now()
        save(output, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    make = sub.add_parser("prepare")
    for key in ("review", "frozen", "output"):
        make.add_argument("--" + key, type=Path, required=True)
    check = sub.add_parser("verify-mac")
    for key in ("manifest", "local-root", "visual-receipt", "output"):
        check.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        value = prepare(args.review.resolve(), args.frozen.resolve(), args.output.resolve())
    else:
        value = verify(args.manifest, args.local_root, args.visual_receipt, args.output)
    print(json.dumps(dict(status=value["status"], artifacts=len(value["artifacts"]))))


if __name__ == "__main__":
    main()
