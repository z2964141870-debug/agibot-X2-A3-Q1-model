"""Verify a saved preview delivery and render its existing numerical rows."""

import argparse
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


SERVER = Path("/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_GPU_PREVIEW_R2_20261106")


def identity(path):
    value = path.read_bytes()
    return dict(path=str(path), bytes=len(value), sha256=hashlib.sha256(value).hexdigest())


def deliver(archive, output, expected_sha, expected_bytes):
    source = identity(archive)
    if source["sha256"] != expected_sha or source["bytes"] != expected_bytes:
        raise ValueError("Delivery archive differs from the server receipt")
    if not output.is_absolute() or output.resolve() != output or output.exists():
        raise ValueError("Use a new absolute local output directory")
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        if (len(members) != 44 or len({m.name for m in members}) != 44
                or sum(m.size for m in members) > 128 * 1024**2):
            raise ValueError("Unexpected delivery member set or size")
        for member in members:
            relative = PurePosixPath(member.name)
            if (not member.isfile() or relative.is_absolute() or ".." in relative.parts
                    or not relative.parts or any(p.startswith("._") for p in relative.parts)):
                raise ValueError("Unsafe delivery member: " + member.name)
        output.mkdir(parents=True)
        stream.extractall(output, members=members)
    extracted = [identity(output / member.name) for member in members]
    summary = json.loads((output / "analysis/summary.json").read_text())
    result = json.loads((output / "result.json").read_text())
    if (result.get("status") != "COMPLETED" or result.get("source_integrity") != "PASSED"
            or summary.get("total_cases") != 27 or summary.get("complete_cases") != 26
            or len(summary["rows"]) != 27):
        raise ValueError("Missing completed nominal evidence")

    from preview_summary import plot
    from PIL import Image, ImageDraw

    figures = output / "mac_figures"
    figures.mkdir()
    plot(summary["rows"], figures)
    media = sorted(output.rglob("*.mp4"))
    if len(media) != 9:
        raise ValueError("Expected nine original-speed candidate videos")
    verified, samples = [], {}
    for video in media:
        relative = video.relative_to(output)
        matching = [row for row in summary["rows"] if Path(row["video"]["path"]).relative_to(SERVER) == relative]
        if len(matching) != 1:
            raise ValueError("Video has no unique saved physical case")
        row, actual = matching[0], identity(video)
        saved = row["video"]
        if (saved.get("status") != "PASSED" or saved.get("original_speed") is not True
                or any(actual[key] != saved[key] for key in ("bytes", "sha256"))):
            raise ValueError("Video differs from its physical record")
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration", "-of", "json", str(video)],
            capture_output=True, check=True, timeout=30)
        if probe.stderr:
            raise ValueError("Video probe reported errors")
        info = json.loads(probe.stdout)["streams"][0]
        expected = saved["probe"]
        if (int(info["nb_frames"]) != expected["nb_frames"] or Fraction(info["r_frame_rate"]) != 50
                or info["width"] != expected["width"] or info["height"] != expected["height"]
                or abs(float(info["duration"])-expected["duration"]) > .001):
            raise ValueError("Video dimensions, frame count or speed changed")
        decoded = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(video),
            "-map", "0:v:0", "-f", "framehash", "-hash", "sha256", "-"],
            capture_output=True, check=True, timeout=120)
        count = sum(bool(line) and not line.startswith(b"#") for line in decoded.stdout.splitlines())
        if decoded.stderr or count != expected["nb_frames"]:
            raise ValueError("Full video decode failed or missed frames")
        indices = [round((count-1)*fraction/3) for fraction in range(4)]
        selection = "+".join("eq(n\\,"+str(index)+")" for index in indices)
        image = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf",
            "select="+selection+",scale=640:196,tile=4x1", "-frames:v", "1",
            "-f", "image2pipe", "-c:v", "png", "-"],
            capture_output=True, check=True, timeout=60)
        samples.setdefault(row["clip"], []).append((row["method"], row["horizon_frames"],
            Image.open(io.BytesIO(image.stdout)).convert("RGB"), indices))
        verified.append(dict(**actual, decoded_frames=count, original_speed=True, metadata=info))
    for clip, rows in samples.items():
        rows.sort(key=lambda value: (value[0] != "masked", -value[1]))
        contact = Image.new("RGB", (2560, len(rows)*228), "white")
        draw = ImageDraw.Draw(contact)
        for index, (method, horizon, image, frames) in enumerate(rows):
            draw.text((12, index*228+8), f"{clip} | {method} H{horizon} | frames {frames}", fill="black")
            contact.paste(image, (0, index*228+28))
        contact.save(figures / (clip+"_media_contact.png"))
    if any(identity(Path(row["path"])) != row for row in extracted):
        raise ValueError("Extracted source changed during local rendering")
    record = dict(status="PASSED", archive=source, extracted_files=extracted,
        media=verified, figures=[identity(path) for path in sorted(figures.iterdir())],
        source_summary=identity(output / "analysis/summary.json"), source_inputs_unchanged=True,
        original_server_plot_status="FAILED_MISSING_MATPLOTLIB_LOG_RETAINED",
        new_inference_calls=0, physical_transitions=0, training_updates=0,
        measured_end_to_end_latency_ms=None, user_accepted_checkpoint=None)
    (output / "mac_delivery_audit.json").write_text(json.dumps(record, indent=2)+"\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--bytes", type=int, required=True)
    args = parser.parse_args()
    record = deliver(args.archive, args.output, args.sha256, args.bytes)
    print(json.dumps(dict(status=record["status"], videos=len(record["media"]),
        figures=len(record["figures"]), decoded_frames=sum(row["decoded_frames"] for row in record["media"]))))
