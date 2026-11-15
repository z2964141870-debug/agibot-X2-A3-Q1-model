"""Forward fresh live CLOTHO SMPL samples to the local Morph advisory service.

This bridge is deliberately limited to the data path. It never imports ROS,
MC, AimDK, or a motor writer. The only outbound request is a HumanFrame JSON
message to the loopback Morph service.
"""
from __future__ import annotations

import argparse
import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

from script.x2_morph_human.client import Client
from script.x2_morph_human.core import Calibration, SmplAdapter


def get_json(url: str, timeout: float):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--preview-url", default="http://127.0.0.1:8768")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5765")
    parser.add_argument("--clotho-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--poll-ms", type=float, default=20.0)
    parser.add_argument("--http-timeout", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cal = Calibration(**json.loads(args.calibration.read_text()))
    from script.x2_morph_human.artifacts import home
    skeleton = home() / "upstream/deploy/vr_teleop/human_skeleton.json"
    adapter = SmplAdapter(skeleton, cal)
    clock = f"fgp_bridge_monotonic:{socket.gethostname()}:{time.monotonic_ns()}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = None
    last_sequence = None
    last_received = None
    sent = accepted = errors = 0
    started = time.monotonic()

    with args.output.open("x", encoding="utf-8") as log:
        print(json.dumps(dict(status="FGP_MORPH_BRIDGE_READY", preview_url=args.preview_url,
                              endpoint=args.endpoint, calibration_id=cal.id,
                              skeleton_id=adapter.skeleton_id, hardware_commands=0)),
              flush=True)
        try:
            while True:
                try:
                    smpl_response = get_json(args.preview_url + "/api/smpl", args.http_timeout)
                    sample = smpl_response.get("sample")
                    if smpl_response.get("status") != "receiving" or not sample:
                        time.sleep(max(0.001, args.poll_ms / 1000.0))
                        continue
                    sequence = int(sample["sequence"])
                    received_ns = int(sample["received_ns"])
                    if last_sequence is not None and sequence <= last_sequence:
                        time.sleep(max(0.001, args.poll_ms / 1000.0))
                        continue
                    if last_received is not None and received_ns <= last_received:
                        raise ValueError("FGP received timestamp did not increase")
                    arrival_ns = time.monotonic_ns()
                    packet = adapter.packet_from_matrices(
                        sample["pose_mat"], sample["root_translation_m"],
                        sequence=sequence, capture_ns=received_ns,
                        capture_clock=str(sample.get("source_clock", "fgp_receiver_monotonic")),
                        arrival_ns=arrival_ns, arrival_clock=clock,
                    )
                    if client is None:
                        client = Client(args.endpoint)
                    try:
                        reply = client.request(packet)
                    except Exception:
                        client.close()
                        client = Client(args.endpoint)
                        raise
                    sent += 1
                    if reply.get("accepted_sequence") == sequence:
                        accepted += 1
                    row = dict(sequence=sequence, received_ns=received_ns,
                               arrival_ns=arrival_ns, reply=reply,
                               fgp_inference_ms=sample.get("inference_ms"),
                               hardware_commands=0)
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    log.flush()
                    last_sequence, last_received = sequence, received_ns
                except (urllib.error.URLError, TimeoutError, OSError, ValueError, Exception) as error:
                    # Keep the bridge alive through BLE warmup and temporary
                    # service restarts; do not fabricate a HumanFrame.
                    errors += 1
                    log.write(json.dumps(dict(kind="bridge_error", error=f"{type(error).__name__}: {error}",
                                              t_monotonic_ns=time.monotonic_ns()), allow_nan=False) + "\n")
                    log.flush()
                    time.sleep(max(0.05, args.poll_ms / 1000.0))
        except KeyboardInterrupt:
            pass
        finally:
            if client is not None:
                client.close()
            print(json.dumps(dict(status="FGP_MORPH_BRIDGE_STOPPED", sent=sent,
                                  accepted=accepted, errors=errors,
                                  wall_s=time.monotonic() - started,
                                  hardware_commands=0)), flush=True)


if __name__ == "__main__":
    main()
