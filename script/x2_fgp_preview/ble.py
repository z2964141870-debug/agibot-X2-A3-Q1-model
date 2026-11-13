"""SmartWear notification decoder and bounded native BLE acquisition."""
from __future__ import annotations

import argparse
import asyncio
from collections import deque
import json
from pathlib import Path
import struct
import threading
import time

import numpy as np

ADDRESSES = {"upper": "F5:C6:03:E6:4B:30", "lower": "DA:1B:78:C1:56:1F"}
NOTIFY_UUID = "adaf0101-c332-42a8-93bd-25e905756cb8"
IMU = struct.Struct("<HI7f")
PACKET_BYTES = 217
ACTIVE_UPPER = [5, 1, 4, 0, 3]
ACTIVE_LOWER = [4, 1, 3, 0, 2]


def decode_packet(payload, role):
    if role not in ADDRESSES:
        raise ValueError("unknown garment role")
    if len(payload) != PACKET_BYTES or bytes(payload[:2]) != b"\xff\xfe":
        raise ValueError("unexpected SmartWear packet length/header")
    offsets = [8 + i * IMU.size for i in range(4)] + [148 + i * IMU.size for i in range(2)]
    if role == "lower":
        offsets = offsets[:3] + offsets[4:]
    rows = [IMU.unpack_from(payload, offset) for offset in offsets]
    values = np.array([[*r[2:5], r[8], *r[5:8]] for r in rows], dtype=np.float64)
    norms = np.linalg.norm(values[:, 3:], axis=1)
    if not np.isfinite(values).all() or np.any((norms < .5) | (norms > 1.5)):
        raise ValueError("invalid IMU values/quaternion norms")
    values[:, 3:] /= norms[:, None]
    return dict(values=values, counters=[r[0] for r in rows],
                sample_time_fine=[r[1] for r in rows],
                wire_length_field=struct.unpack_from("<H", payload, 2)[0],
                checksum_status="NOT_VALIDATED_PROTOCOL_UNDOCUMENTED")


class BleSource:
    def __init__(self, addresses=None, scan_seconds=120, connect_seconds=60):
        self.addresses = dict(addresses or ADDRESSES)
        if set(self.addresses) != set(ADDRESSES) or len(set(self.addresses.values())) != 2:
            raise ValueError("upper and lower must be distinct devices")
        if not 1 <= scan_seconds <= 300 or not 1 <= connect_seconds <= 120:
            raise ValueError("scan/connect duration out of bounds")
        self.scan_seconds, self.connect_seconds = scan_seconds, connect_seconds
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.frames = {role: deque(maxlen=600) for role in ADDRESSES}
        self.status = {role: dict(address=addr, state="idle", packets=0, rejected=0,
                                  duplicate_packets=0, missing_packets=0, error=None)
                       for role, addr in self.addresses.items()}
        self.thread = None
        self.loop = None
        self.task = None

    def feed(self, role, payload, received_ns=None):
        now = time.monotonic_ns() if received_ns is None else received_ns
        with self.lock:
            status = self.status[role]
            status["last_packet_bytes"] = len(payload)
            try:
                packet = decode_packet(payload, role)
            except ValueError as error:
                status["rejected"] += 1
                status["error"] = str(error)
                return False
            previous = self.frames[role][-1] if self.frames[role] else None
            if previous:
                delta = (packet["counters"][0] - previous["counters"][0]) % 65536
                if delta == 0:
                    status["duplicate_packets"] += 1
                    return False
                if delta < 32768:
                    status["missing_packets"] += max(0, delta - 1)
                else:
                    status["epoch"] = status.get("epoch", 0) + 1
                    self.frames[role].clear()
            packet["received_ns"] = now
            self.frames[role].append(packet)
            status.update(state="receiving", packets=status["packets"] + 1,
                          last_received_ns=now, error=None)
            return True

    def snapshot(self):
        now = time.monotonic_ns()
        with self.lock:
            result = {}
            for role, status in self.status.items():
                frames = list(self.frames[role])
                recent = [f for f in frames if now - f["received_ns"] < 2_000_000_000]
                age = (now - frames[-1]["received_ns"]) / 1e6 if frames else None
                fps = ((len(recent) - 1) * 1e9 / (recent[-1]["received_ns"] - recent[0]["received_ns"])) if len(recent) > 1 else 0
                result[role] = dict(status, age_ms=age, hz=fps, fresh=age is not None and age <= 150,
                                    imu_values=frames[-1]["values"].tolist() if frames else None)
            return result

    def pair(self, max_age_ms=150, max_skew_ms=60):
        now = time.monotonic_ns()
        with self.lock:
            if not all(self.frames.values()):
                return None
            upper, lower = (self.frames[r][-1] for r in ("upper", "lower"))
            times = [upper["received_ns"], lower["received_ns"]]
            if now - min(times) > max_age_ms * 1e6 or abs(times[0] - times[1]) > max_skew_ms * 1e6:
                return None
            return dict(upper=upper, lower=lower, received_ns=min(times),
                        skew_ms=abs(times[0] - times[1]) / 1e6,
                        epoch=[self.status[r].get("epoch", 0) for r in ("upper", "lower")])

    @staticmethod
    def active_values(pair):
        return np.concatenate((pair["upper"]["values"][ACTIVE_UPPER],
                               pair["lower"]["values"][ACTIVE_LOWER]))

    async def run(self):
        from bleak import BleakClient, BleakScanner
        devices = {}
        def detected(device, advert):
            for role, address in self.addresses.items():
                if device.address.upper() == address.upper():
                    devices[role] = device
                    with self.lock:
                        self.status[role].update(state="discovered", rssi=advert.rssi, name=device.name)
        with self.lock:
            for row in self.status.values():
                row.update(state="scanning", error=None)
        async with BleakScanner(detection_callback=detected):
            deadline = time.monotonic() + self.scan_seconds
            while len(devices) < 2 and time.monotonic() < deadline and not self.stop_event.is_set():
                await asyncio.sleep(.2)
        connect_lock = asyncio.Lock()
        async def receive(role):
            if role not in devices:
                self.status[role].update(state="not_seen", error=f"not seen in {self.scan_seconds}s scan")
                return
            for attempt in range(1, 3):
                if self.stop_event.is_set():
                    return
                client = BleakClient(devices[role], timeout=self.connect_seconds)
                try:
                    with self.lock:
                        self.status[role].update(state="connecting", attempt=attempt)
                    async with connect_lock:
                        await asyncio.wait_for(client.connect(), self.connect_seconds)
                        await asyncio.wait_for(client.start_notify(NOTIFY_UUID, lambda _, data: self.feed(role, data)), 30)
                    with self.lock:
                        self.status[role]["state"] = "subscribed"
                    while client.is_connected and not self.stop_event.is_set():
                        await asyncio.sleep(.2)
                    with self.lock:
                        self.status[role]["state"] = "disconnected"
                    return
                except Exception as error:
                    with self.lock:
                        self.status[role].update(state="retry_wait" if attempt < 2 else "error",
                                                 error=f"{type(error).__name__}: {error}")
                    if attempt < 2:
                        await asyncio.sleep(3)
                finally:
                    if client.is_connected:
                        await asyncio.wait_for(client.disconnect(), 10)
        await asyncio.gather(*(receive(role) for role in ADDRESSES))

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        def target():
            async def entry():
                self.loop = asyncio.get_running_loop()
                self.task = asyncio.current_task()
                await self.run()
            try:
                asyncio.run(entry())
            except asyncio.CancelledError:
                pass
            except Exception as error:
                with self.lock:
                    for row in self.status.values():
                        row.update(state="error", error=f"{type(error).__name__}: {error}")
        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
            if self.thread.is_alive() and self.loop and self.task:
                self.loop.call_soon_threadsafe(self.task.cancel)
                self.thread.join(timeout=15)
        return not self.thread or not self.thread.is_alive()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--scan-seconds", type=float, default=120)
    parser.add_argument("--observe-seconds", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    if not 5 <= args.observe_seconds <= 600:
        parser.error("observe duration must be 5..600 seconds")
    source = BleSource(scan_seconds=args.scan_seconds)
    source.start()
    began = time.monotonic()
    observation_began = None
    try:
        while time.monotonic() - began < args.scan_seconds + args.observe_seconds + 250:
            status = source.snapshot()
            both = all(s["packets"] > 0 for s in status.values())
            if both and observation_began is None:
                observation_began = time.monotonic()
            print(json.dumps(dict(elapsed_s=round(time.monotonic()-began, 2), devices={r:{k:v for k,v in s.items() if k!="imu_values"} for r,s in status.items()})), flush=True)
            if observation_began is not None and time.monotonic()-observation_began >= args.observe_seconds:
                break
            if all(s["state"] in ("error", "not_seen", "disconnected") for s in status.values()):
                break
            time.sleep(5)
    finally:
        status = source.snapshot()
        stopped = source.stop()
    result = dict(status="BLE_BOTH_RECEIVED" if all(s["packets"] for s in status.values()) else "BLE_INCOMPLETE",
                  devices=status, wall_s=time.monotonic()-began, hardware_commands=0,
                  source_clock="receiver_monotonic; not hardware capture time", checksum_validated=False,
                  owned_acquisition_thread_stopped=stopped)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
