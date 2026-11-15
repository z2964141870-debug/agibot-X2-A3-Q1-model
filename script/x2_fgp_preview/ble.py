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
V2_PACKET_BYTES = 137
V2_IMU = struct.Struct("<HI3h4h")
V2_UTC = struct.Struct("<2sBBQB")
FRAME_HEADER = b"\xff\xfe"
ACTIVE_UPPER = [5, 1, 4, 0, 3]
ACTIVE_LOWER = [4, 1, 3, 0, 2]


def decode_packet(payload, role):
    if role not in ADDRESSES:
        raise ValueError("unknown garment role")
    if len(payload) < 4 or bytes(payload[:2]) != FRAME_HEADER:
        raise ValueError("unexpected SmartWear packet length/header")
    wire_length = struct.unpack_from("<H", payload, 2)[0]
    if wire_length == V2_PACKET_BYTES:
        return _decode_v2_packet(payload, role)
    # The current CLOTHO receiver uses ctypes.from_buffer_copy for the newer
    # float32 layout, which accepts an ATT/MTU tail. FrameAssembler feeds this
    # function complete frames.
    if wire_length not in (PACKET_BYTES - 4, PACKET_BYTES):
        raise ValueError("unsupported SmartWear packet length field")
    if len(payload) < PACKET_BYTES:
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
    return dict(protocol="v1_float32", values=values, counters=[r[0] for r in rows],
                sample_time_fine=[r[1] for r in rows],
                wire_length_field=wire_length,
                checksum_status="NOT_VALIDATED_PROTOCOL_UNDOCUMENTED")


def _decode_v2_packet(payload, role):
    """Decode the 137-byte SmartWear V2 BLE frame used by the current suit."""
    if len(payload) < V2_PACKET_BYTES or bytes(payload[:2]) != FRAME_HEADER:
        raise ValueError("unexpected SmartWear V2 packet length/header")
    if struct.unpack_from("<H", payload, 2)[0] != V2_PACKET_BYTES:
        raise ValueError("unexpected SmartWear V2 packet length field")
    frame = bytes(payload[:V2_PACKET_BYTES])
    if (sum(frame[: V2_PACKET_BYTES - 1]) & 0xFF) != frame[-1]:
        raise ValueError("invalid SmartWear V2 checksum")
    magic, version, flags, frame_utc_ns, _reserved = V2_UTC.unpack_from(frame, 124)
    if magic != b"UT" or version != 2 or flags & ~0x01:
        raise ValueError("invalid SmartWear V2 UTC extension")
    # V2 carries six slots on both devices. The lower garment's slot 3 is a
    # reserved/unused position; the SmartWear reference selects 0,1,2,4,5.
    indices = list(range(6)) if role == "upper" else [0, 1, 2, 4, 5]
    values = np.empty((len(indices), 7), dtype=np.float64)
    counters = []
    sample_time_fine = []
    for output_index, index in enumerate(indices):
        offset = 4 + index * V2_IMU.size
        counter, sample_time, ax, ay, az, qx, qy, qz, qw = V2_IMU.unpack_from(frame, offset)
        values[output_index] = [
            ax / 256.0, ay / 256.0, az / 256.0,
            qw / 16384.0, qx / 16384.0, qy / 16384.0, qz / 16384.0,
        ]
        counters.append(counter)
        sample_time_fine.append(sample_time)
    if not np.isfinite(values).all():
        raise ValueError("invalid SmartWear V2 IMU values")
    norms = np.linalg.norm(values[:, 3:], axis=1)
    if np.any((norms < .5) | (norms > 1.5)):
        raise ValueError("invalid SmartWear V2 quaternion norms")
    values[:, 3:] /= norms[:, None]
    return dict(protocol="v2_quantized", values=values, counters=counters,
                sample_time_fine=sample_time_fine, wire_length_field=V2_PACKET_BYTES,
                frame_utc_ns=(frame_utc_ns if flags & 0x01 else None),
                checksum_status="VALIDATED_V2_SUM8")


class FrameAssembler:
    """Recover fixed SmartWear frames from BLE notification boundaries."""

    def __init__(self, frame_bytes=None):
        self.frame_bytes = frame_bytes
        self.allowed_frame_bytes = {frame_bytes} if frame_bytes else {V2_PACKET_BYTES, PACKET_BYTES}
        self.buffer = bytearray()
        self.fragments = 0
        self.frames = 0
        self.padding_bytes = 0
        self.format_errors = 0
        self.last_error = None

    def feed(self, payload):
        data = bytes(payload)
        if not data:
            return []
        self.fragments += 1
        self.buffer.extend(data)
        frames = []

        while True:
            if len(self.buffer) < 2:
                return frames

            header_at = self.buffer.find(FRAME_HEADER)
            if header_at < 0:
                # Preserve a possible first byte of a split FF FE header.
                keep = 1 if self.buffer[-1:] == FRAME_HEADER[:1] else 0
                dropped = len(self.buffer) - keep
                if dropped:
                    self.format_errors += 1
                    self.last_error = "SmartWear frame header not found"
                self.buffer = self.buffer[-keep:] if keep else bytearray()
                return frames

            if header_at:
                self.format_errors += 1
                self.last_error = "discarded bytes before SmartWear frame header"
                del self.buffer[:header_at]

            if len(self.buffer) < 4:
                return frames

            wire_length = struct.unpack_from("<H", self.buffer, 2)[0]
            length_map = {V2_PACKET_BYTES: V2_PACKET_BYTES,
                          PACKET_BYTES - 4: PACKET_BYTES,
                          PACKET_BYTES: PACKET_BYTES}
            expected = self.frame_bytes or length_map.get(wire_length)
            if expected not in self.allowed_frame_bytes:
                self.format_errors += 1
                self.last_error = f"unsupported SmartWear packet length field: {wire_length}"
                del self.buffer[0]
                continue
            if len(self.buffer) < expected:
                return frames

            frames.append(bytes(self.buffer[:expected]))
            del self.buffer[:expected]
            self.frames += 1

            if not self.buffer:
                return frames

            # A notification may contain a frame followed by MTU padding, or
            # the start of another frame. Keep only bytes from the next header.
            next_header = self.buffer.find(FRAME_HEADER)
            if next_header < 0:
                self.padding_bytes += len(self.buffer)
                self.buffer.clear()
                return frames
            if next_header:
                self.padding_bytes += next_header
                del self.buffer[:next_header]


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
        self.assemblers = {role: FrameAssembler() for role in ADDRESSES}
        self.status = {role: dict(address=addr, state="idle", packets=0, rejected=0,
                                  duplicate_packets=0, missing_packets=0, error=None)
                       for role, addr in self.addresses.items()}
        self.thread = None
        self.loop = None
        self.task = None

    def _reset_buffers_and_status(self):
        self.frames = {role: deque(maxlen=600) for role in ADDRESSES}
        self.assemblers = {role: FrameAssembler() for role in ADDRESSES}
        self.status = {role: dict(address=addr, state="idle", packets=0, rejected=0,
                                  duplicate_packets=0, missing_packets=0, error=None)
                       for role, addr in self.addresses.items()}

    def feed(self, role, payload, received_ns=None):
        now = time.monotonic_ns() if received_ns is None else received_ns
        with self.lock:
            status = self.status[role]
            status["last_packet_bytes"] = len(payload)
            assembler = self.assemblers[role]
            format_errors_before = assembler.format_errors
            packets = assembler.feed(payload)
            status.update(
                fragments=assembler.fragments,
                frames_assembled=assembler.frames,
                padding_bytes=assembler.padding_bytes,
            )
            new_format_errors = assembler.format_errors - format_errors_before
            if new_format_errors:
                status["rejected"] += new_format_errors
                status["error"] = assembler.last_error
            accepted = False
            for raw_packet in packets:
                try:
                    packet = decode_packet(raw_packet, role)
                except ValueError as error:
                    status["rejected"] += 1
                    status["error"] = str(error)
                    continue
                previous = self.frames[role][-1] if self.frames[role] else None
                if previous:
                    delta = (packet["counters"][0] - previous["counters"][0]) % 65536
                    if delta == 0:
                        status["duplicate_packets"] += 1
                        continue
                    if delta < 32768:
                        status["missing_packets"] += max(0, delta - 1)
                    else:
                        status["epoch"] = status.get("epoch", 0) + 1
                        self.frames[role].clear()
                packet["received_ns"] = now
                self.frames[role].append(packet)
                status.update(state="receiving", packets=status["packets"] + 1,
                              last_received_ns=now, error=None)
                accepted = True
            return accepted

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

    def restart(self):
        """Reconnect both garments with fresh buffers and independent scan state."""
        stopped = self.stop()
        if not stopped:
            raise RuntimeError("BLE receiver did not stop before reconnect")
        with self.lock:
            self._reset_buffers_and_status()
            self.loop = None
            self.task = None
        self.start()


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
