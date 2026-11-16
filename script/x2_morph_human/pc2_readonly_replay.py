"""Orin recorded HumanFrames to PC2 ROS feedback and local advisory targets."""

import argparse
from collections import Counter
import json
from pathlib import Path
from queue import Empty, Queue
import socket
import threading
import time

import numpy as np


GROUPS = ("leg", "waist", "arm", "head")
CLOCK = "pc2_local_monotonic"
PERIOD_NS = 20_000_000
PACKETS = 300
MAX_PACKET_BYTES = 32768
DROPOUT = range(144, 152)


def paced_wait(deadline):
    while (remaining := deadline - time.monotonic_ns()) > 0:
        time.sleep(remaining / 1e9)


def send(args):
    with args.capture.open() as stream:
        frames = [json.loads(line) for _, line in zip(range(PACKETS), stream)]
    if len(frames) != PACKETS:
        raise ValueError("recorded human input is too short")
    with socket.create_connection((args.host, args.port), timeout=3) as link:
        link.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        start = time.monotonic_ns() + 150_000_000
        base = frames[0]["arrival_ns"]
        for index, frame in enumerate(frames):
            if args.dropout and index in DROPOUT:
                continue
            paced_wait(start + frame["arrival_ns"] - base)
            link.sendall(json.dumps(frame, allow_nan=False).encode() + b"\n")
    print(json.dumps(dict(sent=len(frames) - (len(DROPOUT) if args.dropout else 0), hardware_commands=0)))


def receive(args):
    import rclpy
    from aimdk_msgs.msg import JointStateArray
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu

    from script.x2_morph_human.artifacts import activate
    from script.x2_morph_human.runtime import Runtime
    from script.x2_morph_human.shadow_reader import Feedback

    root = activate(args.policy_root)
    from deploy.sim2sim import policy_node

    policy_node.ORT_THREADS = 2  # applies only to this isolated process
    runtime = Runtime(root, max_age_ns=120_000_000, max_interp_gap_ns=100_000_000)

    class TimedSession:
        def __init__(self, session):
            self.session = session
            self.last_ms = None

        def run(self, names, feed):
            begin = time.monotonic_ns()
            result = self.session.run(names, feed)
            self.last_ms = (time.monotonic_ns() - begin) / 1e6
            return result

    timer = TimedSession(runtime.runner.session)
    runtime.runner.session = timer
    cfg = json.loads(args.profile.read_text())
    if cfg["imu_body"] not in runtime.robot.tree_names or set(cfg["joint_state_topics"]) != set(GROUPS):
        raise ValueError("unexpected X2 sensor profile")
    feedback = Feedback(list(runtime.robot.native_joint_names), cfg["imu_body"], cfg["body_from_sensor_wxyz"])
    lock = threading.Lock()
    stop = threading.Event()
    packets = Queue(maxsize=4)
    failures = Queue()
    ended = threading.Event()
    rx_stamps = []
    rclpy.init()
    node = Node("x2_pc2_readonly_probe")

    def joints(group, message):
        with lock:
            feedback.joints(group, message, time.monotonic_ns())

    def inertial(message):
        with lock:
            feedback.inertial(message, time.monotonic_ns())

    subscriptions = [node.create_subscription(JointStateArray, cfg["joint_state_topics"][group],
                       lambda message, g=group: joints(g, message), qos_profile_sensor_data) for group in GROUPS]
    subscriptions.append(node.create_subscription(Imu, cfg["imu_topic"], inertial, qos_profile_sensor_data))

    def spin():
        try:
            while not stop.is_set():
                rclpy.spin_once(node, timeout_sec=0.05)
        except Exception as exc:
            failures.put(exc)

    def read_packets(listener):
        try:
            conn, peer = listener.accept()
            with conn:
                if peer[0] != args.allow:
                    raise ValueError("unexpected replay sender")
                conn.settimeout(2)
                with conn.makefile("rb") as stream:
                    while not stop.is_set():
                        raw = stream.readline(MAX_PACKET_BYTES + 1)
                        if not raw:
                            break
                        if len(raw) > MAX_PACKET_BYTES or not raw.endswith(b"\n"):
                            raise ValueError("invalid HumanFrame packet")
                        arrived = time.monotonic_ns()
                        frame = json.loads(raw)
                        packets.put((frame, arrived), timeout=0.1)
                        rx_stamps.append(arrived)
        except Exception as exc:
            failures.put(exc)
        finally:
            ended.set()

    spins = threading.Thread(target=spin, name="feedback-shadow")
    spins.start()
    worker = None
    stats, rejections, traces, missing_human = [], Counter(), [], []
    args.output.mkdir()
    try:
        with socket.socket() as listener:
            listener.bind((args.host, 0))
            listener.listen(1)
            listener.settimeout(40)
            worker = threading.Thread(target=read_packets, args=(listener,), name="human-ingress")
            worker.start()
            print(f"PC2_READONLY_PORT {listener.getsockname()[1]}", flush=True)
            first, stamp = packets.get(timeout=42)
            runtime.handle(first, stamp, CLOCK)
            origin = time.monotonic_ns()
            tick = 0
            while not (ended.is_set() and time.monotonic_ns() - rx_stamps[-1] > 350_000_000):
                if not failures.empty():
                    raise failures.get()
                if time.monotonic_ns() - origin > 15_000_000_000:
                    raise TimeoutError("recorded replay exceeded time bound")
                scheduled = origin + tick * PERIOD_NS
                paced_wait(scheduled)
                start = time.monotonic_ns()
                while True:
                    try:
                        frame, arrived = packets.get_nowait()
                    except Empty:
                        break
                    if arrived > time.monotonic_ns():
                        raise ValueError("human read timestamp in the future")
                    runtime.handle(frame, arrived, CLOCK)
                sampled = time.monotonic_ns()
                with lock:
                    state = feedback.snapshot(tick, sampled)
                if state is None:
                    rejections["feedback_unavailable"] += 1
                elif max(state["source_receive_ages_s"].values()) > .12:
                    rejections["feedback_stale"] += 1
                else:
                    received = time.monotonic_ns()
                    reply = runtime.handle(state, received, CLOCK)
                    finished = time.monotonic_ns()
                    if reply.get("publish_allowed") or reply.get("reset"):
                        raise ValueError(f"unexpected advisory reply: {reply}")
                    if reply["ready"]:
                        current = state["q_rad"]
                        target = reply["q_target_rad"]
                        stats.append(dict(tick=tick, runtime_ms=(finished-received)/1e6,
                                          onnx_ms=timer.last_ms,
                                          preprocessing_ms=reply["preprocessing_ms"],
                                          cycle_ms=(finished-start)/1e6, from_due_ms=(finished-scheduled)/1e6,
                                          feedback_max_age_ms=max(state["source_receive_ages_s"].values())*1000,
                                          human_arrival_age_ms=reply["timing"]["newest_arrival_age_ms"],
                                          target_error_deg=float(np.rad2deg(np.max(np.abs(
                                              np.subtract(target, current))))),
                                          joint_names=reply["joint_names"], feedback=current, targets=target))
                        if missing_human and "first_ready_after" not in missing_human[-1]:
                            missing_human[-1]["first_ready_after"] = tick
                    else:
                        reason = reply.get("reason", "unknown")
                        rejections[reason] += 1
                        if stats:
                            if not missing_human or "first_ready_after" in missing_human[-1]:
                                missing_human.append(dict(last_ready=stats[-1]["tick"], first_rejected=tick,
                                                           rejection_reason=reason))
                            missing_human[-1]["last_rejected"] = tick
                tick += 1
                elapsed = time.monotonic_ns() - origin
                earliest = elapsed // PERIOD_NS + 1
                if tick < earliest:
                    rejections["elapsed_ticks_skipped"] += earliest - tick
                    tick = earliest
            if not failures.empty():
                raise failures.get()
    finally:
        stop.set()
        if worker is not None:
            worker.join(timeout=3)
        spins.join(timeout=3)
        if spins.is_alive() or (worker is not None and worker.is_alive()):
            raise RuntimeError("read-only endpoint did not exit")
        node.destroy_node()
        rclpy.shutdown()

    expected_packets = PACKETS - (len(DROPOUT) if args.dropout else 0)
    if len(rx_stamps) != expected_packets or len(stats) < 50:
        raise ValueError(f"insufficient observation: {len(rx_stamps)} packets {len(stats)} ready, {rejections}")
    for before, after in zip(stats, stats[1:]):
        if after["tick"] == before["tick"] + 1:
            traces.append(float(np.rad2deg(np.max(np.abs(np.subtract(after["targets"], before["targets"]))))))

    def describe(values):
        return dict(p50=float(np.median(values)), p95=float(np.percentile(values, 95)),
                    p99=float(np.percentile(values, 99)), max=float(max(values)))

    report = {name: describe([row[name] for row in stats]) for name in (
        "runtime_ms", "onnx_ms", "preprocessing_ms", "cycle_ms", "from_due_ms", "feedback_max_age_ms",
        "human_arrival_age_ms", "target_error_deg")}
    by_tick = {row["tick"]: row for row in stats}
    for event in missing_human:
        if "first_ready_after" not in event:
            continue
        before, after = (by_tick[event[key]] for key in ("last_ready", "first_ready_after"))
        delta = np.rad2deg(np.abs(np.subtract(after["targets"], after["feedback"])))
        index = int(np.argmax(delta))
        event.update(first_target_to_feedback_deg=float(delta[index]), joint=after["joint_names"][index],
                     previous_advice_to_current_advice_deg=float(np.rad2deg(np.max(np.abs(
                         np.subtract(after["targets"], before["targets"]))))))
    result = dict(human_packets=len(rx_stamps), ready_states=len(stats), rejection_counts=dict(rejections),
                  cycle_over_20ms=sum(row["cycle_ms"] > 20 for row in stats), scope="real feedback, recorded human packets",
                  ready_rate_hz=(len(stats) - 1) * 50 / (stats[-1]["tick"] - stats[0]["tick"]),
                  adjacent_pairs=len(traces), human_input_recovery_events=missing_human,
                  new_publishers=0, hardware_commands=0, sensor_profile_measured=cfg["measured_sensor_calibration"],
                  **report, arrival_gap_ms=describe(np.diff(rx_stamps)/1e6),
                  adjacent_target_max_step_deg=describe(traces) if traces else None)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (args.output / "targets.jsonl").open("x") as log:
        for row in stats:
            log.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    rx = sub.add_parser("receive")
    for name in ("policy-root", "profile", "output"):
        rx.add_argument("--" + name, type=Path, required=True)
    rx.add_argument("--host", required=True)
    rx.add_argument("--allow", required=True)
    tx = sub.add_parser("send")
    tx.add_argument("--capture", type=Path, required=True)
    tx.add_argument("--host", required=True)
    tx.add_argument("--port", type=int, required=True)
    for command in (tx, rx):
        command.add_argument("--dropout", action="store_true", help="omit the same eight archived packets")
    args = parser.parse_args()
    (receive if args.mode == "receive" else send)(args)


if __name__ == "__main__":
    main()
