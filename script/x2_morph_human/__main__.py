"""Mac/SoC1 local development CLI. There is intentionally no powered-control mode."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import platform
from pathlib import Path
import sys
import tarfile
import time

import numpy as np

from . import artifacts
from .core import Calibration, DT_NS, PARENTS, SmplAdapter, axis_angle, inv, mul, rotate


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def new_output(path):
    path.mkdir(parents=True, exist_ok=False)
    return path


def adapter(root, calibration=None):
    c = Calibration(**json.loads(calibration.read_text())) if calibration else Calibration("demo_smpl_yup")
    return SmplAdapter(root/"upstream/deploy/vr_teleop/human_skeleton.json", c)


def demo_stream(a, count):
    """Synthetic stand+small arm motion, not captured human data or quality test."""
    stand = a.doc["stand"]
    world = np.asarray(stand["quat"])
    q = np.empty((24, 4))
    q[0] = mul(inv(np.asarray(a.calibration.world_from_smpl_wxyz)), world[0])
    for i in range(1, 24):
        q[i] = mul(inv(world[PARENTS[i]]), world[i])
    q = q*np.where(q[:, :1] < 0, -1, 1)
    angle = 2*np.arctan2(np.linalg.norm(q[:, 1:], axis=-1), q[:, 0])
    den = np.linalg.norm(q[:, 1:], axis=-1)
    local = q[:, 1:]*(angle/np.maximum(den, 1e-12))[:, None]
    root = rotate(inv(np.asarray(a.calibration.world_from_smpl_wxyz)), np.asarray(stand["pos"])[0])
    for i in range(count):
        pose = local.copy()
        pose[18, 2] += .12*np.sin(2*np.pi*i/100)
        pose[19, 2] -= .12*np.sin(2*np.pi*i/100)
        yield a.packet(pose, root, sequence=i, capture_ns=i*DT_NS,
                       capture_clock="synthetic_capture", arrival_ns=i*DT_NS,
                       arrival_clock="replay_monotonic")


def doctor(args):
    import onnxruntime as ort
    from .runtime import Runtime, synthetic_state
    root = artifacts.home()
    checked = artifacts.verify(root, meshes=args.meshes)
    r = Runtime(root)
    for p in demo_stream(adapter(root), 10):
        r.push(p)
    now = 9*DT_NS
    out = r.state(synthetic_state(r, 0, now), now, "replay_monotonic")
    doc = dict(status="LOCAL_INFERENCE_READY", machine=platform.machine(), system=platform.platform(),
               python=sys.version, providers_available=ort.get_available_providers(),
               providers_used=r.runner.session.get_providers(), assets=checked,
               input_shapes={i.name: i.shape for i in r.runner.session.get_inputs()},
               advisory_target_count=len(out["q_target_rad"]),
               history_dim=930, human_feature_dim=2160, publish_allowed=False,
               hardware_backend_status="NOT_CONNECTED_NOT_INTEGRATED", preview_span_ms=180,
               orin_latency_status="NOT_MEASURED", source_clock_latency_status="UNKNOWN")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, doc)
    print(json.dumps(doc, indent=2))


def convert(args):
    root = artifacts.activate()
    a = adapter(root, args.calibration)
    # Loading arbitrary pickle/object arrays is intentionally disabled.
    with np.load(args.input, allow_pickle=False) as d:
        poses, roots, times = d["local_axis_angle"], d["root_translation_m"], d["capture_ns"]
        clock = str(d["capture_clock"].item())
        if poses.ndim != 3 or poses.shape[1:] != (24, 3) or roots.shape != (len(poses), 3):
            raise ValueError("expected SMPL24 local_axis_angle[T,24,3], root_translation_m[T,3]")
        if times.shape != (len(poses),) or times.dtype.kind not in "iu" or len(times) < 10 or (np.diff(times) <= 0).any():
            raise ValueError("capture_ns must increase; at least ten frames")
        with open(args.output, "x") as f:
            for i, (pose, position, t) in enumerate(zip(poses, roots, times)):
                p = a.packet(pose, position, sequence=i, capture_ns=int(t), capture_clock=clock,
                             arrival_ns=int(t-times[0]), arrival_clock="replay_monotonic")
                f.write(json.dumps(p, allow_nan=False)+"\n")
    print(json.dumps(dict(frames=len(times), output=str(args.output), calibration=a.calibration.id,
                          input_sha256=artifacts.sha(args.input), arrival_clock="synthetic_replay_not_measured")))


def fgp_convert(args):
    from .fgp_bridge import convert_fgp_npz
    res = convert_fgp_npz(args.input, args.calibration, args.output,
                          align_initial_ground=args.align_initial_ground)
    print(json.dumps(res, indent=2))


def synth_motion(args):
    from .fgp_bridge import generate_synth_smpl_npz
    res = generate_synth_smpl_npz(args.preset, args.frames, args.output)
    print(json.dumps(res, indent=2))


def report_html(output, rows, summary):
    # Small self-contained plots; actual runtime samples, no invented performance.
    data = json.dumps(rows, allow_nan=False).replace("<", "\u003c")
    summary_text = html.escape(json.dumps(summary, indent=2))
    video = '<video controls src="preview.mp4" width="640"></video>' if (output/"preview.mp4").exists() else ""
    text = """<!doctype html><meta charset="utf-8"><title>X2 human local integration</title>
<style>body{font:15px system-ui;max-width:1200px;margin:30px auto;background:#f6f7fa;color:#172232}
canvas{background:white;border:1px solid #ccc;width:100%;height:240px}#grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}pre{white-space:pre-wrap}video{max-width:100%}</style>
<h1>SMPL → MorphMimic human → X2</h1><p>Local integration evidence. Advisory targets only. Synthetic human source / development simulation are explicitly labelled. No hardware quality or end-to-end latency claim.</p>"""+video+"<div id=grid></div><pre>"+summary_text+"</pre><script>const rows="+data+";"+"""
const plots=[['Full runtime ms',r=>[r.full_runtime_ms]],['Left arm targets rad',r=>r.q_target_rad.slice(15,22)],['Right arm targets rad',r=>r.q_target_rad.slice(22,29)],['Leg targets rad',r=>r.q_target_rad.slice(0,12)],['Pelvis height m (simulation only)',r=>[r.simulation?.pelvis_z_m??0]],['Pelvis tilt deg (simulation only)',r=>[r.simulation?.pelvis_tilt_deg??0]]];
const colors=['#0072b2','#d55e00','#009e73','#cc79a7','#e69f00','#56b4e9','#333'];
for(const [title,fn] of plots){const box=document.createElement('div');box.innerHTML='<h3>'+title+'</h3><canvas width=550 height=240></canvas>';document.querySelector('#grid').append(box);const c=box.querySelector('canvas').getContext('2d');const a=rows.map(fn), flat=a.flat(),lo=Math.min(...flat),hi=Math.max(...flat),range=Math.max(hi-lo,1e-5);c.font='12px system-ui';c.fillText(hi.toFixed(3),5,15);c.fillText(lo.toFixed(3),5,220);if(!a.length)continue;for(let k=0;k<a[0].length;k++){c.beginPath();c.strokeStyle=colors[k%colors.length];a.forEach((v,i)=>{let x=50+480*i/Math.max(a.length-1,1),y=210-185*(v[k]-lo)/range;i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke()}}
</script>"""
    (output/"report.html").write_text(text)


def replay(args):
    from .core import NotReady
    from .runtime import Runtime, synthetic_state
    root = artifacts.home()
    max_gap_ns = int(round(args.max_interp_gap_ms * 1e6))
    fresh_ms = args.max_freshness_ms if args.max_freshness_ms is not None else (120.0 if args.clock_50hz else 3000.0)
    r = Runtime(root, human_points=args.human_points, max_interp_gap_ns=max_gap_ns, max_age_ns=int(round(fresh_ms * 1e6)))
    out = new_output(args.output)
    start = time.perf_counter()
    if args.human_jsonl:
        with open(args.human_jsonl) as f:
            packets = [json.loads(x) for x in f if x.strip()]
        source = dict(kind="recorded_human_frames", sha256=artifacts.sha(args.human_jsonl))
    else:
        packets = list(demo_stream(adapter(root), args.frames+9))
        source = dict(kind="SYNTHETIC_SMPL_STAND_AND_ARM_MOTION", real_human_capture=False)
    states = None
    if args.state_jsonl:
        with open(args.state_jsonl) as f:
            states = [json.loads(x) for x in f if x.strip()]
    sim, renderer, writer = None, None, None
    if args.simulate:
        from .simulation import development_scene, DevelopmentSim
        sim = DevelopmentSim(r, development_scene(root, out/"development_scene.xml"))
        if args.video:
            import imageio.v2 as imageio
            import mujoco
            renderer = mujoco.Renderer(sim.model, height=480, width=640)
            writer = imageio.get_writer(out/"preview.mp4", fps=50, macro_block_size=1)
    rows, hold_events = [], []
    waiting, midstream_holds = 0, 0
    try:
        with open(out/"input_human.jsonl", "x") as f, open(out/"targets.jsonl", "x") as target_file:
            if args.clock_50hz:
                if not packets:
                    raise ValueError("no input packets")
                for p in packets:
                    if p.get("arrival_clock") != "replay_monotonic":
                        raise ValueError("offline replay expects replay_monotonic arrivals; use convert or fgp-convert")
                pkt_idx = 0
                first_arrival = int(packets[0]["arrival_ns"])
                last_arrival = int(packets[-1]["arrival_ns"])
                tick_idx = 0
                state_seq = 0
                while True:
                    now = first_arrival + tick_idx * DT_NS
                    if now > last_arrival or len(rows) >= args.frames:
                        break
                    while pkt_idx < len(packets) and int(packets[pkt_idx]["arrival_ns"]) <= now:
                        r.push(packets[pkt_idx])
                        f.write(json.dumps(packets[pkt_idx], allow_nan=False)+"\n")
                        pkt_idx += 1
                    if sim:
                        state = sim.measurement(state_seq, now)
                    elif states is not None:
                        if len(rows) >= len(states):
                            break
                        state = states[len(rows)]
                    else:
                        state = synthetic_state(r, state_seq, now)
                    state_seq += 1
                    try:
                        row = r.state(state, now, "replay_monotonic")
                    except NotReady as exc:
                        if not rows:
                            waiting += 1
                        else:
                            midstream_holds += 1
                            hold_step = dict(tick_index=tick_idx, now_ns=now, reason=str(exc))
                            if sim:
                                hold_step["simulation"] = sim.step(sim.last_targets)
                                if renderer is not None:
                                    writer.append_data(sim.video_frame(renderer))
                                if hold_step["simulation"]["fallen"]:
                                    hold_events.append(hold_step)
                                    break
                            hold_events.append(hold_step)
                        tick_idx += 1
                        continue
                    if sim:
                        row["simulation"] = sim.step(row["q_target_rad"])
                    row["tick_index"] = tick_idx
                    row["tick_ns"] = now
                    rows.append(row)
                    target_file.write(json.dumps(row, allow_nan=False)+"\n")
                    if renderer is not None:
                        writer.append_data(sim.video_frame(renderer))
                    if sim and row["simulation"]["fallen"]:
                        break
                    tick_idx += 1
            else:
                for p in packets:
                    if len(rows) >= args.frames:
                        break
                    r.push(p)
                    f.write(json.dumps(p, allow_nan=False)+"\n")
                    now = p["arrival_ns"]
                    clock = p["arrival_clock"]
                    if clock != "replay_monotonic":
                        raise ValueError("offline replay expects replay_monotonic arrivals; use convert or record")
                    if sim:
                        state = sim.measurement(len(rows), now)
                    elif states is not None:
                        if len(rows) >= len(states):
                            break
                        state = states[len(rows)]
                    else:
                        state = synthetic_state(r, len(rows), now)
                    try:
                        row = r.state(state, now, clock)
                    except NotReady:
                        waiting += 1
                        continue
                    if sim:
                        row["simulation"] = sim.step(row["q_target_rad"])
                    rows.append(row)
                    target_file.write(json.dumps(row, allow_nan=False)+"\n")
                    if renderer is not None:
                        writer.append_data(sim.video_frame(renderer))
                    if sim and row["simulation"]["fallen"]:
                        break
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    if hold_events:
        write_json(out/"hold_events.json", hold_events)
    if not rows:
        raise ValueError("no complete human window; inputs retained for diagnosis")
    timings = np.array([x["full_runtime_ms"] for x in rows])
    sim_metrics = None
    if sim:
        s_rows = [x["simulation"] for x in rows]
        l_clr = np.array([s["left_foot_clearance_m"] for s in s_rows])
        r_clr = np.array([s["right_foot_clearance_m"] for s in s_rows])
        slips = np.array([s["stance_slip_speed_m_s"] for s in s_rows])
        tilts = np.array([s["pelvis_tilt_deg"] for s in s_rows])
        torques = np.array([s["max_torque_ratio"] for s in s_rows])
        p_pos = np.array([s["pelvis_pos_m"] for s in s_rows])
        sim_metrics = dict(
            pelvis_z_min_m=float(np.min(p_pos[:, 2])),
            pelvis_z_median_m=float(np.median(p_pos[:, 2])),
            pelvis_xy_displacement_m=float(np.hypot(p_pos[-1, 0]-p_pos[0, 0], p_pos[-1, 1]-p_pos[0, 1])),
            pelvis_tilt_p95_deg=float(np.percentile(tilts, 95)),
            pelvis_tilt_max_deg=float(np.max(tilts)),
            left_foot_max_clearance_m=float(np.max(l_clr)),
            right_foot_max_clearance_m=float(np.max(r_clr)),
            stance_slip_median_m_s=float(np.median(slips)),
            stance_slip_p95_m_s=float(np.percentile(slips, 95)),
            peak_torque_ratio_p95=float(np.percentile(torques, 95)),
            peak_torque_ratio_max=float(np.max(torques)),
        )
    summary = dict(status="LOCAL_INTEGRATION_RAN", source=source,
                   robot_state_source="development_MuJoCo" if sim else ("recorded" if states is not None else "SYNTHETIC_FIXED_DEFAULT_STATE"),
                   human_points=args.human_points,
                   clock_50hz=bool(args.clock_50hz),
                   max_interp_gap_ms=float(args.max_interp_gap_ms),
                   frames=len(rows), startup_wait_packets=waiting,
                   midstream_hold_ticks=midstream_holds,
                   wall_s=time.perf_counter()-start,
                   providers=r.runner.session.get_providers(), policy_sha256=artifacts.MODEL_SHA,
                   full_runtime_ms=dict(zip(("p50", "p95", "p99", "max"),
                                            [float(np.percentile(timings, p)) for p in (50,95,99,100)])),
                   simulation_metrics=sim_metrics,
                   preview_span_ms=180, hardware_end_to_end_latency_ms=None,
                   development_simulation_only=bool(sim), original_training_plant_equivalence="NOT_VERIFIED",
                   simulation_fallen=bool(sim and rows[-1]["simulation"]["fallen"]),
                   hardware_commands=0, training_updates=0, publish_allowed=False)
    write_json(out/"summary.json", summary)
    report_html(out, rows, summary)
    write_json(out/"files.json", {p.name:dict(bytes=p.stat().st_size,sha256=artifacts.sha(p))
                                   for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(summary, indent=2))


def serve(args):
    import zmq
    from .runtime import Runtime
    # No robot routes, no arbitrary remote binding in this development service.
    if not args.endpoint.startswith("tcp://127.0.0.1:"):
        raise ValueError("development advisory service binds to loopback only")
    max_gap_ns = int(round(args.max_interp_gap_ms * 1e6))
    r = Runtime(human_points=args.human_points, max_interp_gap_ns=max_gap_ns)
    clock = f"local_monotonic:{platform.node()}:{time.monotonic_ns()}"
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.MAXMSGSIZE, 256*1024)
    socket.bind(args.endpoint)
    print(json.dumps(dict(status="ADVISORY_ONLY", endpoint=args.endpoint,
                          human_points=args.human_points, max_interp_gap_ms=args.max_interp_gap_ms,
                          publish_allowed=False)), flush=True)
    try:
        while True:
            msg = socket.recv()
            try:
                request = json.loads(msg)
                reply = r.handle(request, time.monotonic_ns(), clock)
            except Exception as exc:
                # Exceptions never become a stale/zero target. Clear policy history.
                r.reset()
                reply = dict(ready=False, error=str(exc), reset=True, publish_allowed=False)
            socket.send_json(reply)
    except KeyboardInterrupt:
        pass
    finally:
        socket.close(); context.term()


def bundle(args):
    root = artifacts.home()
    artifacts.verify(root, meshes=True)
    package = artifacts.PACKAGE
    entries = [(p, "script/x2_morph_human/"+str(p.relative_to(package)))
               for p in sorted(package.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]
    entries += [(root/row["path"], "data/runtime/x2_morph_human/"+row["path"])
                for row in artifacts.manifest()["files"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "xb") as raw, tarfile.open(fileobj=raw, mode="w:gz") as tar:
        for path, dest in entries:
            tar.add(path, arcname=dest, recursive=False)
    # Verify exact portable payload, not just tar creation success.
    with tarfile.open(args.output) as tar:
        members = tar.getmembers()
        if len(members) != len(entries):
            raise ValueError("bundle member count mismatch")
        for path, name in entries:
            f = tar.extractfile(name)
            if f is None or hashlib.sha256(f.read()).hexdigest() != artifacts.sha(path):
                raise ValueError(f"bundle bytes mismatch: {name}")
    row = dict(path=str(args.output), bytes=args.output.stat().st_size, sha256=artifacts.sha(args.output),
               members=len(entries), status="PORTABLE_SOURCE_AND_ASSETS_VERIFIED",
               mac_venv_included=False, orin_wheels_included=False, hardware_backend_integrated=False)
    write_json(args.output.with_suffix(".manifest.json"), row)
    print(json.dumps(row, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor"); p.add_argument("--meshes", action="store_true");p.add_argument("--output", type=Path);p.set_defaults(fn=doctor)
    p = sub.add_parser("convert");p.add_argument("--input", type=Path, required=True);p.add_argument("--calibration", type=Path, required=True);p.add_argument("--output", type=Path, required=True);p.set_defaults(fn=convert)
    p = sub.add_parser("fgp-convert");p.add_argument("--input", type=Path, required=True);p.add_argument("--calibration", type=Path, required=True);p.add_argument("--output", type=Path, required=True);p.add_argument("--align-initial-ground", action="store_true");p.set_defaults(fn=fgp_convert)
    p = sub.add_parser("synth-motion");p.add_argument("--preset", choices=("walk_start_stop", "turn_and_wave"), required=True);p.add_argument("--frames", type=int, default=400);p.add_argument("--output", type=Path, required=True);p.set_defaults(fn=synth_motion)
    p = sub.add_parser("replay");p.add_argument("--human-jsonl", type=Path);p.add_argument("--state-jsonl", type=Path);p.add_argument("--output", type=Path, required=True);p.add_argument("--frames", type=int, default=150);p.add_argument("--human-points", choices=("full", "five"), default="full");p.add_argument("--max-interp-gap-ms", type=float, default=40.0);p.add_argument("--max-freshness-ms", type=float, default=None);p.add_argument("--clock-50hz", action="store_true");p.add_argument("--simulate", action="store_true");p.add_argument("--video", action="store_true");p.set_defaults(fn=replay)
    p = sub.add_parser("serve");p.add_argument("--endpoint", default="tcp://127.0.0.1:5765");p.add_argument("--human-points", choices=("full", "five"), default="full");p.add_argument("--max-interp-gap-ms", type=float, default=100.0);p.add_argument("--max-freshness-ms", type=float, default=120.0);p.set_defaults(fn=serve)
    p = sub.add_parser("bundle");p.add_argument("--output", required=True, type=Path);p.set_defaults(fn=bundle)
    args = parser.parse_args()
    if args.command == "replay" and (not 1 <= args.frames <= 3000 or (args.video and not args.simulate)):
        parser.error("frames must be 1..3000; video requires simulate")
    if args.command == "replay" and args.simulate and args.state_jsonl:
        parser.error("choose recorded states or simulation")
    args.fn(args)


if __name__ == "__main__":
    main()
