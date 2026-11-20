"""Run a complete Q1 trajectory in the archived physical plant and log issued/applied targets."""

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np
from protocol import ROOT, SURFACES, SCHEDULE
from transfer_protocol import EXTRA_SURFACES, EXTRA_SCHEDULE

sys.path.insert(0, str(ROOT.parent))
import bench, relay, q1_recovery_runtime
from run_suite import EXPERT, TRACKS, verify, BUNDLE
from predictor import CommandHistory, corrected_target


def execute(split, surface, arm, kind):
    mu, load, transport = (SURFACES | EXTRA_SURFACES)[surface]
    compensated = transport in ("all", "half", "long") or arm.startswith("unconditional_")
    policy = arm
    if arm.startswith("unconditional_"):
        selection_file = "selection.json" if arm == "unconditional_net" else "motion_selection.json"
        policy = json.loads((ROOT / selection_file).read_text())["selected"]
    seed, nstatic, nrelay = (SCHEDULE | EXTRA_SCHEDULE)[split]
    out = ROOT / "runs" / split / surface / arm / kind
    bench.HERE = out.parent / (kind + "_scene")
    original_scene = bench.scene

    def scene(path):
        original_scene(path)
        xml = path / "fall_hard.xml"
        tree = ET.parse(xml)
        tree.find(".//geom[@name='floor']").set("friction", f"{mu} 0.005 0.0001")
        tree.write(xml)
        if load:
            robot_path = path / "supermimic_abs.xml"
            robot = ET.parse(robot_path)
            pelvis = robot.find(".//body[@name='pelvis']/inertial")
            pelvis.set("mass", str(float(pelvis.get("mass")) + load))
            robot.write(robot_path)
        m = bench.mujoco.MjModel.from_xml_path(str(xml))
        assert (
            np.isclose(m.body("pelvis").mass[0], 1.82817 + load)
            and m.geom("floor").friction[0] == mu
        )
        return m

    bench.scene = relay.scene = scene
    original_branch = q1_recovery_runtime.Q1RecoveryBranch
    segments = []
    log = None
    last_data = pending = older = None
    gate = 0.0
    clips = 0
    delivered_checks = delivered_over = sampled_force_over = 0
    max_demand_ratio = 0.0
    model = None
    scale = 1.0
    if policy.startswith("net"):
        import onnxruntime as ort

        cfg = ort.SessionOptions()
        cfg.intra_op_num_threads = cfg.inter_op_num_threads = 1
        scale = float(policy.split("_")[1])
        model = ort.InferenceSession(
            str(ROOT / "trained" / "predict.onnx"), cfg, providers=["CPUExecutionProvider"]
        )
    if arm.startswith("linear_"):
        scale = float(policy.split("_")[1])

    class Instrumented(original_branch):
        def __init__(self, *args):
            super().__init__(*args)
            self.commands = CommandHistory()

        def reset(self):
            super().reset()
            self.commands = CommandHistory()

        def step(self, R, w, q, qd, previous_sent_action, base, reset_anchor):
            nonlocal gate, log, clips
            new_episode = self.commands.previous is None
            target, tau, gate = super().step(R, w, q, qd, previous_sent_action, base, reset_anchor)
            if new_episode:
                log = {}
                segments.append(log)
            features, delta = self.commands.update(
                self.tracker, self.expert, target, gate, R, w, q, qd, previous_sent_action
            )
            issued = target
            if (
                self.commands.seen
                and compensated
                and (arm.startswith("linear") or model is not None)
            ):
                if arm.startswith("linear"):
                    correction = scale * delta
                else:
                    correction = model.run(None, {"features": features[None]})[0][0] * scale
                issued, tau, clipped = corrected_target(self.tracker, target, q, qd, correction)
                clips += int(clipped)
            for k, value in (
                ("features", features),
                ("base", target),
                ("sent", issued),
                ("gate", gate),
                ("seen", self.commands.seen),
            ):
                log.setdefault(k, []).append(np.array(value, copy=True))
            return issued, tau, gate

    q1_recovery_runtime.Q1RecoveryBranch = Instrumented
    original_step = bench.mujoco.mj_step

    def step(m, d, nstep=1):
        nonlocal \
            last_data, \
            pending, \
            older, \
            delivered_checks, \
            delivered_over, \
            sampled_force_over, \
            max_demand_ratio

        def demand(setpoint):
            nonlocal delivered_checks, delivered_over, max_demand_ratio
            tau = (
                m.actuator_gainprm[:, 0] * (setpoint - d.qpos[7:])
                + m.actuator_biasprm[:, 2] * d.qvel[6:]
            )
            ratio = np.abs(tau) / m.actuator_forcerange[:, 1]
            delivered_checks += len(ratio)
            delivered_over += int(
                np.count_nonzero(np.abs(tau) > m.actuator_forcerange[:, 1] + 1e-4)
            )
            max_demand_ratio = max(max_demand_ratio, float(ratio.max()))

        if nstep != 4:
            return original_step(m, d, nstep=nstep)
        if last_data is not d:
            pending = d.qpos[7:].copy()
            older = pending.copy()
            last_data = d
        issued = d.ctrl.copy()
        delayed = (
            transport == "all"
            or transport == "recover"
            and gate > 0
            or transport == "outside"
            and gate == 0
        )
        first = (
            older if transport == "long" else pending if delayed or transport == "half" else issued
        )
        last = pending if transport == "long" else issued if transport == "half" else first
        older, pending = pending, issued
        log.setdefault("applied", []).append(first.copy())
        log.setdefault("applied_end", []).append(last.copy())
        d.ctrl[:] = first
        if split == "test2":
            demand(first)
        if transport in ("half", "long"):
            original_step(m, d, nstep=2)
            d.ctrl[:] = last
            if split == "test2":
                demand(last)
            original_step(m, d, nstep=2)
        else:
            original_step(m, d, nstep=nstep)
        if split == "test2":
            sampled_force_over += int(
                np.count_nonzero(np.abs(d.actuator_force) > m.actuator_forcerange[:, 1] + 1e-4)
            )

    bench.mujoco.mj_step = step
    verify(BUNDLE)
    assert (
        hashlib.sha256(EXPERT.read_bytes()).hexdigest()
        == "0231328d60f07297b38a4079e01167cf9fec65fdb2c89302144ee65e0011eb92"
    )
    expert = (
        ROOT / "motion" / (policy + ".onnx")
        if policy.startswith("motion_") and compensated
        else EXPERT
    )
    if kind == "static":
        if nstatic < 24:
            poses = bench.poses
            ids = (0, 2, 5, 8, 11, 13, 16, 19, 22)
            bench.poses = lambda c, s: (x for i, x in enumerate(poses(c, s)) if i in ids)
        bench.evaluate(out, seed, expert, public_runtime=True)
    else:
        rng = np.random.default_rng(seed)
        cases = []
        for i in range(nrelay):
            track, kick, _ = relay.CASES[i % 5]
            angle = rng.uniform(-0.1, 0.1)
            mag = rng.uniform(0.92, 1.08)
            c, s = np.cos(angle), np.sin(angle)
            x, y = kick
            cases.append(
                (
                    track,
                    [float(mag * (c * x - s * y)), float(mag * (s * x + c * y))],
                    int(rng.integers(80, 121)),
                )
            )
        relay.run(expert, out, True, TRACKS, cases=cases)
        (out / "cases.json").write_text(json.dumps(cases, indent=2))
    assert len(segments) == (nstatic if kind == "static" else nrelay)
    summary = []
    for i, segment in enumerate(segments):
        d = {k: np.array(v) for k, v in segment.items()}
        assert all(np.isfinite(v).all() for v in d.values())
        if transport == "all":
            assert np.array_equal(d["applied"][1:], d["sent"][:-1])
        if transport == "half":
            assert np.array_equal(d["applied"][1:], d["sent"][:-1]) and np.array_equal(
                d["applied_end"], d["sent"]
            )
        if transport == "long":
            assert np.array_equal(d["applied"][2:], d["sent"][:-2]) and np.array_equal(
                d["applied_end"][1:], d["sent"][:-1]
            )
        if transport == "now":
            assert np.array_equal(d["applied"], d["sent"])
        g = d["gate"]
        engaged = np.flatnonzero(g > 0)
        handoff = (g > 0) & (g < 1)
        released = bool(len(engaged) and np.any(g[engaged[0] :] == 0))
        summary.append(
            dict(
                ticks=len(g),
                active=int(np.sum(g > 0)),
                handoff=bool(handoff.any()),
                released=released,
                correction_rms=float(np.sqrt(np.mean((d["base"] - d["sent"]) ** 2))),
                queue_rms=float(np.sqrt(np.mean((d["applied"] - d["sent"]) ** 2))),
            )
        )
        if split.startswith("collect") or split == "valid":
            np.savez_compressed(
                out / f"signals_{i:02}.npz",
                features=d["features"],
                base=d["base"],
                seen=d["seen"],
                gate=g,
            )
    (out / "receipt.json").write_text(
        json.dumps(
            dict(
                seed=seed,
                arm=arm,
                friction=mu,
                load=load,
                transport=transport,
                clipped_ticks=clips,
                delivered_joint_checks=delivered_checks,
                delivered_demand_over=delivered_over,
                max_demand_ratio=max_demand_ratio,
                final_microstep_force_oob=sampled_force_over,
                expert_sha256=hashlib.sha256(expert.read_bytes()).hexdigest(),
                episodes=summary,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    execute(*sys.argv[1:])
