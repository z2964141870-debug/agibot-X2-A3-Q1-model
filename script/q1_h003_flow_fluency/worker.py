"""Run a complete Q1 trajectory in the archived physical plant and log issued/applied targets."""

import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np
from design import D, OLD, SCENES, SEEDS, PACES, GAINS, destination
from transport import Transport

sys.path.insert(0, str(D.parent))
import bench, q1_recovery_runtime
import flow_relay as relay
from run_suite import EXPERT, TRACKS, verify, BUNDLE

sys.path.append(str(OLD))
from predictor import CommandHistory, corrected_target


def execute(split, surface, arm, kind, independent=False):
    mu, load, transport = SCENES[surface]
    components = arm.split("+")
    pace = next((PACES[x] for x in components if x in PACES), PACES["published"])
    policy = next(
        (x for x in components if x in GAINS or x in ("linear", "fixed", "motion")), "published"
    )
    seed = SEEDS[split]
    nstatic = 9 if split in ("fit", "val", "dev") else 24
    nrelay = (
        2
        if kind == "flat"
        else 5
        if split in ("fit", "val") or kind == "double"
        else 20
        if split.startswith("hold") and surface == "daily"
        else 10
    )
    out = destination(split, surface, arm, kind)
    if independent:
        out = D / "entry_replay" / split / surface / arm / kind
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
    model = None
    scale = GAINS.get(policy, 1.5 if policy == "fixed" else 1.0)
    if policy in GAINS or policy == "fixed":
        import onnxruntime as ort

        cfg = ort.SessionOptions()
        cfg.intra_op_num_threads = cfg.inter_op_num_threads = 1
        path = D / "age.onnx" if policy in GAINS else OLD / "trained/predict.onnx"
        model = ort.InferenceSession(str(path), cfg, providers=["CPUExecutionProvider"])
    queue = None
    clips = force_oob = applied_demand_oob = missing = 0

    class Instrumented(original_branch):
        def __init__(self, *args):
            super().__init__(*args)
            self.commands = CommandHistory()

        def reset(self):
            super().reset()
            self.commands = CommandHistory()

        def step(self, R, w, q, qd, previous_sent_action, base, reset_anchor):
            nonlocal log, clips, queue, missing
            if self.commands.previous is None:
                log = {}
                queue = Transport(transport, seed + 97 * len(segments))
                segments.append(log)
            target, tau, gate = super().step(R, w, q, qd, previous_sent_action, base, reset_anchor)
            f, delta = self.commands.update(
                self.tracker, self.expert, target, gate, R, w, q, qd, previous_sent_action
            )
            aux, valid, age = queue.observation(target)
            features = np.r_[f, aux]
            issued = target
            if self.commands.seen:
                correction = None
                if policy in GAINS and valid and age > 0:
                    correction = scale * model.run(None, {"features": features[None]})[0][0]
                elif policy in ("linear", "fixed") and transport != "0":
                    correction = (
                        delta
                        if policy == "linear"
                        else scale * model.run(None, {"features": f[None]})[0][0]
                    )
                if correction is not None:
                    issued, tau, c = corrected_target(self.tracker, target, q, qd, correction)
                    clips += int(c)
                if not valid:
                    missing += 1
            if split in ("fit", "val"):
                for name, value in (
                    ("features", features),
                    ("base", target),
                    ("seen", self.commands.seen),
                ):
                    log.setdefault(name, []).append(np.array(value, copy=True))
            return issued, tau, gate

    if independent:
        from age_branch import AgeAwareRecoveryBranch

        class Standalone(AgeAwareRecoveryBranch):
            def __init__(self, *args):
                super().__init__(*args, D / "age.onnx", scale)

            def step(self, *args):
                nonlocal queue, log
                if self.seq == 0:
                    queue = Transport(transport, seed + 97 * len(segments))
                    log = {}
                    segments.append(log)
                else:
                    for seq, applied in queue.events:
                        assert self.acknowledge(seq, applied)
                return super().step(*args)

        q1_recovery_runtime.Q1RecoveryBranch = Standalone
    else:
        q1_recovery_runtime.Q1RecoveryBranch = Instrumented
    original_step = bench.mujoco.mj_step

    def step(m, d, nstep=1):
        nonlocal force_oob, applied_demand_oob
        if nstep != 4:
            return original_step(m, d, nstep=nstep)
        delay = queue.submit(d.ctrl, d.qpos[7:])
        if split in ("fit", "val"):
            log.setdefault("delay", []).append(delay / 4.0)
        four = [queue.next() for _ in range(4)]
        i = 0
        while i < 4:
            j = i + 1
            while j < 4 and four[j] is four[i]:
                j += 1
            d.ctrl[:] = four[i]
            demand = (
                m.actuator_gainprm[:, 0] * (d.ctrl - d.qpos[7:])
                + m.actuator_biasprm[:, 2] * d.qvel[6:]
            )
            applied_demand_oob += int(
                np.count_nonzero(abs(demand) > m.actuator_forcerange[:, 1] + 1e-4)
            )
            original_step(m, d, nstep=j - i)
            force_oob += int(
                np.count_nonzero(abs(d.actuator_force) > m.actuator_forcerange[:, 1] + 1e-4)
            )
            i = j

    bench.mujoco.mj_step = step
    verify(BUNDLE)
    assert (
        hashlib.sha256(EXPERT.read_bytes()).hexdigest()
        == "0231328d60f07297b38a4079e01167cf9fec65fdb2c89302144ee65e0011eb92"
    )
    expert = OLD / "motion/motion_2.onnx" if policy == "motion" and transport != "0" else EXPERT
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
        if kind == "flat":
            cases = [(relay.CASES[k][0], [0.0, 0.0], 100) for k in (0, 2)]
        relay.run(expert, out, True, TRACKS, cases=cases, pace=pace, repeat_kick=kind == "double")
        (out / "cases.json").write_text(json.dumps(cases, indent=2))
    assert len(segments) == (nstatic if kind == "static" else nrelay)
    if split in ("fit", "val"):
        for i, segment in enumerate(segments):
            np.savez_compressed(out / f"signals_{i:02}.npz", **segment)
    (out / "receipt.json").write_text(
        json.dumps(
            dict(
                seed=seed,
                arm=arm,
                scene=surface,
                pace=pace,
                force_oob=force_oob,
                applied_demand_oob=applied_demand_oob,
                clipped_ticks=clips,
                missing_ack_seen_ticks=missing,
                episodes=len(segments),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    execute(*sys.argv[1:])
