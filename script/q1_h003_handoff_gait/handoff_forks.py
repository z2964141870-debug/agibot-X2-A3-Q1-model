"""Paired CPU-only forks at the published Q1 gate's first handoff decision."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import bench
from bench import mujoco, np
from check_base import verify
from recovery import Gate
import relay

EXPERT = bench.HERE / "formal_03/q1_recovery_expert.onnx"
TRACKS = bench.HERE / "formal_03/input_tracks"
HELD_OUT = [(track, tuple(s * scale for s in kick), tick)
            for (track, kick, _), scale, tick in zip(relay.CASES,
            (1.04, .96, 1.04, .96, 1.04), (113, 87, 111, 89, 114))]
def blind_cases():
    rng = np.random.default_rng(2026112002)
    cases = []
    for rep in range(4):
        for track, kick, _ in relay.CASES:
            scale = rng.uniform(.90, 1.10)
            tick = int(rng.integers(76, 126))
            cases.append((track, tuple(float(v*scale) for v in kick), tick))
    return cases

ORIGINAL_STEP = mujoco.mj_step
ORIGINAL_ADVANCE = Gate.advance
ORIGINAL_SAVE = np.savez_compressed
ORIGINAL_REFERENCE = relay.HandoffReference

class Forks:
    def __init__(self, path, branches):
        self.path = path
        self.branches = branches
        self.variant = "original"
        self.loads = []
        self.anchor = None
        self.data = None
        self.remaining = 0
        self.trial = 0

    def physics(self, model, data, **kwargs):
        ORIGINAL_STEP(model, data, **kwargs)
        self.model, self.data = model, data
        if kwargs.get("nstep") != 4:
            return
        loads = [0., 0.]
        for i, con in enumerate(data.contact):
            names = [model.geom(int(g)).name or "" for g in (con.geom1, con.geom2)]
            if "floor" not in names:
                continue
            for k, side in enumerate(("left", "right")):
                if any(n.startswith(side+"_ankle") for n in names):
                    force = np.empty(6)
                    mujoco.mj_contactForce(model, data, i, force)
                    loads[k] += max(0., force[0])
        self.loads.append(loads)

    def advance(self, gate, x, width):
        if self.remaining:
            self.remaining -= 1
            if self.remaining:
                return 1., False
            gate.mode = "handoff"
            gate.blend_ticks = 0
            return ORIGINAL_ADVANCE(gate, x, width)
        g, reanchor = ORIGINAL_ADVANCE(gate, x, width)
        if not reanchor or self.anchor is not None:
            return g, reanchor
        # The fork occurs before either controller, CGFA, reference or physics
        # advances this tick. Each child inherits the entire address space.
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, self.data, state, spec)
        self.anchor = (len(self.loads), hashlib.sha256(state.tobytes()).hexdigest())
        for variant in self.branches:
            pid = os.fork()
            if pid == 0:
                self.variant = variant
                if variant.startswith("delay"):
                    self.remaining = int(variant[5:])
                    gate.mode, gate.blend_ticks = "recover", 0
                    return 1., False
                return g, reanchor
            _, status = os.waitpid(pid, 0)
            if os.waitstatus_to_exitcode(status) != 0:
                raise RuntimeError(f"Branch {variant} failed in trial {self.trial}")
        return g, reanchor

    def reference(self, *args):
        if self.variant.startswith("ramp"):
            return ORIGINAL_REFERENCE(*args,resume_ticks=int(self.variant[4:]))
        if self.variant=="pose15xy30":
            return ORIGINAL_REFERENCE(*args,resume_ticks=15,translation_ticks=30)
        if self.variant=="pose30xy15":
            return ORIGINAL_REFERENCE(*args,resume_ticks=30,translation_ticks=15)
        return ORIGINAL_REFERENCE(*args)

    def save(self, file, **arrays):
        dest = self.path / self.variant / Path(file).name
        dest.parent.mkdir(exist_ok=True)
        assert len(self.loads) == len(arrays["trace"])
        ORIGINAL_SAVE(dest, **arrays, loads=self.loads,
                      fork_tick=-1 if self.anchor is None else self.anchor[0],
                      fork_hash="" if self.anchor is None else self.anchor[1])
        if self.variant != "original":
            os._exit(0)
        self.trial += 1
        self.loads = []
        self.anchor = None

def run(out, suite, variants):
    verify(bench.BUNDLE)
    assert hashlib.sha256(EXPERT.read_bytes()).hexdigest() == "0231328d60f07297b38a4079e01167cf9fec65fdb2c89302144ee65e0011eb92"
    out.mkdir(parents=True,exist_ok=False)
    recorder = Forks(out, variants)
    def gate_step(gate,x,width):
        return recorder.advance(gate,x,width)
    with patch.object(mujoco, "mj_step", recorder.physics), patch.object(Gate,"advance",gate_step), \
         patch.object(np, "savez_compressed",recorder.save), patch.object(relay,"HandoffReference",recorder.reference):
        if suite == "e1":
            bench.evaluate(out/"runner", 20261119, EXPERT, public_runtime=True)
        else:
            cases = relay.CASES if suite=="relay" else HELD_OUT
            if suite=="blind": cases=blind_cases()
            if suite=="nominal":
                cases=[(track,(0.,0.),100) for track in ("straight_walk_50hz.csv","loop_figure8_walk.csv")]
            (out/"cases.json").write_text(json.dumps(cases,indent=2))
            relay.run(EXPERT, out/"runner", True, TRACKS, cases=cases)
    # Historical trajectory parity and all common prefixes are audited before scoring.
    audit = {}
    for original in sorted((out/"original").glob("*.npz")):
        with np.load(original) as base:
            anchor = int(base["fork_tick"])
            row = {"fork_tick":anchor, "branches":[]}
            if suite in ("e1","relay"):
                source = bench.HERE/"formal_03"/("pinned_e1" if suite=="e1" else "pinned_relay")/original.name
                with np.load(source) as old:
                    for key in ("qpos","trace"):
                        assert np.array_equal(old[key],base[key]),(suite,original.name,key)
                row["published_trajectory_exact"]=True
            for variant in variants:
                if anchor<0: continue
                with np.load(out/variant/original.name) as branch:
                    assert branch["fork_hash"]==base["fork_hash"] and branch["fork_tick"]==anchor
                    for key in ("trace","qpos","loads"):
                        assert np.array_equal(branch[key][:anchor],base[key][:anchor])
                row["branches"].append(variant)
            audit[original.name] = row
    (out/"audit.json").write_text(json.dumps(audit,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    p.add_argument("suite",choices=["e1","relay","held","nominal","blind"])
    p.add_argument("variants",nargs="*",choices=["delay10","delay20","ramp15","ramp50","clone","pose15xy30","pose30xy15"])
    args=p.parse_args()
    run(args.output.resolve(),args.suite,args.variants)

