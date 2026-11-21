"""Matched floor rollouts using measured poses and archived vendor commands."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from common import CONFIG, EFF, HERE, KD, KP, Plant, STAND, mj


def sample_record(stem):
    path = Path.home() / "projects/Q1_deploy/real_logs" / (stem + ".npz")
    log = np.load(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    order = [log["joint_names"].tolist().index(j) for j in CONFIG["joint_names"]]
    start, stop = [
        next(t["t_rel_s"] for t in meta["transitions"] if t["desc"] == desc)
        for desc in ("STAND_UP", "BIPED_WALK_RUN")
    ]
    grid = np.arange(start, stop + .48, .02)
    arrays = {}
    for key in ("q", "command_position", "command_stiffness", "command_damping", "command_effort", "command_velocity"):
        prefix = "joint_state" if key == "q" else "joint_command"
        times = (log[prefix + "_stamp"] - log["event_ns"].item()) * 1e-9
        arrays[key] = np.stack([np.interp(grid, times, log[key][:, i]) for i in order], axis=1)
    arrays["digest"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return arrays


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    tracks = [sample_record(s) for s in ("q1_official_standup_from_mcap", "q1_official_standup_faceup_20261120")]
    cases = json.loads((HERE / "development/stand01_static/initialization.json").read_text())[:16]
    forms = ["measured", "command_pos", "vendor_pd", "vendor_pd_ff"]
    kwargs = [(i, f, rate) for i in range(len(cases)) for f in forms for rate in (.75, 1.)]
    plant = Plant(len(kwargs), out / "scene")
    rows = []
    for case in cases:
        data = mj.MjData(plant.model)
        data.qpos[:] = case["qpos"]
        data.ctrl[:] = data.qpos[7:]
        mj.mj_forward(plant.model, data)
        mj.mj_step(plant.model, data, nstep=50)
        rows.append(np.r_[data.qpos, data.qvel, data.qpos[7:]])
    initial = np.array([rows[i] for i, _, _ in kwargs])
    obs = plant.reset(initial)
    trace = []
    for tick in range(540):
        targets = []
        for j, (i, form, rate) in enumerate(kwargs):
            rec = tracks[i // 8]
            index = min(int(max(0, tick-25)*rate), len(rec["q"])-1)
            q = obs[j, :22] + STAND
            qd = obs[j, 22:44]*10
            if form == "measured":
                command = rec["q"][index]
            elif form == "command_pos":
                command = rec["command_position"][index]
            else:
                torque = rec["command_stiffness"][index]*(rec["command_position"][index]-q)
                torque += rec["command_damping"][index]*(rate*rec["command_velocity"][index]-qd)
                if form == "vendor_pd_ff":
                    torque += rec["command_effort"][index]
                command = q + (np.clip(torque,-EFF,EFF)+KD*qd)/KP
            blend = .5-.5*np.cos(np.pi*min(1,(tick+1)/25))
            targets.append(initial[j,7:29] + blend*(command-initial[j,7:29]))
        obs, metrics = plant.step(np.array(targets))
        trace.append(metrics)
    plant.close()
    trace = np.array(trace)
    results = []
    for j,(i,form,rate) in enumerate(kwargs):
        t = trace[:,j]
        ready = (t[:,0]>.28)&(t[:,1]>.75)&(t[:,2]>10)&(t[:,3]==0)
        upright = (t[:,0]>.35)&(t[:,1]>.92)&(t[:,2]>10)&(t[:,3]==0)
        results.append(dict(trial=i,pose=cases[i]["pose"],form=form,rate=rate,
            foot_supported=bool(ready.any()),upright=bool(upright.any()),max_up=float(t[:,1].max()),
            max_foot_supported_height=float(np.max(np.where((t[:,1]>.65)&(t[:,3]==0)&(t[:,2]>10),t[:,0],0))),
            peak_qd=float(t[:,4].max()),ground_mm=float(t[:,5].max()),self_mm=float(t[:,6].max())))
    (out/"results.json").write_text(json.dumps(results,indent=2))
    np.savez_compressed(out/"traces.npz", metrics=trace, initial=initial)
    (out/"source.json").write_text(json.dumps([t["digest"] for t in tracks]))
    for form in forms:
        for rate in (.75,1.):
            subset=[r for r in results if r["form"]==form and r["rate"]==rate]
            print(form,rate,sum(r["foot_supported"] for r in subset),sum(r["upright"] for r in subset),
                  round(max(r["peak_qd"] for r in subset),2), flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("out",type=Path)
    run(parser.parse_args().out)
