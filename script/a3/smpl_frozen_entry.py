"""E24 frozen simulator entry with passive contact and full-state capture."""

from pathlib import Path
import sys

import numpy as np


def foot_geometry(model, mujoco):
    feet = []
    for side in ('left', 'right'):
        selected = []
        for geom in range(model.ngeom):
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom]))
            if body and body.startswith(side + '_') and ('ankle_roll' in body or 'foot_' in body):
                kind = model.geom_type[geom]
                if kind == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh = model.geom_dataid[geom]
                    start = model.mesh_vertadr[mesh]
                    vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]].copy()
                elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                    from itertools import product
                    vertices = np.asarray(list(product((-1, 1), repeat=3))) * model.geom_size[geom]
                else:
                    raise ValueError('Unsupported actual foot geometry')
                selected.append((geom, vertices))
        if not selected:
            raise ValueError('Missing actual foot geometry')
        feet.append(selected)
    return feet


def foot_minima(data, feet):
    return [min(float((vertices @ data.geom_xmat[geom].reshape(3, 3).T +
                       data.geom_xpos[geom])[:, 2].min()) for geom, vertices in selected)
            for selected in feet]


def main():
    from script.a3 import sim_entry
    original_loader = sim_entry.load_sim
    records = []

    def load():
        import torch
        torch.manual_seed(0)
        np.random.seed(0)
        sim = original_loader()
        original_record = sim.RolloutMetricsCollector.record
        geometry = {}

        def record(collector, **kwargs):
            original_record(collector, **kwargs)
            if collector.current is None:
                return
            model, data = collector.model, kwargs['data']
            key = id(model)
            if key not in geometry:
                feet = foot_geometry(model, sim.mujoco)
                floor = set(np.flatnonzero(model.geom_type == sim.mujoco.mjtGeom.mjGEOM_PLANE).tolist())
                if not floor:
                    raise ValueError('No floor plane for contact capture')
                geometry[key] = feet, floor
            feet, floor = geometry[key]
            counts, forces = np.zeros(2, dtype=np.int64), np.zeros(2)
            for index in range(data.ncon):
                contact = data.contact[index]
                pair = {int(contact.geom1), int(contact.geom2)}
                if not pair.intersection(floor):
                    continue
                for side, selected in enumerate(feet):
                    if any(geom in pair for geom, _ in selected):
                        force = np.zeros(6)
                        sim.mujoco.mj_contactForce(model, data, index, force)
                        counts[side] += 1
                        forces[side] += abs(float(force[0]))
            records.append((kwargs['policy_tick'], data.qpos.copy(), data.qvel.copy(),
                            counts, forces, foot_minima(data, feet)))

        sim.RolloutMetricsCollector.record = record
        return sim

    sim_entry.load_sim = load
    try:
        return sim_entry.main()
    finally:
        if records:
            metric = Path(sys.argv[sys.argv.index('--metrics-out') + 1])
            path = metric.with_name(metric.name.replace('.metrics.json', '.contact.npz'))
            np.savez_compressed(path, policy_tick=np.asarray([r[0] for r in records]),
                qpos=np.asarray([r[1] for r in records]), qvel=np.asarray([r[2] for r in records]),
                foot_floor_contact_count=np.asarray([r[3] for r in records]),
                foot_floor_normal_force_n=np.asarray([r[4] for r in records]),
                foot_geometry_min_z_m=np.asarray([r[5] for r in records]),
                scope=np.asarray('PASSIVE_50HZ_SIM_CONTACT_SAMPLES_NOT_HUMAN_CONTACT_GROUND_TRUTH'))


if __name__ == '__main__':
    raise SystemExit(main())
