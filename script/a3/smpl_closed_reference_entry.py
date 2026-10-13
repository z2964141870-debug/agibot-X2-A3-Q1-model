"""Frozen simulator with prevalidated passive-reference assembly."""

from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np


def main():
    from script.a3 import sim_entry, smpl_frozen_entry
    from script.a3.checkpoint_store import atomic_json, sha256
    root = Path(os.environ['YUANQI_E33_OUTPUT'])
    contract = json.loads((root / 'assembly_contract.json').read_text())
    if contract['status'] != 'VERIFIED_PASSIVE_REFERENCE_ASSEMBLY':
        raise ValueError('Requires the accepted E33 assembly contract')
    original_loader = sim_entry.load_sim

    def load():
        sim = original_loader()
        original_reference = sim.load_loop_motion_reference

        def reference(model, runtime, solver, motion_path, **kwargs):
            value = original_reference(model, runtime, solver, motion_path, **kwargs)
            stem = Path(motion_path).stem
            expected = next(r for r in contract['motions'] if r['stem'] == stem)
            path = root / 'assembled' / (stem + '.npz')
            if sha256(path) != expected['artifact']['sha256']:
                raise ValueError('Validated passive-reference artifact changed')
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(value.qpos, saved['original_qpos'])
                qpos, qvel = saved['qpos'].copy(), saved['qvel'].copy()
            atomic_json(root / 'entry' / (stem + '.json'), dict(reference_frames=len(qpos),
                passive_reference_loaded=True, source_arrays_equal=True,
                endpoint_max_m=expected['endpoint_after_max_m']))
            return replace(value, qpos=qpos, qvel=qvel)

        sim.load_loop_motion_reference = reference
        return sim

    sim_entry.load_sim = load
    return smpl_frozen_entry.main()


if __name__ == '__main__':
    raise SystemExit(main())
