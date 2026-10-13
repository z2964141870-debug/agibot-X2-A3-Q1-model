"""Locate remaining E23 body errors without repeating IK or policy inference."""

import json
from pathlib import Path

import numpy as np

from script.a3.smpl_corpus_qualify import configure_helpers

configure_helpers()

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.mocap_adapter import BODY_MAP


def main():
    root = Path(__file__).resolve().parents[2]
    directory = root / 'data/experiments/a3_zhy_smpl_hull_20261013_E23'
    verification = directory / 'independent_verification.json'
    receipt = json.loads(verification.read_text())
    if receipt['status'] != 'VERIFIED_KINEMATIC_CORPUS':
        raise ValueError('Requires independently verified E23 corpus')
    names, rows = list(BODY_MAP), []
    for row in receipt['sequences']:
        path = directory / 'sequences' / row['sequence'].replace('/', '__') / 'kinematic_trace.npz'
        if sha256(path) != row['artifacts']['kinematic_trace.npz']['sha256']:
            raise ValueError('Saved E23 configuration identity changed')
        with np.load(path, allow_pickle=False) as trace:
            difference = trace['actual_position_m'] - trace['target_position_m']
            error = np.linalg.norm(difference, axis=-1)
            np.testing.assert_allclose(error, trace['position_error_m'], atol=1e-12, rtol=0)
            if error.shape != (len(trace['qpos']), len(names)):
                raise ValueError('Body order or frame coverage differs')
            bodies = []
            for index, name in enumerate(names):
                values = error[:, index]
                frame = int(values.argmax())
                bodies.append(dict(body=name, error_p95_m=float(np.quantile(values, .95)),
                    error_max_m=float(values.max()), frames_above_5cm=int(np.count_nonzero(values > .05)),
                    median_signed_xyz_m=np.median(difference[:, index], axis=0).tolist(),
                    worst_frame=frame, worst_time_s=float(trace['time_s'][frame]),
                    worst_target_xyz_m=trace['target_position_m'][frame, index].tolist(),
                    worst_actual_xyz_m=trace['actual_position_m'][frame, index].tolist()))
            bodies.sort(key=lambda body:body['error_p95_m'], reverse=True)
            rows.append(dict(sequence=row['sequence'], kinematic_qualified=row['pair_quality']['passed'],
                failed_gates=[key for key, passed in row['pair_quality']['checks'].items() if not passed],
                body_frames=len(error)*len(names), body_frames_above_5cm=int(np.count_nonzero(error > .05)),
                bodies=bodies, trace_sha256=sha256(path)))
    result = dict(experiment=receipt['experiment'], diagnosis='SAVED_TRACE_PER_BODY_POSITION_ERROR',
        verified_result_sha256=sha256(verification), code_sha256=sha256(Path(__file__)), sequences=rows,
        ik_repeated=False, optimizer_steps=0, policy_rollouts=0,
        causal_attribution=False, physical_balance_verified=False)
    atomic_json(directory / 'body_error_diagnosis.json', result)
    atomic_json(root / 'data/manifests/a3_zhy_smpl_hull_20261013_E23_body_errors.json', result)
    print(json.dumps([dict(sequence=row['sequence'], failed_gates=row['failed_gates'],
        worst_bodies=row['bodies'][:4]) for row in rows if not row['kinematic_qualified']], indent=2))


if __name__ == '__main__':
    main()
