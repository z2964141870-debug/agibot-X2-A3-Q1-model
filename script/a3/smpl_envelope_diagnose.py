"""Measure conservative envelope clearance from saved E22 configurations."""

import json
from pathlib import Path

import numpy as np

from script.a3.smpl_corpus_qualify import configure_helpers

configure_helpers()

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.smpl_vertex_floor import EnvelopeA3Retargeter, FootEnvelopeLimit


def main():
    root = Path(__file__).resolve().parents[2]
    directory = root / 'data/experiments/a3_zhy_smpl_envelope_20261012_E22'
    verification = directory / 'independent_verification.json'
    receipt = json.loads(verification.read_text())
    if receipt['status'] != 'VERIFIED_QUALITY_REJECTION':
        raise ValueError('Requires the independently verified E22 result')
    retargeter = EnvelopeA3Retargeter()
    limit = next(value for value in retargeter.limits if isinstance(value, FootEnvelopeLimit))
    corners = dict(limit.envelopes)
    rows = []
    for row in receipt['sequences']:
        path = directory / 'sequences' / row['sequence'].replace('/', '__') / 'kinematic_trace.npz'
        if sha256(path) != row['artifacts']['kinematic_trace.npz']['sha256']:
            raise ValueError('Saved E22 configuration identity changed')
        with np.load(path, allow_pickle=False) as trace:
            actual, envelope = [], []
            for qpos in trace['qpos']:
                retargeter.configuration.update(qpos)
                data = retargeter.configuration.data
                actual.append(retargeter.foot_min_z())
                envelope.append([min(float((corners[geom]
                    @ data.geom_xmat[geom].reshape(3, 3).T
                    + data.geom_xpos[geom])[:, 2].min())
                    for geom, _ in selected) for selected in retargeter.foot_geometries])
            actual, envelope = np.asarray(actual), np.asarray(envelope)
            np.testing.assert_allclose(actual, trace['foot_geometry_min_z_m'], atol=1e-12, rtol=0)
            if np.any(actual - envelope < -1e-10):
                raise ValueError('Envelope failed to contain the actual foot')
            contact = trace['inferred_contact'].astype(bool)
            feet = []
            for side in range(2):
                mask = contact[:, side]
                if not mask.any():
                    raise ValueError('No contact frames for clearance diagnosis')
                feet.append(dict(contact_frames=int(mask.sum()),
                    actual_height_p50_p95_m=np.quantile(actual[mask, side], [.5, .95]).tolist(),
                    envelope_height_p50_p95_m=np.quantile(envelope[mask, side], [.5, .95]).tolist(),
                    conservative_gap_p50_p95_m=np.quantile((actual-envelope)[mask, side], [.5, .95]).tolist()))
        rows.append(dict(sequence=row['sequence'],feet=feet,input_sha256=sha256(path)))
    result = dict(experiment=receipt['experiment'], diagnosis='SAVED_CONFIGURATION_ENVELOPE_CLEARANCE',
        verified_result_sha256=sha256(verification),code_sha256=sha256(Path(__file__)),
        foot_geometries=len(limit.envelopes),plane_rows=sum(len(v) for _, v in limit.envelopes),
        sequences=rows,ik_repeated=False,optimizer_steps=0,policy_rollouts=0,
        measured_human_contact=False,physical_balance_verified=False)
    atomic_json(directory / 'envelope_geometry_diagnosis.json', result)
    atomic_json(root / 'data/manifests/a3_zhy_smpl_envelope_20261012_E22_geometry.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
