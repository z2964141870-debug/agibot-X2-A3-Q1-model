"""Complete passive ball-joint rod orientations without changing commands."""

import numpy as np
from scipy.spatial.transform import Rotation


def rod_contract(model, mujoco):
    rows = []
    for eq in range(model.neq):
        if model.eq_type[eq] != mujoco.mjtEq.mjEQ_CONNECT:
            raise ValueError('Only internal rod connects are supported')
        body, target = int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq])
        joints = np.flatnonzero(model.jnt_bodyid == body)
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if (len(joints) != 1 or 'rod_' not in name or target == 0
                or model.jnt_type[joints[0]] != mujoco.mjtJoint.mjJNT_BALL
                or np.linalg.norm(model.jnt_pos[joints[0]]) > 1e-12):
            raise ValueError('Expected a zero-offset passive rod ball joint')
        rows.append((eq, body, target, int(model.jnt_qposadr[joints[0]])))
    if not rows:
        raise ValueError('No internal rod constraints')
    return rows


def endpoint_errors(model, data, rows):
    return np.asarray([np.linalg.norm(data.xpos[b] + data.xmat[b].reshape(3, 3) @ model.eq_data[e, :3]
        - data.xpos[t] - data.xmat[t].reshape(3, 3) @ model.eq_data[e, 3:6]) for e, b, t, _ in rows])


def close_reference(model, mujoco, qpos, dt):
    rows, data = rod_contract(model, mujoco), mujoco.MjData(model)
    corrected = np.asarray(qpos).copy()
    before, after = [], []
    mutable = np.zeros(model.nq, dtype=bool)
    for _, _, _, address in rows:
        mutable[address:address + 4] = True
    previous = None
    for index, original in enumerate(qpos):
        data.qpos[:] = original
        mujoco.mj_fwdPosition(model, data)
        before.append(endpoint_errors(model, data, rows))
        if previous is not None:
            data.qpos[mutable] = previous[mutable]
        # Single-vector alignment preserves the previous unconstrained twist.
        for eq, body, target, address in rows:
            mujoco.mj_fwdPosition(model, data)
            old_rotation = Rotation.from_matrix(data.xmat[body].reshape(3, 3))
            current = old_rotation.apply(model.eq_data[eq, :3])
            desired = data.xpos[target] + data.xmat[target].reshape(3, 3) @ model.eq_data[eq, 3:6] - data.xpos[body]
            if min(np.linalg.norm(current), np.linalg.norm(desired)) < 1e-9:
                raise ValueError('Degenerate rod endpoint')
            delta, _ = Rotation.align_vectors([desired / np.linalg.norm(desired)],
                                             [current / np.linalg.norm(current)])
            base = Rotation.from_matrix(data.xmat[model.body_parentid[body]].reshape(3, 3)) * (
                Rotation.from_quat(model.body_quat[body], scalar_first=True))
            quaternion = (base.inv() * delta * old_rotation).as_quat(scalar_first=True)
            if np.dot(quaternion, data.qpos[address:address + 4]) < 0:
                quaternion = -quaternion
            data.qpos[address:address + 4] = quaternion
        mujoco.mj_fwdPosition(model, data)
        after.append(endpoint_errors(model, data, rows))
        corrected[index] = data.qpos
        previous = data.qpos.copy()
    np.testing.assert_array_equal(corrected[:, ~mutable], qpos[:, ~mutable])
    velocity = np.zeros((len(qpos), model.nv))
    for frame in range(len(qpos) - 1):
        mujoco.mj_differentiatePos(model, velocity[frame], dt, corrected[frame], corrected[frame + 1])
    velocity[-1] = velocity[-2]
    return corrected, velocity, dict(before_endpoint_error_m=np.asarray(before),
        after_endpoint_error_m=np.asarray(after), mutable_qpos_mask=mutable,
        ball_joint_count=len(rows), twist_scope='MINIMUM_ROTATION_CONTINUOUS_GAUGE_NOT_OBSERVED_ROD_TWIST')


def controls(mujoco):
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="rod_test">'
        '<joint type="ball"/><geom type="capsule" size=".01 .1" pos="0 0 .1" mass="1"/>'
        '</body><body name="target" pos="0 0 .2"><geom type="sphere" size=".01" mass="1"/>'
        '</body></worldbody><equality><connect body1="rod_test" body2="target" anchor="0 0 .2"/>'
        '</equality></mujoco>')
    model.eq_data[0, 3:6] = [.2, 0., -.2]
    q = np.tile(model.qpos0, (3, 1))
    corrected, velocity, evidence = close_reference(model, mujoco, q, .02)
    np.testing.assert_allclose(evidence['before_endpoint_error_m'], np.sqrt(.08), atol=1e-12)
    np.testing.assert_allclose(evidence['after_endpoint_error_m'], 0., atol=1e-12)
    np.testing.assert_allclose(velocity, 0., atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(corrected, axis=1), 1., atol=1e-12)
    return dict(endpoint_rotation=True, unchanged_static_velocity=True,
                unit_quaternion=True, passed=True)
