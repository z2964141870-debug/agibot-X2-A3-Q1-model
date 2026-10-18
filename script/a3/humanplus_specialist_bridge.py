"""CPU name/shape contract for a future expert replay; HP20 stays unchanged."""


def _unique(names, role):
    names = tuple(names)
    if not names or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
        raise ValueError('Missing or ambiguous ' + role)
    return names


def _indices(source, selected, role):
    positions = {name: index for index, name in enumerate(source)}
    if any(name not in positions for name in selected):
        raise ValueError('Missing ' + role + ' correspondence')
    return tuple(positions[name] for name in selected)


def correspondence(*, selected_body_names, reference_body_count, robot_body_names,
                   reference_joint_names, reference_joint_count, robot_joint_names,
                   action_joint_names):
    """Validate metadata before GPU indexing, keeping each source's own ordering.

    Native command.body_pos_w is already in selected_body_names order. It must
    never be indexed a second time with full-robot body IDs. Joint references
    require explicit source names because a complete robot can exceed 29 DOFs.
    """
    bodies = _unique(selected_body_names, 'selected body names')
    robot_bodies = _unique(robot_body_names, 'robot body names')
    reference_joints = _unique(reference_joint_names, 'reference joint names')
    robot_joints = _unique(robot_joint_names, 'robot joint names')
    actions = _unique(action_joint_names, 'action joint names')
    if type(reference_body_count) is not int or reference_body_count != len(bodies):
        raise ValueError('Native selected-reference body shape differs from names')
    if type(reference_joint_count) is not int or reference_joint_count != len(reference_joints):
        raise ValueError('Reference joint shape differs from explicit source names')
    if len(actions) != 29:
        raise ValueError('A3 policy requires exactly 29 named action joints')
    return dict(reference_body_indices=tuple(range(reference_body_count)),
                robot_body_indices=_indices(robot_bodies, bodies, 'body'),
                reference_joint_indices=_indices(reference_joints, actions, 'reference action-joint'),
                robot_joint_indices=_indices(robot_joints, actions, 'robot action-joint'),
                body_names=bodies, joint_names=actions)
