"""E51 bounded finite-candidate acceptance; E50's separate gate is unchanged."""

import math

UPDATES = 8
STEPS = 32
TRANSITIONS = 3072
CHECKPOINTS = (2, 4, 8)
SCALES = (1., .5, .25, .125, .0625, .03125, .015625, .0078125)
LIMITS = dict(updates=UPDATES, optimizer_proposals=STEPS, committed_steps=STEPS,
              transitions=TRANSITIONS, candidates_per_proposal=len(SCALES),
              candidate_evaluations=STEPS * len(SCALES),
              baseline_evaluations=STEPS, effect_rollouts=10,
              runtime_seconds_per_stage=900)


def accept(before_objective, row):
    values = (before_objective, row['objective'], row['teacher_rmse'], row['teacher_max_abs'])
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite candidate is a computation failure, not a backtrack')
    if row['teacher_rmse'] < 0 or row['teacher_max_abs'] < 0:
        raise ValueError('Negative teacher error')
    return (row['teacher_rmse'] <= .05 and row['teacher_max_abs'] <= .5
            and row['objective'] <= before_objective + 1e-7)


def validate_recipe(new, old):
    """Normalize only known storage paths, never arbitrary keys or numeric values."""
    import copy
    left, right = copy.deepcopy(new), copy.deepcopy(old)
    for tree in (left, right):
        env = tree['manager_env']
        env['commands']['motion']['motion_lib_cfg']['motion_file'] = '<MOTION_DIRECTORY>'
        # This one literal output path is emitted by the launcher.
        if 'trl' in tree['algo']:
            tree['algo']['trl']['output_dir'] = '<STORAGE>'
    if left['manager_env'] != right['manager_env'] or left['algo'] != right['algo']:
        raise ValueError('Environment or loss/optimizer recipe differs beyond declared storage')
