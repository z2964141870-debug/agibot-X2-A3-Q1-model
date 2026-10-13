"""Fixed E44 exposure comparison; no tensor imports or permission to launch."""

CHECKPOINTS = (8, 16, 32)
UPDATES = CHECKPOINTS[-1]
OPTIMIZER_STEPS = UPDATES * 4
TRANSITIONS = UPDATES * 16 * 24


def outcome(effects):
    """Require every whole checkpoint, never combine per-motion winners."""
    if set(effects) != set(CHECKPOINTS):
        raise ValueError('The complete predeclared checkpoint curve is required')
    candidates, rows = [], []
    expected = None
    for step in CHECKPOINTS:
        effect = effects[step]
        motions = effect['rows']
        identities = {(r['kind'], r['motion']) for r in motions}
        if (len(motions) != 10 or len(identities) != 10 or
                sum(r['kind'] == 'cloth' for r in motions) != 6 or
                sum(r['kind'] == 'official' for r in motions) != 4):
            raise ValueError('Every checkpoint needs six clothing and four official results')
        if expected is not None and identities != expected:
            raise ValueError('Checkpoint motion sets differ')
        expected = identities
        old, new = effect['baseline_clothing_mean_rad'], effect['candidate_clothing_mean_rad']
        if not (0 < old < float('inf') and 0 < new < float('inf')):
            raise ValueError('Invalid effect summary')
        # Recheck the original E42 effect gates, rather than trust a status flag.
        passed = (new <= old * .99 and all(
            not r['candidate_fall'] and
            (r['baseline_fall'] or r['ratio'] <= 1.02) and
            set(r['world_error_ratios']) == {'root_pos_error_m_mean', 'wrists_world_mean_m'} and
            all(0 <= v <= 1.05 for v in r['world_error_ratios'].values())
            for r in motions))
        if passed != effect['development_gate_passed']:
            raise ValueError('Effect status contradicts the original acceptance gates')
        rows.append(dict(step=step, development_gate_passed=passed,
            baseline_clothing_mean_rad=old, candidate_clothing_mean_rad=new,
            clothing_rmse_ratio=new / old,
            clothing_nonfalls=sum(not r['candidate_fall'] for r in motions if r['kind'] == 'cloth')))
        if passed:
            candidates.append((new, step))
    chosen = min(candidates)[1] if candidates else None
    return dict(checkpoints=rows, selected_development_step=chosen,
        status='NEEDS_INDEPENDENT_VALIDATION' if chosen else 'REJECTED_FIXED_EXPOSURE_BRANCH',
        deploy_allowed=False, unseen_generalization_proven=False,
        automatic_more_training_allowed=False)
