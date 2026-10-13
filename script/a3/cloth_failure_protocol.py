"""Fixed E45 corpus comparison; no tensor imports or permission to launch."""

CHECKPOINTS = (2, 4, 8, 16, 32)
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
        status='NEEDS_INDEPENDENT_VALIDATION' if chosen else 'REJECTED_KNOWN_FAILURE_CORPUS_BRANCH',
        deploy_allowed=False, unseen_generalization_proven=False,
        automatic_more_training_allowed=False)


ADDED_SEQUENCE = 'G_20251229_zzx/zzx_111505_walk'
BASE_TRAINING = (
    'G_0113_zj_train/183926_zj_walk',
    'G_0113_zj_train/192333_zj_warmUp',
    'G_20251226_fjw/fjw_140008_walk',
)
TRAINING = BASE_TRAINING + (ADDED_SEQUENCE,)


def corpus_roles(sequences):
    """New experiment roles never modify the source partition records."""
    accepted = [r for r in sequences if r['pair_quality']['passed']]
    names = {r['sequence'] for r in accepted}
    old_train = {r['sequence'] for r in accepted if r['partition'] == 'train'}
    if len(accepted) != 6 or len(names) != 6 or old_train != set(BASE_TRAINING):
        raise ValueError('Original six references and three training roles must be unchanged')
    if ADDED_SEQUENCE not in names or ADDED_SEQUENCE in old_train:
        raise ValueError('The new example must be the qualified prior development sequence')
    return [dict(sequence=r['sequence'], original_partition=r['partition'],
        experiment_role='DEVELOPMENT_TRAIN' if r['sequence'] in TRAINING else 'DEVELOPMENT_EVAL',
        included_in_development_evaluation=True, unseen_final_test=False) for r in accepted]
