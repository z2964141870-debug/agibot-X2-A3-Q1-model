"""Single-use independent specialist implementation/source/input identity freeze."""

import json
from pathlib import Path
import subprocess

from script.a3 import humanplus_specialist_trial as trial
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.humanplus_specialist_protocol import UPDATES, INTERACTIONS, OPTIMIZER_STEPS, OUTER_CALLS


def identity(path):
    return dict(bytes=path.stat().st_size, sha256=sha256(path))


def main():
    trial.runtime.require_root()
    trial.base.clean_vendor()
    if trial.DESIGN.exists() or any(trial.ROOT.glob('data/manifests/*HP20*')):
        raise FileExistsError('D04/HP20 freeze must not replace an existing ledger')
    prior_path = trial.ROOT / 'data/experiments/a3_zhy_cloth_failure_20261013_E45/prepare.json'
    motions = trial.read(prior_path)['motions']
    training = next(row for row in motions if row['sequence'] == 'G_20251229_zzx/zzx_111505_walk')
    if training['frames'] != 300 or training['fps'] != 30:
        raise ValueError('Complete unchanged zzx training reference required')
    path = trial.ROOT / training['path']
    if identity(path) != {key: training[key] for key in ('bytes', 'sha256')}:
        raise ValueError('Training reference changed')
    baseline = trial.read(trial.CONTRACT)
    row = next(r for r in baseline['motions'] if r['sequence'] == training['sequence'])
    baseline_plan = trial.read(trial.ROOT / 'data/manifests/a3_humanplus_baseline_reference_20261018_HP11_plan.json')
    source_row = next(r for r in baseline_plan['motions'] if r['sequence'] == training['sequence'])
    row = dict(row, path=source_row['path'])
    if row['full_policy_steps'] != 499:
        raise ValueError('Expected fixed 499-step matched motion')
    recipe = 'data/training/a3_humanplus_route_20261018_HP19/R25_001_s0/config.yaml'
    paths = [prior_path, path, trial.CONTRACT, trial.OFFICIAL, trial.OFFICIAL.with_name('config.yaml'),
             trial.ROOT / recipe, trial.ROOT / row['path']]
    assets_path = trial.ROOT / 'data/manifests/a3_humanplus_policy_assets_20261018.json'
    for asset in trial.read(assets_path)['assets']:
        p = trial.VENDOR / asset['path']
        if identity(p) != {k: asset[k] for k in ('bytes', 'sha256')}:
            raise ValueError('Required static robot asset changed')
        paths.append(p)
    paths.append(assets_path)
    if identity(trial.OFFICIAL) != dict(bytes=402050931,
            sha256='9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d'):
        raise ValueError('Fresh035 missing or changed')
    files = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard',
        'script/a3', 'script/project_tools'], text=True).splitlines()
    code = {}
    for name in sorted(set(files)):
        if name.endswith('.py'):
            p = trial.ROOT / name
            compile(p.read_text(), name, 'exec')
            code[name] = identity(p)
    # Read only preserved environment identities, not retired artifact chains.
    source = trial.read(trial.ROOT / 'data/manifests/a3_humanplus_route_20261018_HP19_plan.json')
    environment = {}
    for name, expected in source['environment_source_sha256'].items():
        p = Path(name)
        if sha256(p) != expected:
            raise ValueError('Shared environment source changed')
        environment[name] = identity(p)
    lab_env = Path('/home/humanplus/x2_teleop_final/third_party/IsaacLab-v2.3.2/source/isaaclab/isaaclab/envs/manager_based_rl_env.py')
    environment[str(lab_env)] = identity(lab_env)
    inputs = {str(p.relative_to(trial.ROOT)): identity(p) for p in sorted(set(paths))}
    record = dict(status='IMPLEMENTED_CONTROLLED_READY_FOR_REGISTRATION',
        at_utc=trial.runtime.stamp(), training_motion=training, baseline_motion=row,
        recipe_source=recipe, code_identity=code, input_identity=inputs, environment_identity=environment,
        hypothesis='BOUNDED_SINGLE_MOTION_FULL_DECODER_SPECIALIST_THEN_CONDITIONAL_UNIFIED_DISTILLATION',
        causal_single_variable_claim=False,
        regime_changes=['ZZX_ONLY_COMPLETE_TRAINING_MOTION', 'FULL_ORIGINAL_DECODER_TRAINABLE',
                        'NATIVE_PPO_WITHOUT_UNIFIED_TEACHER_RETENTION_OR_FINITE_CANDIDATES',
                        'PREDECLARED_LONGER_LEARNING_BUDGET'],
        fixed_recipe=['FRESH035_ACTOR_CRITIC_EMPTY_ADAM', 'FROZEN_CODEC_AND_STD',
                      'NATIVE_PAIRED_A3_FAST_AND_ACTUAL_SAMPLED_TOKENS',
                      'ORIGINAL_REWARDS_PHYSICS_RESET_ADAPTIVE_SAMPLING_KL_AND_FP32'],
        updates=UPDATES, interactions=INTERACTIONS, optimizer_steps=OPTIMIZER_STEPS,
        training_outer_actor_decoder_calls=OUTER_CALLS, convergence_claim=False,
        saved_steps=[0, UPDATES], complete_isaac_baseline_and_final_steps=499,
        complete_final_mujoco_steps=499, hp11_mujoco_baseline_replay_forbidden=True,
        specialist_gate='BOTH_SIMULATORS_FULL_NO_FALL_AND_JOINT_RMSE_IMPROVEMENT_AT_LEAST_1_PERCENT;MUJOCO_ROOT_WRIST_AT_MOST_PLUS_5_PERCENT',
        specialist_success_does_not_pass_unified_gate=True, student_visited_corrections_require_new_reliability_contract=True,
        independent_root_translation_input_limitation_unchanged=True,
        motion_role='DEVELOPMENT_TRAIN_NOT_UNSEEN', deploy_allowed=False,
        supersedes_unregistered_v1_reason='ISAAC_BASE_EVAL_HAS_NO_HEADLESS_SEED_NUM_ENVS_KEYS_FORCE_ADD_BEFORE_MERGE',
        controls=dict(mac_passed=6, mac_vendor_source_skipped=1, server_passed=7,
                      no_tensor_or_model_or_simulation_controls=True),
        activity=dict(tensor_loads=0, policy_forwards=0, simulation_steps=0, optimizer_steps=0),
        references=dict(beyondmimic=dict(arxiv='2508.08241v4', first_published='2025-08-11',
            updated='2025-11-13', verification='PRIMARY_HTML_METHOD_TEXT_AND_ATOM_METADATA_NOT_REPRODUCED'),
            dagger=dict(pmlr='v15/ross11a', published='2011-06-14', verification='PRIMARY_ABSTRACT_PAGE_NOT_REPRODUCED')),
        github_synced=False, backup_status='LOCAL_ONLY')
    atomic_json(trial.DESIGN, record)
    print(json.dumps(dict(status=record['status'], code_files=len(code), input_files=len(inputs),
                         environment_files=len(environment), updates=UPDATES, interactions=INTERACTIONS)))


if __name__ == '__main__':
    main()
