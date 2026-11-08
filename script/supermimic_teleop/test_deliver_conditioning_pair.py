"""Synthetic consumer contracts; no training, inference or scientific plots."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from deliver_conditioning_pair import ARMS, PLAN, RAW_NAMES, RISKS, SERVER_OUTPUT, load_evidence, validate


def fixture():
    plan = json.loads(PLAN.read_text())
    initial = dict(path='/server/initial.pt', bytes=123, sha256='c'*64)
    traces, arms, proofs = {}, [], []
    for arm in ARMS:
        rows = []
        for update in range(0, 3001, 500):
            risk = 1.+update/30000.
            anchors = 1.+update/100000.
            rows.append(dict(update=update, validation_risk=risk,
                old_matched_validation_risk=risk, pure_validation_risk=risk,
                **{k:anchors for k in RISKS}, fixed_anchor_non_regression=update == 0,
                metrics=None if update == 0 else dict(loss=2.2, point_loss=2., retention_loss=.01, gradient_norm=.5),
                pure_validation=[dict(source_clip_id=k, action_mse=risk) for k in (101,202)],
                schedule_sha256=str(update).zfill(64), source_sample_counts={'old:0':16*update, 'pure:0':16*update},
                elapsed_seconds=update/30.))
        traces[arm+'/training_trace.json'] = rows
        item = dict(variant=arm, initial_adapter_sha256='a'*64, initial_human_adapter_sha256='b'*64,
            initial_validation_risk=1., selected_validation_risk=1., initial_fixed_anchor_risks=[1.]*3,
            selected_fixed_anchor_risks=[1.]*3, schedule_sha256=rows[-1]['schedule_sha256'],
            source_sample_counts=rows[-1]['source_sample_counts'], trainable_parameters=256, trainable_tensors=8,
            actual_updates=3000, actual_adam_steps=[3000]*8, selected_update=0, selected_policy_unchanged=True,
            provenance=dict(initial_student=initial))
        arms.append(item)
        proofs.append(dict(variant=arm, selected_update=0, cpu_finite=True,
            initial_policy_equal=True, current_statistics_exact=True, frozen_teacher_and_decoder_exact=True,
            new_closed_loop_required=False, prior_closed_loop_cases_reused=7, prior_closed_loop_complete_cases_reused=0,
            checkpoints=[dict(name=name, checkpoint_update=step, cpu_finite_tensor_count=12,
                checkpoint=dict(path=SERVER_OUTPUT+'/'+arm+'/'+name, bytes=123, sha256='d'*64),
                robot_adapter_initial_equal=step == 0, human_output_layer_zero=step == 0)
                for name,step in (('selected.pt',0),('last.pt',3000))]))
    cleanup = dict(worker_reaped=True, worker_return_code=0, worker_pid=392130)
    status = 'HUMAN_CONDITIONING_PAIR_TRAINED_PENDING_CLOSED_LOOP'
    campaign = dict(status=status, source_integrity='PASSED', actual_optimizer_updates=6000,
        controller=dict(pid=392056,start_ticks='41964409'),
        arms=[{k:a[k] for k in ('variant','selected_update','selected_policy_unchanged',
            'initial_validation_risk','selected_validation_risk')} for a in arms],
        cost_settlement=dict(state='SETTLED', cleanup=deepcopy(cleanup), run_id=SERVER_OUTPUT,
            stage='r2',charged_seconds=216,over_reserved_budget=False),cumulative_gpu_charged_seconds=5036)
    fit = dict(status=status, actual_optimizer_updates=6000, arms=arms,
        matched_initialization_capacity_and_source_slots=True, human_real_future_frames=0,
        test_sources_read=False, retention_weight=20.,
        initial_action_audit=dict(status='CONDITIONING_INITIAL_ACTION_PARITY_PASSED'))
    guard = dict(status='WORKER_COMPLETED',cleanup=cleanup)
    proof = dict(status='PASSED', result=deepcopy(campaign),guard=deepcopy(guard),arms=proofs,
        initial_checkpoint=initial, task_and_original_training_processes_exited=True,
        gpu_lock_released=True,cuda_initialized=False,
        cost_ledger=dict(charged_seconds=5036,by_stage=dict(r1=1614,r2=2051,r3=1371),outstanding_reservations=0),
        new_model_calls=0,new_optimizer_updates=0,new_physical_steps=0,new_rendered_frames=0)
    auth = dict(gpu_lock_owner=deepcopy(campaign['controller']),maximum_human_future_frames=0,
        original_training_restarted=False,final_acceptance=False)
    return dict(zip(RAW_NAMES[:5],(campaign,fit,guard,proof,auth)),**traces),plan


class ConditioningDeliveryTests(unittest.TestCase):
    def test_complete_consumer_does_not_import_model_or_plot_runtimes(self):
        bundle,plan = fixture()
        before = set(sys.modules)
        runs,selected = validate(bundle,plan)
        self.assertEqual(list(runs),list(ARMS))
        self.assertEqual([r['update'] for r in selected.values()],[0,0])
        self.assertFalse({'torch','matplotlib','mujoco'} & (set(sys.modules)-before))

    def test_selected_zero_does_not_imply_all_nonzero_points_are_ineligible(self):
        bundle,plan = fixture()
        for arm in ARMS:
            for row in bundle[arm+'/training_trace.json'][1:]:
                row.update({k:1. for k in RISKS})
                row['fixed_anchor_non_regression'] = True
        _,selected = validate(bundle,plan)
        self.assertTrue(all(r['update'] == 0 for r in selected.values()))

    def test_changed_selected_candidate_requires_new_closed_loop_evidence(self):
        bundle,plan = fixture()
        arm = ARMS[0]
        row = bundle[arm+'/training_trace.json'][1]
        row.update(validation_risk=.8,old_matched_validation_risk=.8,pure_validation_risk=.8,
            fixed_anchor_non_regression=True,**{k:1. for k in RISKS})
        for value in row['pure_validation']: value['action_mse'] = .8
        item = bundle['fit_result.json']['arms'][0]
        item.update(selected_update=500,selected_validation_risk=.8,selected_policy_unchanged=False)
        bundle['result.json']['arms'][0].update(selected_update=500,selected_validation_risk=.8,selected_policy_unchanged=False)
        proof = bundle['checkpoint_cpu_audit.json']
        proof['result'] = deepcopy(bundle['result.json'])
        cpu = proof['arms'][0]
        cpu.update(selected_update=500,initial_policy_equal=False,new_closed_loop_required=True,
            prior_closed_loop_cases_reused=None,prior_closed_loop_complete_cases_reused=None)
        cpu['checkpoints'][0].update(checkpoint_update=500,robot_adapter_initial_equal=False,human_output_layer_zero=False)
        _,chosen = validate(bundle,plan)
        self.assertEqual(chosen[arm]['update'],500)

    def test_missing_or_duplicate_arm_in_each_record_is_rejected(self):
        for filename in ('result.json','fit_result.json','checkpoint_cpu_audit.json'):
            for change in ('missing','duplicate'):
                bundle,plan = fixture()
                arms = bundle[filename]['arms']
                if change == 'missing': arms.pop()
                else: arms[-1] = deepcopy(arms[0])
                if filename == 'result.json': bundle['checkpoint_cpu_audit.json']['result'] = deepcopy(bundle[filename])
                with self.subTest(filename=filename,change=change), self.assertRaises(ValueError):
                    validate(bundle,plan)

    def test_audit_selection_and_checkpoint_identity_cannot_be_mixed(self):
        for change in ('selection','robot','human','path','reuse'):
            bundle,plan = fixture()
            arm = bundle['checkpoint_cpu_audit.json']['arms'][0]
            if change == 'selection': arm['selected_update'] = 500
            elif change == 'robot': arm['checkpoints'][0]['robot_adapter_initial_equal'] = False
            elif change == 'human': arm['checkpoints'][0]['human_output_layer_zero'] = False
            elif change == 'path': arm['checkpoints'][0]['checkpoint']['path'] = '/other/selected.pt'
            else: arm['prior_closed_loop_complete_cases_reused'] = 7
            with self.subTest(change=change), self.assertRaises(ValueError): validate(bundle,plan)

    def test_cleanup_and_cost_proof_must_match_original_records(self):
        for change in ('cleanup','audit_guard','cost','future'):
            bundle,plan = fixture()
            if change == 'cleanup': bundle['fit.guard.json']['cleanup']['worker_reaped'] = False
            elif change == 'audit_guard': bundle['checkpoint_cpu_audit.json']['guard']['cleanup']['worker_pid'] = 1
            elif change == 'cost': bundle['checkpoint_cpu_audit.json']['cost_ledger']['by_stage']['r2'] = 0
            else: bundle['authorization.json']['maximum_human_future_frames'] = 1
            with self.subTest(change=change), self.assertRaises(ValueError): validate(bundle,plan)

    def test_nonfinite_risk_and_training_metrics_rejected(self):
        for change in ('risk','gradient','loss'):
            bundle,plan = fixture()
            row = bundle[ARMS[0]+'/training_trace.json'][1]
            if change == 'risk': row['pure_validation_risk'] = float('nan')
            elif change == 'gradient': row['metrics']['gradient_norm'] = float('inf')
            else: row['metrics']['loss'] = -1.
            with self.subTest(change=change), self.assertRaises(ValueError): validate(bundle,plan)

    def test_zero_percentage_denominators_are_rejected_before_plotting(self):
        for key in ('validation_risk','pure_validation_risk',*RISKS,'source'):
            bundle,plan = fixture()
            row = bundle[ARMS[0]+'/training_trace.json'][0]
            if key == 'source': row['pure_validation'][0]['action_mse'] = 0.
            else: row[key] = 0.
            with self.subTest(key=key), self.assertRaises(ValueError): validate(bundle,plan)

    def test_all_three_anchor_gates_and_matched_risk_are_recomputed(self):
        for change in ('anchor','weight','objective','relative_tolerance','initial_loss'):
            bundle,plan = fixture()
            row = bundle[ARMS[0]+'/training_trace.json'][1]
            if change == 'anchor': row['fixed_anchor_non_regression'] = True
            elif change == 'weight': row['validation_risk'] += .1
            elif change == 'objective': row['metrics']['retention_loss'] += .1
            elif change == 'relative_tolerance': row['metrics'].update(loss=1000000.000002,point_loss=1000000.,retention_loss=0.)
            else: bundle[ARMS[0]+'/training_trace.json'][0]['metrics'] = dict(row['metrics'])
            with self.subTest(change=change), self.assertRaises(ValueError): validate(bundle,plan)

    def test_per_source_duplicates_missing_rows_and_intermediate_exposure_rejected(self):
        for change in ('duplicate','missing','schedule','count'):
            bundle,plan = fixture()
            row = bundle[ARMS[0]+'/training_trace.json'][1]
            if change == 'duplicate': row['pure_validation'][-1] = deepcopy(row['pure_validation'][0])
            elif change == 'missing': row['pure_validation'].pop()
            elif change == 'schedule': row['schedule_sha256'] = 'f'*64
            else: row['source_sample_counts']['old:0'] += 1
            with self.subTest(change=change), self.assertRaises(ValueError): validate(bundle,plan)

    def test_original_json_bytes_and_published_plan_are_required(self):
        def encoded(value): return (json.dumps(value,indent=2)+'\n').encode()
        def record(path,raw): return dict(path=path,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        bundle,plan = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root/PLAN.name
            plan_raw = encoded(plan)
            plan_path.write_bytes(plan_raw)
            bundle['authorization.json']['conditioning_plan'] = record('/opt/data/private/ZHY/script/supermimic_teleop/'+PLAN.name,plan_raw)
            fit_raw,auth_raw = (encoded(bundle[name]) for name in ('fit_result.json','authorization.json'))
            bundle['result.json']['fit_result'] = record(SERVER_OUTPUT+'/fit_result.json',fit_raw)
            proof = bundle['checkpoint_cpu_audit.json']
            proof['protected_authorization'] = record(SERVER_OUTPUT+'/authorization.json',auth_raw)
            proof['result'] = deepcopy(bundle['result.json'])
            for name,value in bundle.items():
                target = root/name
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(encoded(value))
            load_evidence(root,plan_path)
            (root/'fit_result.json').write_bytes(fit_raw+b' ')
            with self.assertRaises(ValueError): load_evidence(root,plan_path)
            (root/'fit_result.json').write_bytes(fit_raw)
            plan_path.write_bytes(plan_raw+b' ')
            with self.assertRaises(ValueError): load_evidence(root,plan_path)


if __name__ == '__main__':
    unittest.main()
