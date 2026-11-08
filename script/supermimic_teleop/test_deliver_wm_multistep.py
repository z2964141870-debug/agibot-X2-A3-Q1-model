"""Consumer contracts use synthetic numeric fixtures, never scientific evidence."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from deliver_wm_multistep import ARMS, ERRORS, HORIZONS, METHODS, RAW_NAMES, grouped, load_evidence, validate, validation_screen

HERE = Path(__file__).resolve().parent


def fixture():
    plan = json.loads((HERE/'wm_multistep_plan_20261108.json').read_text())
    cases = [dict(job_index=j,arm=arm,clip='clip_'+str(j//2),anchor=64+32*(j%2),
        split='train' if j < 4 else 'validation',valid_prefix_steps=n)
        for j,n in enumerate(plan['common_support_steps']) for arm in ARMS]
    rows = []
    for case in cases:
        for h in HORIZONS:
            if h > case['valid_prefix_steps']:
                continue
            methods = {m:{s:{k:(.5 if m == 'open_loop' else 1.) for k in ERRORS}
                for s in ('endpoint','prefix_rms')} for m in METHODS}
            terms = {m:dict(world=methods[m]['prefix_rms']['body_position_m']**2,
                root=1.,relative=1.,signed_cross=methods[m]['prefix_rms']['body_position_m']**2-2.) for m in METHODS}
            rows.append(dict(job_index=case['job_index'],arm=case['arm'],clip=case['clip'],anchor=case['anchor'],
                split=case['split'],steps=h,common_valid_steps=case['valid_prefix_steps'],methods=methods,
                translation_mse_decomposition=terms,linear_velocity_mse_decomposition=deepcopy(terms),
                last_substep_load_modes=[3]*(h+1),observed_load_mode_changes=0))
    gradients = [dict(job_index=j,split='train' if j == 0 else 'validation',steps=h,
        autograd_directional_derivative=1.,gradient_norm=1.,absent_joint_gradient_exact_zero=True,
        nonzero_response=True,numerical_directional_gradient_passed=True,differentiated_action_index=0,
        later_actions_fixed=True,command_memory_excluded=True,physical_action_gradient_validated=False,
        finite_differences=[dict(epsilon=e,finite_difference=1.,absolute_error=0.,passed=True) for e in (.01,.001)])
        for j in (0,4) for h in HORIZONS]
    result = dict(status='WM_MULTISTEP_COMPLETED',source_integrity='PASSED',per_prefix=rows,gradients=gradients,
        native_state_reconstruction=[dict(job_index=c['job_index'],arm=c['arm'],maximum_link_position_difference_m=0.) for c in cases],
        common_support_steps=plan['common_support_steps'],horizon_cohorts=[dict(steps=h,pairs=sum(n >= h for n in plan['common_support_steps'])) for h in HORIZONS],
        cuda_wm_step_forward_calls=2046,cuda_backward_calls=14,actual_optimizer_updates=0,new_physical_steps=0,
        new_policy_calls=0,test_sources_read=False,terminal_transitions_excluded=True,numerical_gradient_is_not_physical_gradient=True,
        natural_teleoperation_qualified=False,hardware_end_to_end_latency_ms=None,user_accepted_checkpoint=None)
    result['screen'] = validation_screen(rows,gradients,plan)
    cleanup = dict(worker_reaped=True,worker_return_code=0)
    campaign = dict(status=result['status'],source_integrity='PASSED',cost_settlement=dict(state='SETTLED',
        cleanup=cleanup,run_id=plan['output'],stage='r3',charged_seconds=10),cumulative_gpu_charged_seconds=5046,
        cuda_wm_step_forward_calls=2046,cuda_backward_calls=14,actual_optimizer_updates=0,new_physical_steps=0,
        user_accepted_checkpoint=None,diagnostic_screen='PASSED')
    auth = dict(new_optimizer_updates=0,new_physical_steps=0,original_training_restarted=False,final_acceptance=False)
    return dict(zip(RAW_NAMES,(campaign,dict(status='WORKER_COMPLETED',cleanup=cleanup),result,auth))),plan,dict(cases=cases)


class MultistepDeliveryTests(unittest.TestCase):
    def test_complete_consumer_checks_and_command_memory_is_not_screened(self):
        bundle,plan,recovery = fixture()
        bundle['multistep_result.json']['per_prefix'][0]['methods']['open_loop']['prefix_rms']['command_memory_rms'] = 99999.
        self.assertTrue(validate(bundle,plan,recovery)['all_requested_screens_passed'])

    def test_missing_duplicate_and_padded_horizon_rejected(self):
        for change in ('missing','duplicate','padded'):
            bundle,plan,recovery = fixture()
            rows = bundle['multistep_result.json']['per_prefix']
            if change == 'missing': rows.pop()
            elif change == 'duplicate': rows[-1] = deepcopy(rows[0])
            else: rows[-1]['steps'] = 40
            with self.assertRaises(ValueError): validate(bundle,plan,recovery)

    def test_signed_negative_cross_retained_and_broken_closure_rejected(self):
        bundle,plan,recovery = fixture()
        validate(bundle,plan,recovery)
        terms = bundle['multistep_result.json']['per_prefix'][0]['translation_mse_decomposition']['open_loop']
        self.assertLess(terms['signed_cross'],0)
        terms['signed_cross'] = abs(terms['signed_cross'])
        with self.assertRaises(ValueError): validate(bundle,plan,recovery)

    def test_equal_clip_weighting_is_not_pooled_anchor_weighting(self):
        rows = [dict(clip=clip,methods=dict(open_loop=dict(prefix_rms=dict(metric=value))))
            for clip,value in (('a',.2),('a',.2),('a',.2),('b',.8))]
        self.assertAlmostEqual(grouped(rows,'open_loop','metric'),.5)

    def test_failed_scientific_screen_is_valid_but_tampered_flag_rejected(self):
        bundle,plan,recovery = fixture()
        result = bundle['multistep_result.json']
        result['gradients'][0].update(nonzero_response=False,numerical_directional_gradient_passed=False)
        result['screen'] = validation_screen(result['per_prefix'],result['gradients'],plan)
        bundle['result.json']['diagnostic_screen'] = 'FAILED'
        self.assertFalse(validate(bundle,plan,recovery)['all_requested_screens_passed'])
        result['screen']['all_requested_screens_passed'] = True
        with self.assertRaises(ValueError): validate(bundle,plan,recovery)

    def test_missing_gradient_or_wrong_finite_difference_flag_rejected(self):
        for change in ('missing','flag'):
            bundle,plan,recovery = fixture()
            gradients = bundle['multistep_result.json']['gradients']
            if change == 'missing': gradients.pop()
            else: gradients[0]['finite_differences'][0]['passed'] = False
            with self.assertRaises(ValueError): validate(bundle,plan,recovery)

    def test_cleanup_cost_support_and_nonfinite_gates(self):
        for change in ('cleanup','cost','support','finite','mode'):
            bundle,plan,recovery = fixture()
            if change == 'cleanup': bundle['multistep.guard.json']['cleanup']['worker_reaped'] = False
            elif change == 'cost': bundle['result.json']['cumulative_gpu_charged_seconds'] = 10
            elif change == 'support': recovery['cases'][0]['valid_prefix_steps'] = 79
            elif change == 'mode': bundle['multistep_result.json']['per_prefix'][0]['last_substep_load_modes'][0] = 4
            else: bundle['multistep_result.json']['per_prefix'][0]['methods']['open_loop']['endpoint']['joint_position_rad'] = float('nan')
            with self.assertRaises(ValueError): validate(bundle,plan,recovery)

    def test_original_bytes_and_published_identity_chain_required(self):
        def encode(value): return (json.dumps(value,indent=2)+'\n').encode()
        def record(path,raw): return dict(path=path,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        bundle,plan,recovery = fixture()
        recovery_raw = encode(recovery)
        plan['required_recovery_result'] = record(plan['required_recovery_result']['path'],recovery_raw)
        plan_raw = encode(plan)
        filenames = ('wm_multistep_probe.py','wm_multistep_worker.py','wm_multistep_math.py','wm_multistep_plan_20261108.json','test_wm_multistep.py')
        records = [record('/server/'+name,plan_raw if name.endswith('.json') else b'fixture') for name in filenames]
        manifest = dict(schema='q1_wm_multistep_code_v1',evaluation_commit='a'*40,files=records)
        manifest_raw = encode(manifest)
        auth = bundle['authorization.json']
        auth.update(evaluation_commit='a'*40,protected_inputs=records+[record('/server/manifest.json',manifest_raw),
            plan['required_recovery_result'],plan['required_precision_result']])
        bundle['result.json']['multistep_result'] = record(plan['output']+'/multistep_result.json',encode(bundle['multistep_result.json']))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name,value in bundle.items(): (root/name).write_bytes(encode(value))
            plan_path,manifest_path,recovery_path = root/filenames[3],root/'manifest.json',root/'recovery.json'
            for path,raw in ((plan_path,plan_raw),(manifest_path,manifest_raw),(recovery_path,recovery_raw)): path.write_bytes(raw)
            load_evidence(root,plan_path,manifest_path,recovery_path)
            (root/'multistep_result.json').write_bytes(encode(bundle['multistep_result.json'])+b' ')
            with self.assertRaises(ValueError): load_evidence(root,plan_path,manifest_path,recovery_path)


if __name__ == '__main__':
    unittest.main()
