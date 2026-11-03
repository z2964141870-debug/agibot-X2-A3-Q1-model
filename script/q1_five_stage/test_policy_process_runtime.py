"""Real two-runtime CPU transport; synthetic native weights, no integration."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROBE = r'''
import importlib.metadata,json,pathlib,sys
from unittest.mock import patch
runtime,shared,plan,checkpoint,output=map(pathlib.Path,sys.argv[1:])
runtime,shared=json.loads(runtime.read_text()),json.loads(shared.read_text())
root=pathlib.Path('/opt/data/private/ZHY')
sys.path[:0]=[str(root/'script/q1_five_stage'),str(root/'script/primebot_q1'),
 str(root/'script/primebot_q1_review'),str(pathlib.Path(runtime['adapted_source'])/'src')]
import mujoco,torch
torch.set_num_threads(1)
from evaluate import load_actor
from policy_process import PolicyProcess
from supermimic.data import Q1Motions
from supermimic.simulator import Q1Simulator
from common import FIXED
audit=json.loads((pathlib.Path(shared['primebot_review'])/'checkpoint_audit.json').read_text())
baseline=audit['final']['checkpoint']['path']
robot,unused,config,_,_,_=load_actor(baseline)
del unused
command=[runtime['morph_python'],'-u',str(root/'script/q1_five_stage/policy_process.py'),
 '--plan',str(plan),'--checkpoint',str(checkpoint),'--frozen',shared['frozen_data'],
 '--primebot-checkpoint',baseline,'--morph-source',runtime['morph_source'],
 '--primebot-source',runtime['adapted_source']]
worker=PolicyProcess(command,output.with_suffix('.worker.log'))
record=dict(plant_python=sys.executable,plant_packages={k:importlib.metadata.version(k) for k in ('torch','mujoco')},
 policy_runtime=worker.ready,cases=[],new_physical_transitions=0,new_optimizer_updates=0)
try:
 with patch.object(mujoco,'mj_step',side_effect=AssertionError('no integration')),\
      patch.object(mujoco,'mj_step1',side_effect=AssertionError('no integration')),\
      patch.object(mujoco,'mj_step2',side_effect=AssertionError('no integration')),\
      patch.object(Q1Simulator,'step',side_effect=AssertionError('no rollout')):
  simulator=Q1Simulator(robot,1,'cpu',config['action_scale'],use_tn=config.get('use_tn',False),
   physics_timestep=config.get('physics_timestep'))
  rows=json.loads(pathlib.Path(shared['frozen_data']).read_text())['rows']
  for row in rows:
   name=pathlib.Path(row['q1_primebot']['path']).stem
   if name not in FIXED: continue
   reference=Q1Motions(robot,[pathlib.Path(row['q1_primebot']['path'])],device='cpu').load(0)
   for interface in ('motion','human'):
    worker.start(row,0,interface)
    simulator.reset(torch.tensor([0]),reference[:1],randomize=False)
    case=dict(clip=name,interface=interface,frames=[])
    for frame in range(3):
     state=simulator.state()[0] if frame==0 else reference[frame]
     case['frames'].append(dict(frame=frame,state=state.tolist(),**worker.action(frame,state)))
    record['cases'].append(case)
 worker.close()
 record.update(worker_reaped=True,worker_return_code=worker.child.returncode,cuda_initialized=torch.cuda.is_initialized())
 output.write_text(json.dumps(record,allow_nan=False))
finally:
 if not worker.closed: worker.close(graceful=False)
'''


class RealPolicyProcessTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("Q1_COMMON_TEST_RUNTIME"), "Requires both fixed server environments and assets")
    def test_locked_plant_and_native_policy_match_direct_cpu_math(self):
        import numpy as np
        import torch
        runtime_path = Path(os.environ["Q1_COMMON_TEST_RUNTIME"])
        shared_path = Path(os.environ["Q1_COMMON_TEST_SHARED_CONFIG"])
        runtime, shared = json.loads(runtime_path.read_text()), json.loads(shared_path.read_text())
        root = Path("/opt/data/private/ZHY")
        sys.path[:0] = [str(Path(runtime["morph_source"]) / "src"),
                       str(Path(runtime["adapted_source"]) / "src"), str(root / "script/morphmimic_q1"),
                       str(root / "script/primebot_q1_review")]
        from bootstrap import install
        spec = install("q1")["robot"]
        from morphmimic.constants import ANCHOR_IDX, GOAL_DIM, HUMAN_FEAT_DIM, HUMAN_HEADING_DIM, PROPRIO_DIM
        from morphmimic.data.motion_library import MotionLibrary
        from morphmimic.model.human import pack_human_goal
        from morphmimic.model.policy import Policy, PolicyCfg
        from morphmimic.model.rewards_torch import analytic_obs, goal_features
        from morphmimic.model.state import split_state
        from morphmimic.train.normalizers import RunningNormalizer
        from morphmimic.train.obs_history import ObsHistory
        from evaluate import load_actor
        from inference import load_policy
        from q1_native_review import native_state, reference_window_indices
        from test_inference import checkpoint
        torch.set_num_threads(1)
        self.assertFalse(torch.cuda.is_initialized())
        value = checkpoint(0)
        value["cfg"].update(policy_hidden=[32], act="elu", latent=32, decoder_film=True)
        plan = value["native_q1_contract"]["plan"]
        plan.update(source=runtime["morph_source"], human_root=shared["human_root"])
        from morphmimic.robots.morphology import morphology_vector
        morphology = morphology_vector("q1", actor=True)
        policy = Policy(PolicyCfg(enc_hidden=(32,), dec_hidden=(32,), morph_dim=154, film=True, human=True))
        policy.set_morphology(morphology)
        value.update(policy=policy.state_dict(), morph=dict(actor=morphology, film=True), imu_body=spec.imu_body)
        for name, width in (("obs_norm", PROPRIO_DIM), ("goal_norm", GOAL_DIM),
                            ("human_norm", HUMAN_FEAT_DIM), ("heading_norm", HUMAN_HEADING_DIM)):
            value[name] = RunningNormalizer(width).state_dict()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path, plan_path, output, probe = (folder / name for name in ("synthetic.pt", "plan.json", "result.json", "probe.py"))
            torch.save(value, path)
            plan_path.write_text(json.dumps(plan))
            probe.write_text(PROBE)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            result = subprocess.run([runtime["primebot_python"], str(probe), str(runtime_path), str(shared_path),
                str(plan_path), str(path), str(output)], env=env, cwd=root, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            actual = json.loads(output.read_text())
            self.assertEqual(actual["plant_python"], runtime["primebot_python"])
            self.assertEqual(actual["plant_packages"]["mujoco"], "3.6.0")
            self.assertEqual(actual["policy_runtime"]["packages"]["mujoco"], "3.11.0")
            self.assertEqual(len(actual["cases"]), 6)
            self.assertTrue(actual["worker_reaped"])
            self.assertEqual(actual["worker_return_code"], 0)
            self.assertFalse(actual["cuda_initialized"] or actual["policy_runtime"]["cuda_initialized"])
            baseline = actual["policy_runtime"]["primebot_checkpoint"]["path"]
            robot, unused, _, _, _, _ = load_actor(baseline)
            del unused
            actor, _ = load_policy(path, spec, plan["code_commit"], expected_plan=plan)
            rows = {Path(row["q1_primebot"]["path"]).stem: row for row in json.loads(Path(shared["frozen_data"]).read_text())["rows"]}
            default, mask = spec.translator.to_canonical(torch.zeros(22)), actor.present
            comparisons = 0
            for case in actual["cases"]:
                row = rows[case["clip"]]
                library = MotionLibrary([row["q1_morph"]["path"]], device="cpu", robot=spec, human_root=shared["human_root"])
                history = ObsHistory(1, 10, 93, "cpu")
                for frame in case["frames"]:
                    state = native_state(robot, torch.tensor(frame["state"])[None], spec)
                    observation = analytic_obs(state, torch.zeros(29), default, state[:, 240:], spec.imu_slot)
                    observation = torch.cat((torch.where(mask, observation[:, :29], 0.),
                        torch.where(mask, observation[:, 29:58], 0.), observation[:, 58:64],
                        torch.where(mask, observation[:, 64:], 0.)), -1)
                    if frame["frame"] == 0:
                        history.reset(None, observation)
                    else:
                        history.push(observation)
                    indices = torch.tensor(reference_window_indices(frame["frame"], int(library.clip_len[0])))
                    goals = dict(goal=goal_features(state, library.state(indices)[None], anchor="reference"),
                        goal_kp=pack_human_goal(library.human_pos_w[indices][None],
                            library.human_quat_w[indices][None], split_state(state).body_quat[:, ANCHOR_IDX]))
                    action = actor(history.flat(), goals, case["interface"])
                    np.testing.assert_array_equal(action[0].numpy(), np.array(frame["canonical_action"], dtype=np.float32))
                    np.testing.assert_array_equal(spec.translator.to_native_delta(action)[0].numpy(),
                                                  np.array(frame["native_action"], dtype=np.float32))
                    comparisons += 1
            self.assertEqual(comparisons, 18)
            print("Q1_POLICY_BRIDGE_CPU_EVIDENCE=" + json.dumps(dict(
                plant_python=actual["plant_python"], plant_packages=actual["plant_packages"],
                policy_packages=actual["policy_runtime"]["packages"], synthetic_native_checkpoint=True,
                cases=6, action_comparisons=18, maximum_action_difference=0., physical_transitions=0,
                optimizer_updates=0, worker_reaped=True, worker_return_code=0, cuda_initialized=False)), flush=True)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
