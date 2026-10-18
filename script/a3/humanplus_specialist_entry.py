"""Independent native training and complete Isaac replay entry for zzx experts."""

import ast
import copy
import os
from pathlib import Path

from script.a3.checkpoint_store import atomic_json
from script.a3.humanplus_kit_log import binding, logged_tree
from script.a3.humanplus_specialist_protocol import (
    CHECKPOINTS, ENVS, UPDATES, require_training_receipt, validate_config,
)


def environment(original, config, device, args_cli):
    from script.a3 import humanplus_specialist_trial as trial
    from carb.settings import get_settings
    from isaaclab.sim.converters import UrdfConverter
    directory = trial.native_directory()
    atomic_json(directory / 'kit_binding.json', binding(directory, get_settings(), os.getpid()))
    training = os.environ['YUANQI_SPECIALIST_STAGE'] == 'train'
    if training:
        validate_config(config, trial.OUT / 'motionlib')
    else:
        from omegaconf import open_dict
        with open_dict(config.manager_env.config):
            config.manager_env.config.experiment_dir = str(trial.OUT / os.environ['YUANQI_SPECIALIST_STAGE'])
            config.manager_env.config.save_rendering_dir = str(trial.OUT / os.environ['YUANQI_SPECIALIST_STAGE'] / 'renderings')
    original_init = UrdfConverter.__init__
    expected = trial.VENDOR / config.manager_env.config.robot.asset_path
    converted = []

    def initialize(instance, cfg):
        if Path(cfg.asset_path).resolve() != expected.resolve():
            raise ValueError('Unregistered robot asset')
        local = copy.deepcopy(cfg)
        before = local.to_dict()
        local.usd_dir = str(trial.OUT / 'usd' / os.environ['YUANQI_SPECIALIST_STAGE'])
        local.usd_file_name = 'a3_035.usd'
        after = local.to_dict()
        for key in ('usd_dir', 'usd_file_name'):
            before.pop(key, None)
            after.pop(key, None)
        if before != after:
            raise ValueError('USD relocation changed physics')
        original_init(instance, local)
        converted.append(str(instance.usd_path))

    UrdfConverter.__init__ = initialize
    try:
        env = original(config, device, args_cli)
    finally:
        UrdfConverter.__init__ = original_init
    if len(converted) != 1 or env.num_envs != (ENVS if training else 1):
        raise ValueError('Unexpected environment or conversion count')
    atomic_json(directory / 'environment_created.json', dict(
        num_envs=env.num_envs, device=str(device), usd=converted, training=training))
    return env


def close(app, receipt):
    import torch
    from isaaclab.sim import SimulationContext
    from script.a3 import humanplus_specialist_trial as trial
    directory = trial.native_directory()
    torch.cuda.synchronize()
    atomic_json(directory / 'native_complete.json', dict(status='COMPLETE',
        native_pid=os.getpid(), cuda_synchronized=True, **receipt))
    SimulationContext.clear_instance()
    atomic_json(directory / 'close_begin.json', dict(native_pid=os.getpid(), callbacks_cleared=True))
    app.close()


def finish(app, trainer):
    from script.a3 import humanplus_specialist_trial as trial
    receipt = trial.read(trial.LOG / 'train/specialist_contract.json')
    require_training_receipt(receipt)
    trainer._assert_expert_frozen()
    if trainer.state.global_step != UPDATES:
        raise ValueError('Incomplete specialist training')
    for step in CHECKPOINTS:
        path = trial.checkpoint(step)
        identity = trial.read(path.with_suffix('.json'))
        if path.stat().st_size != identity['bytes']:
            raise ValueError('Missing specialist checkpoint')
    close(app, dict(training=receipt))


def evaluation_tree(source, filename, directory):
    tree = ast.parse(source, filename)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    counts = dict(environment=0, logging=0, loop=0, close=0, precision=0)
    for node in ast.walk(main):
        if isinstance(node, ast.Assign):
            target = ast.unparse(node.targets[0])
            if target == 'env' and isinstance(node.value, ast.Call) and ast.unparse(
                    node.value.func) == 'train_agent_trl.create_manager_env':
                node.value = ast.copy_location(ast.Call(func=ast.Name(id='humanplus_environment', ctx=ast.Load()),
                    args=[node.value.func, *node.value.args], keywords=node.value.keywords), node.value)
                counts['environment'] += 1
            elif target == 'base_kit_args':
                expected = '--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error'
                if not isinstance(node.value, ast.Constant) or node.value.value != expected:
                    raise ValueError('Pinned evaluation logging changed')
                node.value = ast.copy_location(ast.Constant(value=(
                    '--/log/level=info --/log/fileLogLevel=info --/log/outputStreamLevel=error '
                    '--/log/file=' + str(directory / 'kit.log'))), node.value)
                counts['logging'] += 1
            elif target in ('torch.backends.cuda.matmul.allow_tf32', 'torch.backends.cudnn.allow_tf32'):
                if not isinstance(node.value, ast.Constant) or node.value.value is not True:
                    raise ValueError('Pinned evaluation TF32 assignment changed')
                node.value = ast.copy_location(ast.Constant(value=False), node.value)
                counts['precision'] += 1
    for i, node in enumerate(main.body):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "config.get('run_eval_loop', True)":
            main.body[i] = ast.copy_location(ast.parse('specialist_replay(env, model, config)').body[0], node)
            counts['loop'] += 1
    tail = main.body[-1]
    expected = ast.parse('if simulator_type == "IsaacSim":\n    os._exit(0)').body[0]
    if ast.dump(tail, include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise ValueError('Pinned evaluation native exit changed')
    main.body[-1] = ast.copy_location(ast.parse('specialist_close(simulation_app)').body[0], tail)
    counts['close'] += 1
    if counts != dict(environment=1, logging=1, loop=1, close=1, precision=2):
        raise ValueError('Incomplete evaluation source adapter: ' + repr(counts))
    return ast.fix_missing_locations(tree)


def replay(env, model, config):
    import numpy as np
    import torch
    from script.a3 import humanplus_specialist_trial as trial
    base = env.env
    command = base.command_manager.get_term('motion')
    robot = base.scene['robot']
    env.set_is_evaluating(True)
    obs = env.reset_all()
    if (env.num_envs != 1 or command.motion_lib._num_motions != 1
            or command.motion_start_time_steps.item() != 0):
        raise ValueError('One complete zzx from its first frame required')
    model.eval()
    model.policy.init_rollout()
    # Retain termination flags, but continue physical state after a fall.
    # Automatic resets would turn a full-motion score into stitched episodes.
    original_reset = base._reset_idx
    suppressed = []
    base._reset_idx = lambda ids: suppressed.append(ids.detach().cpu().tolist())
    rows = []
    try:
        with torch.no_grad():
            for tick in range(499):
                if int(command.time_steps.item()) != tick or command.motion_start_time_steps.item() != 0:
                    raise ValueError('Reference skipped or reset during complete replay')
                ref_q = command.joint_pos[0].clone()
                ref_body = command.body_pos_w[0, command.body_indexes_data].clone()
                if len(command.body_joint_indices) != ref_q.numel():
                    raise ValueError('Reference joint ABI changed')
                actions = model.policy.rollout(obs_dict=obs)
                mean = model.policy.action_mean.detach()
                if not torch.isfinite(mean).all():
                    raise ValueError('Nonfinite Isaac action')
                values = env.step(dict(actions=mean, obs_dict=actions['obs_dict']))
                obs = values[0]
                q = robot.data.joint_pos[0, command.body_joint_indices]
                body = robot.data.body_pos_w[0, command.body_indexes].clone()
                quat = robot.data.root_quat_w[0]
                w, x, y, z = quat
                roll = torch.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
                pitch = torch.asin(torch.clamp(2 * (w*y-z*x), -1., 1.))
                rows.append(dict(tick=tick, q=q.cpu().tolist(), q_ref=ref_q.cpu().tolist(),
                    body=body.cpu().tolist(), body_ref=ref_body.cpu().tolist(),
                    height=float(robot.data.root_pos_w[0, 2] - base.scene.env_origins[0, 2]),
                    roll=float(roll * 180 / torch.pi), pitch=float(pitch * 180 / torch.pi),
                    terminated=bool(values[2].item()),
                    raw_action=mean[0].cpu().tolist()))
    finally:
        base._reset_idx = original_reset
    directory = trial.native_directory()
    path = directory / 'full_isaac_trace.json'
    atomic_json(path, dict(rows=rows, suppressed_reset_calls=suppressed,
        reference_body_names=list(command.cfg.body_names),
        actual_body_names=[robot.body_names[i] for i in command.body_indexes]))
    delta = np.asarray([r['q'] for r in rows]) - np.asarray([r['q_ref'] for r in rows])
    falls = [r['tick'] for r in rows if r['height'] < .45 or abs(r['roll']) > 60 or abs(r['pitch']) > 60]
    atomic_json(directory / 'isaac_metrics.json', dict(full_policy_steps=499,
        joint_rmse_rad=float(np.sqrt(np.mean(delta**2))), fall=bool(falls),
        first_fall_tick=falls[0] if falls else None,
        trace_sha256=trial.sha256(path), poststep_state_vs_precommand_reference=True,
        physical_continuation_after_fall=True, training_simulator=True,
        not_numerically_interchangeable_with_mujoco=True))


def execute(path, training):
    from script.a3 import humanplus_specialist_trial as trial
    namespace = dict(__name__='__main__', __file__=str(path), humanplus_environment=environment,
                     humanplus_finish=finish, specialist_replay=replay,
                     specialist_close=lambda app: close(app, dict(isaac=trial.read(
                         trial.native_directory() / 'isaac_metrics.json'))))
    tree = (logged_tree(path.read_text(), str(path), trial.native_directory()) if training else
            evaluation_tree(path.read_text(), str(path), trial.native_directory()))
    exec(compile(tree, str(path), 'exec'), namespace)
