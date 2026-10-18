"""In-memory storage and native-exit adapter for the pinned vendor trainer."""

import ast
import copy
import json
from pathlib import Path


def adapted_tree(source, filename):
    tree = ast.parse(source, filename=filename)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    tail = main.body[-1]
    expected = ast.parse('if simulator_type == "IsaacSim":\n    os._exit(0)').body[0]
    if ast.dump(tail, include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise ValueError('Pinned terminal native exit shape changed')
    main.body[-1] = ast.copy_location(ast.parse(
        'if simulator_type == "IsaacSim":\n    humanplus_finish(simulation_app, trainer)').body[0], tail)
    changed = dict(environment=0, logging=0)
    for node in ast.walk(main):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'env'
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == 'create_manager_env'):
            node.value = ast.copy_location(ast.Call(func=ast.Name(id='humanplus_environment', ctx=ast.Load()),
                args=[ast.Name(id='create_manager_env', ctx=ast.Load()), *node.value.args],
                keywords=node.value.keywords), node.value)
            changed['environment'] += 1
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and ast.unparse(node.targets[0]) == 'args_cli.kit_args'):
            if not isinstance(node.value, ast.Constant) or 'fileLogLevel=error' not in node.value.value:
                raise ValueError('Pinned Kit logging assignment changed')
            node.value.value = node.value.value.replace('fileLogLevel=error', 'fileLogLevel=info')
            changed['logging'] += 1
    if changed != dict(environment=1, logging=1):
        raise ValueError('Exactly one environment and logging adapter required')
    return ast.fix_missing_locations(tree)


def validate_training_receipts(directory):
    def read(name):
        return json.loads((directory / name).read_text())
    finite = read('finite_contract.json')
    routes = read('reference_route_coverage.json')['rollouts']
    cloth = read('cloth_contract.json')
    if (finite['proposals'] != 32 or finite['committed'] != 32 or len(finite['rows']) != 32
            or len(cloth['optimizer_steps']) != 32 or len(routes) != 8
            or [r['update'] for r in routes] != list(range(1, 9))
            or any(r['transitions'] != 384 or sum(r['routes'].values()) != 384 for r in routes)):
        raise ValueError('Native finish requires all eight updates and 32 commits')
    return dict(updates=8, optimizer_proposals=32, optimizer_commits=32,
                sampled_transitions=3072, finite_evaluations=finite['evaluations'])


def environment(original, config, device, args_cli):
    from script.a3 import humanplus_finite as trial
    from isaaclab.sim.converters import UrdfConverter
    original_init = UrdfConverter.__init__
    conversions = []
    expected = trial.VENDOR / config.manager_env.config.robot.asset_path

    def initialize(instance, cfg):
        if Path(cfg.asset_path).resolve() != expected.resolve():
            raise ValueError('Unregistered URDF conversion')
        local = copy.deepcopy(cfg)
        before = local.to_dict()
        local.usd_dir = str(trial.OUT / 'usd')
        local.usd_file_name = 'a3_035.usd'
        after = local.to_dict()
        for name in ('usd_dir', 'usd_file_name'):
            before.pop(name, None)
            after.pop(name, None)
        if before != after:
            raise ValueError('USD storage adapter changed physics')
        trial.event('urdf_conversion_begin', asset_path=str(expected), usd_dir=local.usd_dir)
        original_init(instance, local)
        conversions.append(str(instance.usd_path))
        trial.event('urdf_conversion_returned', usd_path=str(instance.usd_path))

    UrdfConverter.__init__ = initialize
    try:
        env = original(config, device, args_cli)
    finally:
        UrdfConverter.__init__ = original_init
    if len(conversions) != 1 or env.num_envs != 16:
        raise ValueError('Exactly one robot conversion and 16 environments required')
    trial.event('training_environment_created', num_envs=16, device=str(device), usd=conversions)
    return env


def finish(app, trainer):
    import torch
    from isaaclab.sim import SimulationContext
    from script.a3 import humanplus_finite as trial
    from script.a3.checkpoint_store import atomic_json
    if int(trainer.state.global_step) != 8 or trainer._optimizer_step_ordinal != 32:
        raise ValueError('Training returned before fixed budget')
    receipt = validate_training_receipts(trial.LOG / 'train')
    for step in (0, 2, 4, 8):
        path = trial.checkpoint(step)
        record = json.loads(path.with_suffix('.json').read_text())
        if path.stat().st_size != record['bytes']:
            raise ValueError('Missing durable checkpoint')
    torch.cuda.synchronize()
    atomic_json(trial.LOG / 'train/native_training.json', dict(status='TRAINING_COMPLETE',
        global_step=8, cuda_synchronized=True, **receipt))
    trial.event('training_complete', **receipt)
    SimulationContext.clear_instance()
    trial.event('context_clear_returned')
    trial.event('native_close_begin')
    app.close()
    trial.event('native_close_returned')


def execute(path):
    namespace = dict(__name__='__main__', __file__=str(path),
                     humanplus_environment=environment, humanplus_finish=finish)
    exec(compile(adapted_tree(path.read_text(), str(path)), str(path), 'exec'), namespace)
